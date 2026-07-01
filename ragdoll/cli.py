"""
RAGdoll CLI — local RAG memory for your dev tools.

Primary usage (no daemon needed):
  ragdoll index <path>              Index a file or directory
  ragdoll search <query>            Hybrid search (BM25 + vector)
  ragdoll forget <path>             Remove a file/directory from the index
  ragdoll list                      Show all indexed repos
  ragdoll stats                     Breakdown by repo + language
  ragdoll status                    Quick health check
  ragdoll remember "<note>"         Store a free-text memory note
  ragdoll memories                  List stored memory notes
  ragdoll context "<query>"         Pack top results into a token-budgeted block
  ragdoll hooks install             Install git hooks into a repo

Optional daemon (only needed for live MCP integration):
  ragdoll serve                     API server + file watcher
  ragdoll mcp                       MCP stdio server (called by Claude Code / Cursor)
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich import box

app = typer.Typer(
    name="ragdoll",
    help="Local RAG memory for your dev tools. No daemon required.",
    no_args_is_help=True,
)
console = Console()

import os


# Process-wide flags driven by top-level options (set in @app.callback).
# Subcommands consume them via _build_stack(). We use module state instead of
# threading typer Context everywhere because every command would otherwise need
# the same boilerplate, and these are pure read-only switches.
_FAST_MODE = False


def _default_db(fast: bool = False) -> Path:
    """Resolve the default DB path, respecting the RAGDOLL_DB env var.

    `fast=True` picks a separate file so 384-dim and 768-dim vectors never
    share a store (sqlite-vec would refuse, and silent corruption would be
    worse). Override either with RAGDOLL_DB.
    """
    env = os.environ.get("RAGDOLL_DB")
    if env:
        return Path(env).expanduser()
    name = "ragdoll-fast.db" if fast else "ragdoll.db"
    return Path.home() / ".ragdoll" / name


DEFAULT_DB   = _default_db()
DEFAULT_PORT = 7474


def _resolve_db(db: Optional[Path]) -> Path:
    """Resolve the DB path for a command.

    An explicit `--db` always wins. Otherwise fall back to DEFAULT_DB, which the
    top-level `--fast` callback has already pointed at the 384-dim fast store or
    the 768-dim default store. Commands bind `--db` with a None default (instead
    of capturing DEFAULT_DB at import time) so this runtime resolution sees the
    post-callback value — otherwise `--fast` would silently write fast-model
    vectors into the default store.
    """
    return db if db is not None else DEFAULT_DB


def _active_model_dim() -> int:
    """Embedding dimension for the active model (384 in --fast, else 768).

    A plain dict lookup — does not load the model — so it's cheap to call when
    opening a store for read-only commands.
    """
    from .embedder import DEFAULT_MODEL, FAST_MODEL, MODEL_DIMS
    return MODEL_DIMS[FAST_MODEL if _FAST_MODE else DEFAULT_MODEL]


def _open_store(db: Optional[Path]):
    """Open the vector store at the resolved DB path (honours --fast/--db)."""
    from .store import VectorStore
    return VectorStore(_resolve_db(db), dim=_active_model_dim())


def _build_stack(db_path: Optional[Path]):
    from .embedder import Embedder, DEFAULT_MODEL, FAST_MODEL
    from .indexer import Indexer
    from .store import VectorStore

    model = FAST_MODEL if _FAST_MODE else DEFAULT_MODEL
    embedder = Embedder(model_name=model)
    store    = VectorStore(_resolve_db(db_path), dim=embedder.dim)
    indexer  = Indexer(store, embedder)
    return store, embedder, indexer


def _daemon_base() -> str:
    """Base URL of a local RAGdoll daemon, honouring RAGDOLL_HOST/RAGDOLL_PORT."""
    host = os.environ.get("RAGDOLL_HOST", "127.0.0.1")
    port = int(os.environ.get("RAGDOLL_PORT", DEFAULT_PORT))
    return f"http://{host}:{port}"


def _daemon_search(query: str, top_k: int, repo, mode: str, db):
    """Route a search to a running daemon so its warm model is reused.

    Skips a ~500 MB cold model load (faster, and no second model in RAM).
    Returns a list of SearchResult on success, or None if no compatible daemon
    is reachable — the caller then falls back to loading the model locally.
    """
    import json
    import urllib.request
    from .store import SearchResult

    base = _daemon_base()
    want_db = os.path.abspath(_resolve_db(db))
    try:
        with urllib.request.urlopen(f"{base}/status", timeout=0.5) as resp:
            status = json.loads(resp.read().decode())
    except Exception:
        return None  # no daemon listening
    if status.get("status") != "ok":
        return None
    # Only trust the daemon if it serves the same DB we'd query locally,
    # otherwise --db / --fast would silently hit the wrong index.
    served = status.get("db")
    if served and os.path.abspath(served) != want_db:
        return None

    payload = json.dumps(
        {"query": query, "top_k": top_k, "repo": repo, "mode": mode}
    ).encode()
    req = urllib.request.Request(
        f"{base}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None  # daemon hiccup — fall back to local
    return [SearchResult(**r) for r in data.get("results", [])]


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is accepting TCP connections on host:port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _pid_alive(pid: int) -> bool:
    """True if the process still exists (signal 0 probes without killing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _listening_pids(port: int) -> list[int]:
    """PIDs listening on a TCP port. Tries lsof, then psutil; [] if neither works."""
    import subprocess

    try:
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(x) for x in out.stdout.split()]
        if pids:
            return sorted(set(pids))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass

    try:
        import psutil  # type: ignore

        pids = [
            c.pid
            for c in psutil.net_connections(kind="inet")
            if c.pid and c.laddr and c.laddr.port == port
            and c.status == psutil.CONN_LISTEN
        ]
        return sorted(set(pids))
    except Exception:
        return []


@app.callback()
def _root(
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Use the smaller bge-small (384-dim) model — ~3× faster, slightly weaker recall. "
             "Uses a separate DB (~/.ragdoll/ragdoll-fast.db) to avoid dim clash.",
    ),
):
    """Top-level options applied before any subcommand runs."""
    global _FAST_MODE, DEFAULT_DB
    _FAST_MODE = fast
    # Re-resolve the default DB path now that we know whether fast is on.
    # Subcommands using `db: Path = typer.Option(None, ...)` capture the
    # value at import time, so we also update the typer default *and* let
    # _build_stack pick it up via _default_db() if a command opts in.
    DEFAULT_DB = _default_db(fast=fast)


def _relative_time(iso: Optional[str]) -> str:
    if not iso:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso)
        delta = datetime.now(timezone.utc) - dt
        s = int(delta.total_seconds())
        if s < 60:      return "just now"
        if s < 3600:    return f"{s // 60}m ago"
        if s < 86400:   return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except ValueError:
        return iso


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def _ascii_bar(done: int, total: int, width: int = 32) -> str:
    """A plain ASCII progress bar for terminals that can't render rich output."""
    pct = (done / total) if total else 0.0
    filled = int(width * pct)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total} ({pct * 100:3.0f}%)"


@app.command()
def index(
    path: Path = typer.Argument(..., help="File or directory to index"),
    db:   Path = typer.Option(None, "--db", help="Path to SQLite DB"),
):
    """Index a file or directory into the local vector store.

    Progress display honours RAGDOLL_PROGRESS: auto (default — rich bar on a
    real terminal, ASCII bar otherwise), rich, plain (force ASCII), or none.
    """
    p = path.expanduser().resolve()
    if not p.exists():
        console.print(f"[red]Path not found: {p}[/red]")
        raise typer.Exit(1)
    from .indexer import LockBusy
    store, embedder, indexer = _build_stack(db)

    progress_mode = os.environ.get("RAGDOLL_PROGRESS", "auto").lower()
    use_rich = progress_mode == "rich" or (
        progress_mode == "auto" and console.is_terminal
    )
    show_progress = progress_mode != "none"

    try:
        if p.is_dir() and use_rich:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} files"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Indexing {p.name}", total=None)

                def on_progress(done: int, total: int) -> None:
                    progress.update(task, total=total, completed=done)

                n = indexer.index_path(p, progress=on_progress)
        elif p.is_dir():
            # Plain ASCII bar for terminals that don't render rich's live bar
            # (pipes, captured/agent shells, dumb terminals). Throttled so it
            # stays cheap; uses \r to update in place where supported.
            from time import monotonic

            on_progress = None
            if show_progress:
                print(f"Indexing {p.name} ...", flush=True)
                state = {"last": 0.0}

                def on_progress(done: int, total: int) -> None:  # noqa: F811
                    now = monotonic()
                    if not total:
                        return
                    if done >= total or now - state["last"] >= 0.5:
                        state["last"] = now
                        print(
                            f"\r{_ascii_bar(done, total)}", end="", flush=True
                        )
                        if done >= total:
                            print()

            n = indexer.index_path(p, progress=on_progress)
        else:
            console.print(f"Indexing [bold]{p}[/bold] ...")
            n = indexer.index_path(p)
    except LockBusy as e:
        # Another ragdoll process holds the single-inference lock. Fail fast
        # instead of loading a second ~500 MB model and risking OOM.
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        # Indexer's own SIGINT handler usually catches this first and stops
        # cooperatively; this is the belt-and-braces path for the rare case
        # where a signal slips through (e.g. between handler swaps).
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(130)

    if indexer._cancelled:
        console.print(f"[yellow]Cancelled — {n} chunks indexed before stop.[/yellow]")
        raise typer.Exit(130)
    console.print(f"[green]Done — {n} chunks indexed.[/green]")


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------

@app.command()
def dedupe(
    db: Path = typer.Option(None, "--db", help="Path to SQLite DB"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report only, don't delete"),
):
    """Remove duplicate-casing chunk rows (e.g. /RAGdoll vs /ragdoll on macOS).

    macOS's filesystem is case-insensitive but `Path.resolve()` keeps the
    casing the user typed. Older RAGdoll builds stored that exact string,
    so indexing the same repo with two different casings doubled every
    chunk row. New builds canonicalize on the way in; this command cleans
    up DBs created before that fix.
    """
    import os as _os
    import sqlite3 as _sql
    from .indexer import _canonical_path

    db = _resolve_db(db)
    con = _sql.connect(db)
    paths = [r[0] for r in con.execute("SELECT DISTINCT source_path FROM chunks")]

    # Group by inode. Files we can't stat (deleted/moved) are left alone.
    by_inode: dict[int, list[str]] = {}
    for p in paths:
        try:
            ino = _os.stat(p).st_ino
        except OSError:
            continue
        by_inode.setdefault(ino, []).append(p)

    to_delete: list[str] = []
    for ino, group in by_inode.items():
        if len(group) <= 1:
            continue
        # Keep the canonical (real on-disk casing) variant; drop the rest.
        canonical = str(_canonical_path(Path(group[0])))
        for p in group:
            if p != canonical:
                to_delete.append(p)

    if not to_delete:
        console.print("[green]No duplicate-casing rows found.[/green]")
        return

    console.print(f"Found [yellow]{len(to_delete)}[/yellow] duplicate source paths:")
    for p in to_delete[:10]:
        console.print(f"  [dim]{p}[/dim]")
    if len(to_delete) > 10:
        console.print(f"  [dim]... and {len(to_delete) - 10} more[/dim]")

    if dry_run:
        console.print("[yellow]--dry-run set, no changes made.[/yellow]")
        return

    placeholders = ",".join("?" * len(to_delete))
    n = con.execute(
        f"DELETE FROM chunks WHERE source_path IN ({placeholders})", to_delete
    ).rowcount
    con.commit()
    con.close()
    console.print(f"[green]Deleted {n} chunk rows under non-canonical paths.[/green]")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    query:   str           = typer.Argument(..., help="Natural language or code snippet"),
    top_k:   int           = typer.Option(8,    "--top-k",  "-k"),
    repo:    Optional[str] = typer.Option(None, "--repo",   "-r", help="Filter to a repo root"),
    mode:    str           = typer.Option("hybrid", "--mode", "-m",
                                help="Search mode: hybrid | vector | bm25"),
    no_mem:  bool          = typer.Option(False, "--no-memories",
                                help="Exclude memory notes from results"),
    db:      Path          = typer.Option(None, "--db"),
):
    """
    Semantic + keyword search over your indexed code and notes.

    Modes:
      hybrid  BM25 + vector via Reciprocal Rank Fusion (default, best overall)
      vector  Pure cosine similarity (better for vague / conceptual queries)
      bm25    Pure keyword (better for exact identifiers and error strings)
    """
    if mode not in ("hybrid", "vector", "bm25"):
        console.print("[red]--mode must be one of: hybrid, vector, bm25[/red]")
        raise typer.Exit(1)

    results = None
    # Prefer a running daemon: its model is already warm, so we skip a ~500 MB
    # cold load (faster, and no second model in RAM). The daemon API has no
    # --no-memories switch, so fall back to local when that's requested.
    if not no_mem:
        results = _daemon_search(query, top_k, repo, mode, db)

    if results is None:
        from .indexer import _index_lock, LockBusy
        store, embedder, _ = _build_stack(db)
        try:
            with _index_lock(purpose="search"):
                vec = embedder.embed_query(query)
                results = store.search(
                    query_vector=vec,
                    query_text=query,
                    top_k=top_k,
                    repo=repo,
                    mode=mode,           # type: ignore[arg-type]
                    include_memories=not no_mem,
                )
        except LockBusy as e:
            console.print(f"[yellow]{e}[/yellow]")
            raise typer.Exit(1)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Print in reverse rank order so the highest-scoring result lands at the
    # bottom of the terminal, right above the prompt — no scrolling needed.
    for rank, r in enumerate(reversed(results), start=1):
        display_rank = len(results) - rank + 1  # 1 = best
        label = (
            f"[dim italic]memory[/dim italic]"
            if r.language == "note"
            else f"[bold]{r.source_path}[/bold]  lines {r.start_line}–{r.end_line}"
        )
        console.rule(f"#{display_rank}  {label}  [dim]score={r.score:.4f}[/dim]")
        console.print(r.content)
        console.print()


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

@app.command()
def explain(
    query: str           = typer.Argument(..., help="Query to debug"),
    top_k: int           = typer.Option(8, "--top-k", "-k"),
    repo:  Optional[str] = typer.Option(None, "--repo", "-r"),
    db:    Path          = typer.Option(None, "--db"),
):
    """
    Show per-result scoring breakdown for a hybrid search.

    Useful for debugging why a chunk did or didn't surface. Columns:
      vec_rank   — position in pure vector (cosine) search
      bm25_rank  — position in pure BM25 keyword search
      rrf        — final Reciprocal Rank Fusion score (higher = better)
    """
    from .indexer import _index_lock, LockBusy
    store, embedder, _ = _build_stack(db)
    try:
        with _index_lock(purpose="search"):
            vec = embedder.embed_query(query)
            rows = store.explain(vec, query, top_k=top_k, repo=repo)
    except LockBusy as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)

    if not rows:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(box=box.SIMPLE_HEAVY, title=f"Explain: {query!r}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("source")
    table.add_column("lines", justify="right", style="dim")
    table.add_column("vec", justify="right")
    table.add_column("bm25", justify="right")
    table.add_column("rrf", justify="right", style="bold")

    for i, r in enumerate(rows, 1):
        vec_r = str(r["vector_rank"]) if r["vector_rank"] else "—"
        bm_r  = str(r["bm25_rank"]) if r["bm25_rank"] else "—"
        path  = r["source_path"]
        # Shorten home-prefixed paths
        home = str(Path.home())
        if path.startswith(home):
            path = "~" + path[len(home):]
        table.add_row(
            str(i), path,
            f"{r['start_line']}–{r['end_line']}",
            vec_r, bm_r,
            f"{r['rrf_score']:.4f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------

@app.command()
def forget(
    path: str  = typer.Argument(..., help="File, directory, or memory ID to remove"),
    db:   Path = typer.Option(None, "--db"),
):
    """Remove a file, directory, or memory note from the index."""
    from .store import VectorStore, MEMORY_PREFIX

    store = _open_store(db)

    # Memory note
    if path.startswith(MEMORY_PREFIX) or (len(path) == 16 and all(c in "0123456789abcdef" for c in path)):
        memory_id = path.replace(MEMORY_PREFIX, "")
        removed = store.delete_memory(memory_id)
        if removed:
            console.print(f"[green]Memory {memory_id} removed.[/green]")
        else:
            console.print(f"[yellow]Memory not found: {memory_id}[/yellow]")
        return

    p = Path(path).expanduser().resolve()
    if p.is_file():
        store.delete_by_source(str(p))
        console.print(f"[green]Removed {p}[/green]")
    else:
        n = store.delete_by_prefix(str(p))
        console.print(f"[green]Removed {n} chunks under {p}[/green]")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_repos(
    db: Path = typer.Option(None, "--db"),
):
    """List all indexed repos with chunk counts and last-indexed time."""
    from .store import VectorStore

    store = _open_store(db)
    repos = store.list_repos()

    if not repos:
        console.print("[yellow]Nothing indexed yet. Run: ragdoll index <path>[/yellow]")
        return

    table = Table(box=box.SIMPLE_HEAVY, show_footer=False)
    table.add_column("Repo",         style="bold", no_wrap=True)
    table.add_column("Chunks",       justify="right")
    table.add_column("Languages",    style="dim")
    table.add_column("Last indexed", style="dim")

    for r in repos:
        age = _relative_time(r.last_indexed)
        stale = " [yellow]⚠[/yellow]" if _is_stale(r.last_indexed) else ""
        table.add_row(
            r.repo,
            str(r.chunks),
            ", ".join(r.languages),
            f"{age}{stale}",
        )

    console.print(table)


def _is_stale(iso: Optional[str], threshold_days: int = 7) -> bool:
    if not iso:
        return False
    try:
        dt = datetime.fromisoformat(iso)
        return (datetime.now(timezone.utc) - dt).days >= threshold_days
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@app.command()
def stats(db: Path = typer.Option(None, "--db")):
    """Detailed breakdown of the index by repo and language."""
    from .store import VectorStore

    store = _open_store(db)
    rows  = store.stats_breakdown()

    if not rows:
        console.print("[yellow]Nothing indexed yet.[/yellow]")
        return

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Repo",         style="bold", no_wrap=True)
    table.add_column("Language",     style="cyan")
    table.add_column("Chunks",       justify="right")
    table.add_column("Last indexed", style="dim")

    current_repo = None
    for r in rows:
        repo_label = "" if r["repo"] == current_repo else r["repo"]
        current_repo = r["repo"]
        table.add_row(
            repo_label,
            r["language"],
            str(r["chunks"]),
            _relative_time(r["last_indexed"]),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(db: Path = typer.Option(None, "--db")):
    """Quick health check — DB size, repos, chunks, and embedding model."""
    from .store import VectorStore

    db = _resolve_db(db)
    store = _open_store(db)
    repos = store.list_repos()
    mems = store.list_memories()

    db_size_mb = db.stat().st_size / 1_048_576 if db.exists() else 0

    table = Table(box=box.SIMPLE_HEAVY, title="RAGdoll Status")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("DB path",          str(db))
    table.add_row("DB size",          f"{db_size_mb:.1f} MB")
    table.add_row("Repos indexed",    str(len(repos)))
    table.add_row("Total chunks",     str(store.count()))
    table.add_row("Memory notes",     str(len(mems)))
    table.add_row("Embedding model",  "nomic-ai/nomic-embed-text-v1.5 (768-dim, ONNX)")
    table.add_row("Backend",          "FastEmbed (ONNX Runtime)")

    # Hardware acceleration info
    from .embedder import _detect_providers, _default_threads
    from .indexer import BATCH_SIZE
    providers = _detect_providers() or ["CPUExecutionProvider"]
    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    accel = provider_names[0].replace("ExecutionProvider", "")
    table.add_row("Accelerator",      accel + (" + CPU fallback" if len(provider_names) > 1 else ""))
    table.add_row("ONNX threads",     str(_default_threads()))
    table.add_row("Batch size",       str(BATCH_SIZE))
    console.print(table)


# ---------------------------------------------------------------------------
# remember / memories
# ---------------------------------------------------------------------------

@app.command()
def remember(
    note: str            = typer.Argument(..., help="The note to store"),
    tags: Optional[str]  = typer.Option(None, "--tags", "-t",
                               help="Comma-separated tags, e.g. 'auth,decisions'"),
    db:   Path           = typer.Option(None, "--db"),
):
    """
    Store a free-text memory note, searchable alongside your code.

    Examples:
      ragdoll remember "we use JWT not sessions — mobile client can't do cookies"
      ragdoll remember "payments service owns the DB, never query it directly" --tags arch,payments
    """
    store, embedder, _ = _build_stack(db)
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    memory_id = store.add_memory(note, tags=tag_list, embedder=embedder)
    console.print(f"[green]Memory stored — ID: {memory_id}[/green]")
    console.print(f"[dim]Remove with: ragdoll forget {memory_id}[/dim]")


@app.command()
def memories(db: Path = typer.Option(None, "--db")):
    """List all stored memory notes."""
    from .store import VectorStore, MEMORY_PREFIX

    store = _open_store(db)
    mems  = store.list_memories()

    if not mems:
        console.print("[yellow]No memories stored yet. Try: ragdoll remember \"...[/yellow]")
        return

    for m in mems:
        memory_id = m["source_path"].replace(MEMORY_PREFIX, "")
        age = _relative_time(m["indexed_at"])
        console.rule(f"[bold]{memory_id}[/bold]  [dim]{age}[/dim]")
        console.print(m["content"])
        console.print()


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

@app.command()
def context(
    query:    str          = typer.Argument(..., help="What you need context for"),
    tokens:   int          = typer.Option(4000, "--tokens", "-t",
                                help="Approximate token budget"),
    mode:     str          = typer.Option("hybrid", "--mode", "-m"),
    repo:     Optional[str]= typer.Option(None, "--repo", "-r"),
    no_mem:   bool         = typer.Option(False, "--no-memories"),
    db:       Path         = typer.Option(None, "--db"),
):
    """
    Pack the most relevant chunks into a single context block within a token budget.
    Output is plain text — pipe it anywhere or paste into any LLM.

    Example:
      ragdoll context "auth flow" --tokens 4000 | pbcopy
      ragdoll context "rate limiting" --tokens 8000 --repo ~/my-project > context.txt
    """
    if mode not in ("hybrid", "vector", "bm25"):
        console.print("[red]--mode must be one of: hybrid, vector, bm25[/red]")
        raise typer.Exit(1)

    # Fetch generously — we'll trim to budget. Prefer the warm daemon, else
    # embed locally under the single-inference lock.
    results = None
    if not no_mem:
        results = _daemon_search(query, 50, repo, mode, db)
    if results is None:
        from .indexer import _index_lock, LockBusy
        store, embedder, _ = _build_stack(db)
        try:
            with _index_lock(purpose="search"):
                vec = embedder.embed_query(query)
                results = store.search(
                    query_vector=vec,
                    query_text=query,
                    top_k=50,
                    repo=repo,
                    mode=mode,           # type: ignore[arg-type]
                    include_memories=not no_mem,
                )
        except LockBusy as e:
            console.print(f"[yellow]{e}[/yellow]")
            raise typer.Exit(1)

    if not results:
        raise typer.Exit(0)

    # Deduplicate overlapping chunks (e.g. class + its methods from Python AST)
    from .store import VectorStore
    results = VectorStore.deduplicate(results)

    # Group by file — sort by (source_path, start_line) so contiguous chunks merge
    results.sort(key=lambda r: (r.source_path, r.start_line))

    def _estimate_tokens(text: str) -> int:
        """Word-count × 1.3 — more accurate than len/4 for code."""
        return math.ceil(len(text.split()) * 1.3)

    token_budget = tokens
    used_tokens = 0
    blocks: list[str] = []
    current_file: str | None = None

    for r in results:
        if r.language == "note":
            header = "# [memory]\n"
        elif r.source_path != current_file:
            # New file — full header
            header = f"# {r.source_path}  ({r.language})\n## lines {r.start_line}–{r.end_line}\n"
            current_file = r.source_path
        else:
            # Same file, additional chunk — lighter header
            header = f"## lines {r.start_line}–{r.end_line}\n"

        block = f"{header}```{r.language}\n{r.content}\n```"
        cost  = _estimate_tokens(block)
        if used_tokens + cost > token_budget and blocks:
            break
        blocks.append(block)
        used_tokens += cost

    output = "\n\n".join(blocks)
    # Print raw — meant to be piped/redirected
    print(output)

    # Show summary on stderr so it doesn't pollute the piped output
    print(
        f"\n[ragdoll] {len(blocks)} chunks, ~{used_tokens} tokens "
        f"(budget: {tokens})",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@app.command()
def doctor(
    db:   Path = typer.Option(None, "--db"),
    port: int  = typer.Option(DEFAULT_PORT, "--port", "-p"),
):
    """
    Diagnose a broken or half-configured RAGdoll install.

    Checks:
      - DB file exists and is readable
      - FTS schema is current
      - Embedding model is loadable
      - Port 7474 is free or daemon is responding
      - launchd agent is installed (macOS)
    """
    import socket
    from .store import VectorStore, FTS_SCHEMA_VERSION
    from .embedder import DEFAULT_MODEL

    checks: list[tuple[str, bool, str]] = []  # (label, ok, detail)

    db = _resolve_db(db)
    # 1. DB path
    if db.exists():
        size_mb = db.stat().st_size / 1_048_576
        checks.append(("DB file", True, f"{db} ({size_mb:.1f} MB)"))
    else:
        checks.append(("DB file", False, f"{db} does not exist — run 'ragdoll index <path>'"))

    # 2. DB readable + FTS version
    try:
        store = _open_store(db)
        chunks = store.count()
        fts_v = store.get_meta("fts_schema_version")
        if fts_v == FTS_SCHEMA_VERSION:
            checks.append(("FTS schema", True, f"v{fts_v} ({chunks} chunks)"))
        else:
            checks.append((
                "FTS schema", False,
                f"v{fts_v or '?'} — expected v{FTS_SCHEMA_VERSION}. "
                f"Re-open any RAGdoll command to auto-migrate.",
            ))
    except Exception as exc:
        checks.append(("DB readable", False, str(exc)))

    # 3. Embedding model
    try:
        from .embedder import Embedder
        emb = Embedder()
        emb._load_model()
        checks.append(("Embedding model", True, f"{emb.model_name} loaded"))
    except Exception as exc:
        checks.append(("Embedding model", False, f"failed to load: {exc}"))

    # 4. Daemon / port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            checks.append(("Daemon", True, f"responding on :{port}"))
        except (ConnectionRefusedError, socket.timeout):
            checks.append(("Daemon", True, f":{port} free — direct CLI mode OK"))
        except OSError as exc:
            checks.append(("Daemon", False, f"port check failed: {exc}"))

    # 5. launchd agent (macOS only) — check both presence AND loaded state
    if sys.platform == "darwin":
        import subprocess
        plist = Path.home() / "Library" / "LaunchAgents" / "com.ragdoll.daemon.plist"
        if not plist.exists():
            checks.append((
                "launchd agent", True,
                "not installed — run 'ragdoll autostart install' to enable",
            ))
        else:
            res = subprocess.run(
                ["launchctl", "list", "com.ragdoll.daemon"],
                capture_output=True, text=True,
            )
            if res.returncode == 0:
                checks.append(("launchd agent", True, f"loaded ({plist})"))
            else:
                checks.append((
                    "launchd agent", False,
                    f"plist present but not loaded — try: launchctl load {plist}",
                ))

    # 6. Disk space — warn if <500 MB free under ~/.ragdoll
    try:
        import shutil as _shutil
        target = db.parent if db.parent.exists() else Path.home()
        free_mb = _shutil.disk_usage(target).free / 1_048_576
        if free_mb < 500:
            checks.append((
                "Disk space", False,
                f"{free_mb:.0f} MB free under {target} (need >500 MB for safe indexing)",
            ))
        else:
            checks.append(("Disk space", True, f"{free_mb:.0f} MB free under {target}"))
    except Exception as exc:
        checks.append(("Disk space", True, f"check skipped: {exc}"))

    # 7. Embedding dim recorded vs expected
    try:
        expected = _active_model_dim()
        recorded = store.get_meta("embed_dim")
        if recorded and int(recorded) != expected:
            checks.append((
                "Embed dim", False,
                f"DB recorded dim={recorded}, current model expects {expected}. "
                f"Run 'ragdoll reindex' (or re-run with/without --fast to match).",
            ))
        elif recorded:
            checks.append(("Embed dim", True, f"{recorded} (matches build)"))
    except Exception:
        pass  # store may not have been opened

    # 8. Partial index check — detect files where chunk count in DB doesn't
    #    match what the chunker would produce (from a killed/crashed index run)
    try:
        from .chunker import chunk_file
        db_counts = store.chunk_counts_by_file()
        partial: list[str] = []
        missing: list[str] = []
        sampled = 0
        for source_path, db_count in db_counts.items():
            p = Path(source_path)
            if not p.exists():
                missing.append(source_path)
                continue
            # Sample up to 50 files to keep doctor fast
            if sampled >= 50:
                break
            expected = len(chunk_file(p))
            if expected and db_count != expected:
                partial.append(f"{p.name} ({db_count}/{expected} chunks)")
            sampled += 1

        if missing:
            checks.append((
                "Deleted files", False,
                f"{len(missing)} indexed file(s) no longer on disk. "
                f"Run 'ragdoll forget <path>' or reindex to clean up.",
            ))
        elif len(missing) == 0:
            checks.append(("Deleted files", True, "all indexed files exist on disk"))

        if partial:
            checks.append((
                "Partial indexes", False,
                f"{len(partial)} file(s) have incomplete chunks "
                f"(likely from a killed index run): {', '.join(partial[:5])}"
                f"{f' +{len(partial)-5} more' if len(partial) > 5 else ''}. "
                f"Run 'ragdoll index <repo>' to repair.",
            ))
        else:
            checks.append(("Partial indexes", True, f"checked {sampled} files, all complete"))
    except Exception as exc:
        checks.append(("Partial indexes", True, f"check skipped: {exc}"))

    # Render
    table = Table(box=box.SIMPLE_HEAVY, title="RAGdoll Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", style="dim")
    all_ok = True
    for label, ok, detail in checks:
        mark = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        if not ok:
            all_ok = False
        table.add_row(label, mark, detail)
    console.print(table)
    if not all_ok:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------

@app.command()
def reindex(
    db: Path = typer.Option(None, "--db"),
):
    """
    Re-embed all indexed chunks with the current model.

    Use this after switching embedding models, or if 'ragdoll status' reports a
    model mismatch. Deletes all vectors and re-indexes every known repo.
    """
    from .store import VectorStore, MEMORY_REPO

    store = _open_store(db)
    repos = store.list_repos()
    if not repos:
        console.print("[yellow]Nothing to reindex.[/yellow]")
        return

    # Clear model tracking so check_embed_model records the new model
    store.set_meta("embed_model", "")
    store.set_meta("embed_dim", "")

    _, embedder, indexer = _build_stack(db)

    # Purge all non-memory chunks and re-index from disk
    total_chunks = 0
    for r in repos:
        repo_path = Path(r.repo)
        if not repo_path.exists():
            console.print(f"[yellow]Skipping (not found): {r.repo}[/yellow]")
            continue
        store.delete_by_prefix(r.repo)
        console.print(f"Re-indexing [bold]{r.repo}[/bold] ...")
        n = indexer.index_path(repo_path)
        total_chunks += n
        console.print(f"  {n} chunks")

    console.print(f"[green]Done — {total_chunks} total chunks re-indexed.[/green]")


# ---------------------------------------------------------------------------
# export / import (JSONL)
# ---------------------------------------------------------------------------

@app.command(name="export")
def export_cmd(
    out: Path = typer.Argument(..., help="Output JSONL file (use '-' for stdout)"),
    db:  Path = typer.Option(None, "--db"),
):
    """
    Dump all indexed chunks (with vectors) to a JSONL file.

    Useful for backups or sharing a prebuilt index with a teammate.
    Each line is a JSON object with content, metadata, and embedding.
    """
    import json
    from .store import VectorStore

    store = _open_store(db)
    if str(out) == "-":
        fh = sys.stdout
        close_after = False
    else:
        fh = open(out, "w")
        close_after = True

    n = 0
    try:
        for chunk in store.iter_all_chunks():
            fh.write(json.dumps(chunk, separators=(",", ":")) + "\n")
            n += 1
    finally:
        if close_after:
            fh.close()

    if close_after:
        console.print(f"[green]Exported {n} chunks to {out}[/green]")


@app.command(name="import")
def import_cmd(
    src: Path = typer.Argument(..., help="JSONL file produced by 'ragdoll export'"),
    db:  Path = typer.Option(None, "--db"),
    replace: bool = typer.Option(
        False, "--replace",
        help="Wipe the DB before importing (default: merge)",
    ),
):
    """
    Load chunks from a JSONL file into the local DB.

    The source file is expected to be produced by 'ragdoll export'.
    Vectors are reused as-is — the embedding model must match the one this
    DB is bound to. We sniff the first row to verify the dimension and
    refuse the import on mismatch (rather than silently corrupting search
    quality).
    """
    import json
    from .embedder import Embedder, DEFAULT_MODEL, FAST_MODEL
    from .store import VectorStore

    # Use the active embedder's expected dim (honors --fast). Building a bare
    # Embedder is cheap — model isn't loaded until we actually embed.
    expected_dim = Embedder(FAST_MODEL if _FAST_MODE else DEFAULT_MODEL).dim
    store = _open_store(db)

    # --- preflight: dim check off the first row ----------------------------
    first_vec_dim: int | None = None
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                console.print(f"[red]Malformed JSONL on first non-empty line: {exc}[/red]")
                raise typer.Exit(1)
            vec = row.get("vector")
            if isinstance(vec, list):
                first_vec_dim = len(vec)
            break

    if first_vec_dim is not None and first_vec_dim != expected_dim:
        console.print(
            f"[red]Vector dim mismatch: import file has dim={first_vec_dim}, "
            f"this build expects dim={expected_dim}.[/red]\n"
            f"[yellow]Re-export from the source machine with the matching "
            f"embedding model, or reindex from source code instead.[/yellow]"
        )
        raise typer.Exit(2)

    # Cross-check the recorded model name if both sides have one
    db_model = store.get_meta("embed_model")
    file_model = None
    # Look at row metadata if present (older exports may not have it)
    if isinstance(row, dict):
        file_model = row.get("embed_model")
    if db_model and file_model and db_model != file_model:
        console.print(
            f"[red]Model mismatch: DB={db_model!r}, file={file_model!r}.[/red]\n"
            f"[yellow]Run `ragdoll reindex` after import or import into a "
            f"fresh DB.[/yellow]"
        )
        raise typer.Exit(2)

    if replace:
        for r in store.list_repos():
            store.delete_by_prefix(r.repo)

    batch: list[dict] = []
    n = 0
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            batch.append(row)
            if len(batch) >= 500:
                store.upsert(batch)
                n += len(batch)
                batch = []
    if batch:
        store.upsert(batch)
        n += len(batch)

    console.print(f"[green]Imported {n} chunks from {src}[/green]")


# ---------------------------------------------------------------------------
# autostart (launchd, macOS)
# ---------------------------------------------------------------------------

autostart_app = typer.Typer(help="Manage the RAGdoll daemon as a background service.")
app.add_typer(autostart_app, name="autostart")

PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ragdoll.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{ragdoll_bin}</string>
        <string>serve</string>
{watch_args}    </array>
    <key>RunAtLoad</key>
    <true/>
    <!-- Restart only on abnormal termination, not on clean exit. Avoids a
         crash-loop if the user runs `ragdoll serve` interactively too. -->
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    <!-- Don't relaunch faster than every 30s — protects against tight crash
         loops from a bad config / port-in-use error. -->
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>StandardOutPath</key>
    <string>{log_dir}/ragdoll.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/ragdoll.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path}</string>
{env_extra}    </dict>
</dict>
</plist>
"""


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "com.ragdoll.daemon.plist"


@autostart_app.command("install")
def autostart_install(
    watch: list[Path] = typer.Option([], "--watch", "-w",
                           help="Directories the daemon should watch (repeatable)"),
):
    """Install a launchd agent so the daemon starts on login (macOS only)."""
    import shutil

    if sys.platform != "darwin":
        console.print("[red]autostart is currently macOS-only (uses launchd).[/red]")
        raise typer.Exit(1)

    ragdoll_bin = shutil.which("ragdoll")
    if not ragdoll_bin:
        console.print(
            "[red]`ragdoll` not found on PATH. "
            "Activate your venv or reinstall the package before running this.[/red]"
        )
        raise typer.Exit(1)

    watch_args = "".join(
        f"        <string>--watch</string>\n"
        f"        <string>{Path(w).expanduser().resolve()}</string>\n"
        for w in watch
    )
    log_dir = Path.home() / ".ragdoll" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build a sane PATH that doesn't depend on the install-time shell. Capturing
    # os.environ['PATH'] at install time bakes in venv paths that disappear at
    # login, so prefer well-known system bins plus the bin dir of the wrapper.
    bin_dir = str(Path(ragdoll_bin).parent)
    path_parts = [bin_dir, "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    seen = set()
    path_clean = ":".join(p for p in path_parts if not (p in seen or seen.add(p)))

    # If the user is on a multi-profile setup, propagate RAGDOLL_DB so the
    # daemon serves from the same DB as their interactive shell.
    env_extra = ""
    if os.environ.get("RAGDOLL_DB"):
        env_extra = (
            "        <key>RAGDOLL_DB</key>\n"
            f"        <string>{os.environ['RAGDOLL_DB']}</string>\n"
        )

    plist_path = _plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(PLIST_TEMPLATE.format(
        ragdoll_bin=ragdoll_bin,
        watch_args=watch_args,
        log_dir=log_dir,
        path=path_clean,
        env_extra=env_extra,
    ))

    import subprocess
    # Unload first in case it's already loaded (ignore errors)
    subprocess.run(["launchctl", "unload", str(plist_path)],
                   capture_output=True, check=False)
    result = subprocess.run(
        ["launchctl", "load", str(plist_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]launchctl load failed: {result.stderr}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Installed {plist_path}[/green]")
    console.print(f"[dim]Logs: {log_dir}/ragdoll.log[/dim]")


@autostart_app.command("uninstall")
def autostart_uninstall():
    """Stop and remove the launchd agent."""
    if sys.platform != "darwin":
        console.print("[red]autostart is currently macOS-only.[/red]")
        raise typer.Exit(1)

    plist_path = _plist_path()
    if not plist_path.exists():
        console.print("[yellow]No launchd agent installed.[/yellow]")
        return

    import subprocess
    subprocess.run(["launchctl", "unload", str(plist_path)],
                   capture_output=True, check=False)
    plist_path.unlink()
    console.print(f"[green]Removed {plist_path}[/green]")


@autostart_app.command("status")
def autostart_status():
    """Check whether the launchd agent is loaded."""
    if sys.platform != "darwin":
        console.print("[red]autostart is currently macOS-only.[/red]")
        raise typer.Exit(1)

    import subprocess
    result = subprocess.run(
        ["launchctl", "list", "com.ragdoll.daemon"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("[green]RAGdoll launchd agent is loaded.[/green]")
        console.print(f"[dim]{result.stdout.strip()}[/dim]")
    else:
        console.print("[yellow]RAGdoll launchd agent is not loaded.[/yellow]")


# ---------------------------------------------------------------------------
# git hooks
# ---------------------------------------------------------------------------

hooks_app = typer.Typer(help="Manage git hooks for auto-indexing.")
app.add_typer(hooks_app, name="hooks")

# Markers let the uninstaller surgically remove our block from a hook file
# that the user has appended their own commands to, without nuking those.
HOOK_BLOCK_START = "# >>> ragdoll hook >>>"
HOOK_BLOCK_END   = "# <<< ragdoll hook <<<"


def _hook_registry_path() -> Path:
    """File listing every repo we've installed hooks into.

    Used by uninstall.sh to clean up reliably across all repos rather than
    relying on the user's CWD at uninstall time.
    """
    return Path.home() / ".ragdoll" / "hook_registry"


def _record_hook_repo(repo_path: Path) -> None:
    reg = _hook_registry_path()
    reg.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if reg.exists():
        existing = {line.strip() for line in reg.read_text().splitlines() if line.strip()}
    existing.add(str(repo_path))
    reg.write_text("\n".join(sorted(existing)) + "\n")


def _unrecord_hook_repo(repo_path: Path) -> None:
    reg = _hook_registry_path()
    if not reg.exists():
        return
    keep = [
        line for line in reg.read_text().splitlines()
        if line.strip() and line.strip() != str(repo_path)
    ]
    if keep:
        reg.write_text("\n".join(keep) + "\n")
    else:
        reg.unlink()


def _hook_block(db_override: str | None) -> str:
    """Build the bracketed hook block.

    We capture RAGDOLL_DB at install time so a user with multiple profiles
    (e.g. RAGDOLL_DB=~/.ragdoll/work.db) gets hooks that write to the right
    DB. If no override is set we omit --db so the runtime default applies.
    """
    db_arg = f' --db "{db_override}"' if db_override else ""
    return (
        f"{HOOK_BLOCK_START}\n"
        f"# RAGdoll — auto-index on checkout/merge (managed; do not edit between markers)\n"
        f'ragdoll index .{db_arg} >/dev/null 2>&1 || true\n'
        f"{HOOK_BLOCK_END}\n"
    )


def _write_hook(hook_path: Path, block: str) -> str:
    """Write or update a single hook file. Returns 'installed' | 'updated' | 'appended'."""
    if not hook_path.exists():
        hook_path.write_text("#!/bin/sh\n" + block)
        hook_path.chmod(0o755)
        return "installed"

    existing = hook_path.read_text()
    if HOOK_BLOCK_START in existing and HOOK_BLOCK_END in existing:
        # Replace the bracketed block in-place
        import re
        new = re.sub(
            re.escape(HOOK_BLOCK_START) + r".*?" + re.escape(HOOK_BLOCK_END) + r"\n?",
            block,
            existing,
            flags=re.DOTALL,
        )
        hook_path.write_text(new)
        hook_path.chmod(0o755)
        return "updated"

    # User-owned hook — append our block, preserve theirs.
    hook_path.write_text(existing.rstrip() + "\n\n" + block)
    hook_path.chmod(0o755)
    return "appended"


@hooks_app.command("install")
def hooks_install(
    repo: Path = typer.Argument(Path("."), help="Git repo root (default: cwd)"),
):
    """Install post-checkout and post-merge hooks to auto-index on git operations."""
    repo_root = repo.expanduser().resolve()
    hooks_dir = repo_root / ".git" / "hooks"
    if not hooks_dir.exists():
        console.print(f"[red]Not a git repo: {repo}[/red]")
        raise typer.Exit(1)

    # Capture RAGDOLL_DB now so the hook always writes to the intended profile
    # even when the user runs git in a shell where the env isn't set.
    db_override = os.environ.get("RAGDOLL_DB")
    block = _hook_block(db_override)

    for hook in ("post-checkout", "post-merge"):
        hook_path = hooks_dir / hook
        try:
            action = _write_hook(hook_path, block)
        except PermissionError:
            console.print(f"[red]Permission denied writing {hook_path}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]{action.capitalize()} {hook_path}[/green]")

    _record_hook_repo(repo_root)
    if db_override:
        console.print(f"[dim]Hook will write to: {db_override}[/dim]")


@hooks_app.command("uninstall")
def hooks_uninstall(
    repo: Path = typer.Argument(Path("."), help="Git repo root (default: cwd)"),
):
    """Remove RAGdoll git hooks. Preserves any non-RAGdoll content in the file."""
    import re
    repo_root = repo.expanduser().resolve()
    hooks_dir = repo_root / ".git" / "hooks"
    for hook in ("post-checkout", "post-merge"):
        hook_path = hooks_dir / hook
        if not hook_path.exists():
            continue
        existing = hook_path.read_text()
        if HOOK_BLOCK_START in existing and HOOK_BLOCK_END in existing:
            stripped = re.sub(
                re.escape(HOOK_BLOCK_START) + r".*?" + re.escape(HOOK_BLOCK_END) + r"\n?",
                "",
                existing,
                flags=re.DOTALL,
            ).rstrip() + "\n"
            # If nothing meaningful left (just shebang / blank lines), delete.
            meaningful = [
                line for line in stripped.splitlines()
                if line.strip() and not line.startswith("#!")
            ]
            if meaningful:
                hook_path.write_text(stripped)
                console.print(f"[green]Removed RAGdoll block from {hook_path}[/green]")
            else:
                hook_path.unlink()
                console.print(f"[green]Removed {hook_path}[/green]")
        elif "RAGdoll" in existing:
            # Pre-marker legacy install — safe to remove the whole file.
            hook_path.unlink()
            console.print(f"[green]Removed legacy {hook_path}[/green]")

    _unrecord_hook_repo(repo_root)


# ---------------------------------------------------------------------------
# serve  (optional daemon)
# ---------------------------------------------------------------------------

@app.command()
def serve(
    watch:          list[Path] = typer.Option([], "--watch", "-w",
                                     help="Directories to watch for changes"),
    db:             Path       = typer.Option(None, "--db"),
    port:           int        = typer.Option(DEFAULT_PORT, "--port", "-p"),
    index_on_start: bool       = typer.Option(True, "--index/--no-index"),
):
    """
    (Optional) Start the RAGdoll daemon: API server + live file watcher.

    Only needed for real-time MCP with Claude Code / Cursor.
    For everything else use 'ragdoll index' + 'ragdoll search' directly.
    """
    logging.basicConfig(level=logging.INFO)
    store, embedder, indexer = _build_stack(db)

    from . import api
    api.init(indexer, embedder, store)

    watch_paths = [Path(w).expanduser().resolve() for w in watch]

    if watch_paths and index_on_start:
        console.print(f"[bold]Indexing {len(watch_paths)} path(s) on startup...[/bold]")
        for p in watch_paths:
            n = indexer.index_path(p)
            console.print(f"  {p}: {n} chunks")

    if watch_paths:
        from .watcher import Watcher

        def on_change(path: Path, event_type: str) -> None:
            if event_type == "deleted":
                indexer.remove_path(path)
            else:
                indexer.index_path(path)

        watcher = Watcher(watch_paths, on_change)
        watcher.start()
        console.print(f"[green]Watching {len(watch_paths)} path(s)[/green]")

    console.print(f"[bold green]RAGdoll daemon on http://localhost:{port}[/bold green]")
    try:
        uvicorn.run(api.app, host="127.0.0.1", port=port, log_level="warning")
    except OSError as exc:
        console.print(f"[red]Failed to start on port {port}: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def stop(
    port:  Optional[int] = typer.Option(None, "--port", "-p",
                               help="Daemon port (default: $RAGDOLL_PORT or 7474)"),
    force: bool          = typer.Option(False, "--force", "-f",
                               help="SIGKILL immediately instead of graceful SIGTERM"),
):
    """
    Stop a running RAGdoll daemon (one started by 'ragdoll serve').

    Finds the process listening on the daemon port and terminates it. For the
    login-persistent launchd daemon, use 'ragdoll autostart uninstall' instead.
    """
    import signal
    import time

    if port is None:
        port = int(os.environ.get("RAGDOLL_PORT", DEFAULT_PORT))

    pids = _listening_pids(port)
    if not pids:
        if _port_open(port):
            console.print(
                f"[yellow]Something is listening on port {port}, but I couldn't "
                f"resolve its PID (need lsof or psutil). Stop it manually:[/yellow]\n"
                f"  lsof -ti :{port} | xargs kill"
            )
            raise typer.Exit(1)
        console.print(f"[yellow]No RAGdoll daemon listening on port {port}.[/yellow]")
        raise typer.Exit(0)

    for pid in pids:
        try:
            if force:
                os.kill(pid, signal.SIGKILL)
                console.print(f"[green]Killed daemon (pid {pid}).[/green]")
                continue
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):  # wait up to ~5s for a clean exit
                time.sleep(0.1)
                if not _pid_alive(pid):
                    break
            if _pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
                console.print(f"[green]Stopped daemon (pid {pid}, forced).[/green]")
            else:
                console.print(f"[green]Stopped daemon (pid {pid}).[/green]")
        except ProcessLookupError:
            console.print(f"[yellow]Process {pid} already gone.[/yellow]")
        except PermissionError:
            console.print(f"[red]No permission to stop pid {pid}.[/red]")
            raise typer.Exit(1)


# ---------------------------------------------------------------------------
# mcp  (stdio server for Claude Code / Cursor)
# ---------------------------------------------------------------------------

@app.command()
def mcp():
    """
    Start the MCP stdio server (invoked by Claude Code / Cursor, not by humans).

    Add to ~/.claude/settings.json:
      "mcpServers": { "ragdoll": { "command": "ragdoll", "args": ["mcp"] } }
    """
    from .mcp_server import run
    asyncio.run(run())


if __name__ == "__main__":
    app()
