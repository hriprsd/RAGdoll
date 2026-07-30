"""
Indexer - walks directories, chunks files, embeds, and stores.
Handles incremental updates by comparing content hashes.

Memory safety:
  - A file-based lock prevents multiple `ragdoll index` processes from
    running ONNX inference simultaneously (the main source of OOM on
    machines with <= 24 GB RAM).
  - Batch size adapts to available memory: smaller batches on constrained
    machines, larger on well-provisioned ones.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import git

from .chunker import chunk_file, SKIP_DIRS
from .embedder import Embedder
from .store import VectorStore

logger = logging.getLogger(__name__)

def _embed_throttle() -> float:
    """Seconds to sleep after each embed batch (from RAGDOLL_THROTTLE_MS).

    Lets a big index stay cool and the machine responsive by capping the embed
    duty cycle, at the cost of wall-clock time. 0 (default) = full speed.
    """
    try:
        return max(0.0, float(os.environ.get("RAGDOLL_THROTTLE_MS", "0")) / 1000.0)
    except ValueError:
        return 0.0


# --- Adaptive batch sizing --------------------------------------------------
# ONNX working-set per batch is roughly: batch_size * max_chunk_chars * 4 bytes
# (tokenizer + attention). We pick batch size based on available RAM so a
# single ragdoll process stays well under pressure, even when the user has
# Chrome, Cursor, Slack, and Zoom open.
#
# Override with RAGDOLL_BATCH_SIZE env var.

_DEFAULT_BATCH_SIZES = {
    "low":    16,   # <= 8 GB RAM
    "medium": 32,   # 8-16 GB
    "high":   64,   # 16-32 GB
    "ultra":  128,  # > 32 GB
}


def _adaptive_batch_size() -> int:
    """Pick a batch size based on total system RAM.

    Conservative: we assume the user is running other apps. A single
    batch of 64 chunks at 8000 chars each uses ~200 MB of ONNX working
    memory. That's fine on 16+ GB machines but can push a 8 GB machine
    into swap when combined with editors, browsers, etc.
    """
    env = os.environ.get("RAGDOLL_BATCH_SIZE")
    if env:
        return int(env)

    try:
        import platform
        if platform.system() == "Darwin":
            # macOS: sysctl is reliable and fast
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, timeout=2,
            ).strip()
            total_bytes = int(out)
        else:
            # Linux: /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_bytes = int(line.split()[1]) * 1024
                        break
                else:
                    total_bytes = 0
    except Exception:
        return _DEFAULT_BATCH_SIZES["medium"]  # safe default

    gb = total_bytes / (1024 ** 3)
    if gb <= 8:
        return _DEFAULT_BATCH_SIZES["low"]
    elif gb <= 16:
        return _DEFAULT_BATCH_SIZES["medium"]
    elif gb <= 32:
        return _DEFAULT_BATCH_SIZES["high"]
    else:
        return _DEFAULT_BATCH_SIZES["ultra"]


BATCH_SIZE = _adaptive_batch_size()

# Pipeline tuning (directory indexing). A producer thread reads + chunks files
# (disk I/O + CPU) while the main thread embeds (ONNX releases the GIL during
# inference) and writes to SQLite. Chunks are batched across files so the model
# always gets full batches instead of one underfull batch per small file.
#   - PLAN_QUEUE_MAX: how many planned files may wait ahead of the embedder.
#     Bounded so a fast producer can't read the whole repo into RAM.
#   - ROW_FLUSH: upsert staged rows once this many accumulate (keeps the write
#     amortised and memory flat).
PLAN_QUEUE_MAX = int(os.environ.get("RAGDOLL_PLAN_QUEUE", 16))
ROW_FLUSH = int(os.environ.get("RAGDOLL_ROW_FLUSH", 512))

# How many pending chunks to accumulate before handing them to the embedder.
# The embedder length-sorts each call to minimise padding waste (onnxruntime
# pads every sequence in a batch to the longest), so a bigger pool means each
# padded sub-batch is tighter. Buffering chunks is cheap (~700 chars each), the
# producer keeps refilling it, so a window well above BATCH_SIZE is a free win.
# Tunable via RAGDOLL_EMBED_WINDOW.
EMBED_WINDOW = int(os.environ.get("RAGDOLL_EMBED_WINDOW", max(BATCH_SIZE * 8, 1024)))

# --- Process lock ------------------------------------------------------------
# Prevents multiple `ragdoll index` invocations from loading ONNX models
# simultaneously. Each ONNX model load allocates ~500 MB; 5 parallel
# processes = ~2.5 GB just for model weights, plus per-batch working memory.
# On a 24 GB machine with normal app usage, that's a guaranteed OOM.

_LOCK_PATH = Path(os.environ.get("RAGDOLL_HOME", Path.home() / ".ragdoll")) / ".index.lock"

# How long a second process waits for the ONNX inference lock before giving up.
# Overridable so power users on beefier machines can wait longer if they want to
# queue back-to-back runs. Kept short by default: waiting 10 minutes silently
# (the old behaviour) looks exactly like a hang.
_DEFAULT_LOCK_TIMEOUT = int(os.environ.get("RAGDOLL_LOCK_TIMEOUT", "120"))


class LockBusy(RuntimeError):
    """Raised when the single-inference lock can't be acquired in time.

    Signals that another ragdoll process is already running the model. Callers
    should surface a friendly message and exit rather than loading a second
    ~500 MB model and risking OOM.
    """


@contextmanager
def _index_lock(timeout: Optional[int] = None, *, purpose: str = "index"):
    """File lock so only one ragdoll process runs ONNX inference at a time.

    Loading the model costs ~500 MB; letting several processes do it at once is
    the main cause of OOM on memory-constrained machines. A second process
    prints a one-line notice, waits up to `timeout` seconds, then raises
    `LockBusy` (fail fast) instead of blocking silently or piling on another
    model.

    Uses fcntl on Unix. On platforms without fcntl, skips locking (better to
    risk contention than to crash).
    """
    if timeout is None:
        timeout = _DEFAULT_LOCK_TIMEOUT
    try:
        import fcntl
    except ImportError:
        # Windows or other platform without fcntl -- skip locking
        yield
        return

    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = open(_LOCK_PATH, "w")
        deadline = time.monotonic() + timeout
        notified = False
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() > deadline:
                    raise LockBusy(
                        f"Another ragdoll process is using the model - timed out "
                        f"after {timeout}s waiting to {purpose}. It's still running, "
                        f"not stuck; retry shortly or set RAGDOLL_LOCK_TIMEOUT to wait longer."
                    )
                if not notified:
                    print(
                        f"Another ragdoll process is using the model - "
                        f"waiting up to {timeout}s to {purpose}...",
                        file=sys.stderr,
                        flush=True,
                    )
                    notified = True
                time.sleep(1)
        yield
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fd.close()


_model_checked = False  # only validate once per process


@lru_cache(maxsize=4096)
def _real_case_dir(parent: Path) -> dict[str, str]:
    """Return {lowercased_name: actual_on_disk_name} for entries of `parent`.

    Used by `_canonical_path` to recover the true casing of each path
    component on case-insensitive filesystems (macOS APFS/HFS+, Windows NTFS).
    Cached because directories rarely change during a single index run.
    """
    try:
        return {e.name.lower(): e.name for e in os.scandir(parent)}
    except OSError:
        return {}


def _canonical_path(p: Path) -> Path:
    """Return `p` with the actual on-disk casing for every component.

    macOS's filesystem is case-insensitive but case-preserving - `Path.resolve()`
    keeps whatever casing the user typed (`/RAGdoll` vs `/ragdoll`). Indexing
    with two different casings creates two unrelated sets of chunk rows in the
    DB and forces a full re-embed every time the casing changes. Walking each
    component and looking up the directory entry gives us a stable, canonical
    string regardless of how the user typed the path.
    """
    p = p.resolve()
    parts = p.parts
    if not parts:
        return p
    cur = Path(parts[0])  # filesystem root, e.g. "/"
    for part in parts[1:]:
        entries = _real_case_dir(cur)
        cur = cur / entries.get(part.lower(), part)
    return cur


@lru_cache(maxsize=128)
def _find_repo_root(path: Path) -> str:
    """Resolve the git repo root for a directory. Cached - avoids re-reading .git per file."""
    try:
        repo = git.Repo(path, search_parent_directories=True)
        root = repo.working_tree_dir
        if root is None:
            return str(path)
        return root
    except git.InvalidGitRepositoryError:
        return str(path)


@dataclass
class FilePlan:
    """Result of planning one file: what to prune, reuse, and embed.

    Produced without touching the DB or the embedder, so it can be built on a
    background thread while the main thread embeds the previous batch.
    """
    path: Path
    repo: str
    raw_count: int
    reused: list = field(default_factory=list)      # (idx, rc, hash, vector)
    to_embed: list = field(default_factory=list)    # (idx, rc, hash)
    removed: list = field(default_factory=list)      # chunk indices to delete
    unchanged: int = 0                               # chunks identical in place, skipped entirely


class Indexer:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self._store = store
        self._embedder = embedder
        self._cancelled = False
        # Per-run stats, reset at the start of each index_path() call so the
        # CLI can report embedded-vs-reused-vs-unchanged instead of one opaque
        # total (a resume of a partial index otherwise looks like a full rebuild).
        self._last_embedded = 0
        self._last_reused = 0
        self._last_unchanged = 0

    def cancel(self) -> None:
        """Request graceful cancellation. Honored between files and batches."""
        self._cancelled = True

    @contextmanager
    def _sigint_handler(self):
        """Install a SIGINT handler that flips the cancellation flag.

        onnxruntime's `model.embed()` is a C call that won't return mid-batch
        even when SIGINT fires - Python only delivers the signal when control
        re-enters the interpreter. We can't interrupt the in-flight batch, but
        we can guarantee the loop stops at the next safe point (between files
        or between batches) instead of running to completion.

        First Ctrl+C: set flag, log a notice, keep going to a safe stop.
        Second Ctrl+C: restore default handler so the user can hard-kill.
        """
        if signal.getsignal(signal.SIGINT) is None:
            # Non-main thread - can't install signal handlers. No-op.
            yield
            return
        hits = {"n": 0}

        def handler(signum, frame):
            hits["n"] += 1
            self._cancelled = True
            if hits["n"] == 1:
                logger.warning(
                    "Cancellation requested - finishing current batch then stopping. "
                    "Press Ctrl+C again to abort immediately."
                )
            else:
                # Restore default so the next signal kills the process
                signal.signal(signal.SIGINT, signal.SIG_DFL)

        try:
            previous = signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Not in main thread; can't install
            yield
            return
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)

    def index_path(
        self,
        path: Path,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Index a file or directory. Returns number of chunks indexed.

        Acquires a file lock so only one ragdoll process runs ONNX
        inference at a time, preventing parallel OOM on memory-constrained
        machines.

        Args:
            progress: Optional callback(files_done, total_files) for progress reporting.
        """
        # Reset cancellation flag and per-run stats for this run
        self._cancelled = False
        self._last_embedded = 0
        self._last_reused = 0
        self._last_unchanged = 0

        # Validate embedding model against DB on first call
        global _model_checked
        if not _model_checked:
            self._store.check_embed_model(self._embedder.model_name, self._embedder.dim)
            _model_checked = True

        # Normalize on-disk casing once for the run - `_walk_files` does this
        # per-file too, but doing it here covers the single-file branch.
        path = _canonical_path(path)

        with _index_lock(), self._sigint_handler():
            logger.debug(
                f"Index lock acquired, batch_size={BATCH_SIZE}"
            )

            try:
                if path.is_file():
                    return self._index_file(path)

                # Collect indexable files first (with directory pruning)
                files = list(self._walk_files(path))
                total = len(files)

                if not files:
                    return 0

                # Preload per-chunk hashes AND existing vectors keyed by content hash
                # in one query each - lets the indexer skip embedding any chunk whose
                # content is unchanged (even when its *position* in the file shifted).
                all_hashes = self._store.all_hashes_by_index()
                all_vectors = self._store.all_vectors_by_hash()

                return self._index_dir(files, total, all_hashes, all_vectors, progress)
            finally:
                # Release ONNX model memory immediately. Without this,
                # the model weights (~500 MB) and inference buffers stay
                # resident until process exit, which is the main cause of
                # runaway memory after `ragdoll index` completes.
                self._embedder.unload()

    def _index_dir(self, files, total, all_hashes, all_vectors, progress) -> int:
        """Index a directory's files with a read/embed/write pipeline.

        A background thread reads + chunks + plans files (disk I/O + CPU, no DB,
        no embedding). This thread embeds (onnxruntime releases the GIL during
        inference, so planning genuinely overlaps it) and does all DB writes -
        keeping SQLite single-threaded. Chunks are batched across files so the
        model always gets full batches instead of one underfull batch per file.
        """
        SENTINEL = object()
        plan_q: queue.Queue = queue.Queue(maxsize=PLAN_QUEUE_MAX)

        def produce():
            try:
                for f in files:
                    if self._cancelled:
                        break
                    plan = self._plan_file(
                        f,
                        all_hashes.get(str(f), {}),
                        all_vectors.get(str(f), {}),
                    )
                    plan_q.put((f, plan))
            except Exception as exc:  # surface to the consumer, don't die silently
                plan_q.put(exc)
            finally:
                plan_q.put(SENTINEL)

        producer = threading.Thread(
            target=produce, name="ragdoll-planner", daemon=True
        )
        producer.start()

        staged_rows: list[dict] = []
        embed_buf: list = []  # (path, repo, idx, rc, hash)
        indexed = 0
        files_done = 0

        # Global content-hash -> vector cache. Seeded from every vector already in
        # the DB, then grown as we embed. Lets identical chunks (boilerplate
        # repeated across files - license headers, shared helm templates, vendored
        # snippets) be embedded ONCE per run instead of once per file. Measured
        # ~18-25% of chunks in config-heavy repos are cross-file duplicates.
        run_vectors: dict[str, list] = {}
        for fv in all_vectors.values():
            run_vectors.update(fv)

        def flush_embed() -> None:
            nonlocal indexed
            if not embed_buf:
                return
            # Only embed content we don't already have a vector for (globally or
            # earlier this run). Everything else reuses the cached vector.
            need: dict[str, str] = {}  # hash -> content, unique
            for (_, _, _, rc_, h_) in embed_buf:
                if h_ not in run_vectors and h_ not in need:
                    need[h_] = rc_.content
            if need:
                hashes = list(need)
                vectors = self._embedder.embed([need[h] for h in hashes])
                for h, vec in zip(hashes, vectors):
                    run_vectors[h] = vec
                throttle = _embed_throttle()
                if throttle:
                    time.sleep(throttle)
            for (p_, repo_, idx_, rc_, h_) in embed_buf:
                staged_rows.append(self._row(p_, repo_, idx_, rc_, h_, run_vectors[h_]))
            indexed += len(embed_buf)
            self._last_embedded += len(need)
            self._last_reused += len(embed_buf) - len(need)
            embed_buf.clear()

        def flush_rows() -> None:
            if staged_rows:
                self._store.upsert(staged_rows)
                staged_rows.clear()

        try:
            while True:
                item = plan_q.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item

                f, plan = item
                files_done += 1
                if plan is not None:
                    if plan.removed:
                        self._store.delete_chunks_by_index(str(f), plan.removed)
                    for (idx, rc, h, vec) in plan.reused:
                        staged_rows.append(self._row(f, plan.repo, idx, rc, h, vec))
                        indexed += 1
                    self._last_reused += len(plan.reused)
                    self._last_unchanged += plan.unchanged
                    for (idx, rc, h) in plan.to_embed:
                        embed_buf.append((f, plan.repo, idx, rc, h))
                    if len(embed_buf) >= EMBED_WINDOW:
                        flush_embed()
                    if len(staged_rows) >= ROW_FLUSH:
                        flush_rows()

                if progress:
                    progress(files_done, total)

                if self._cancelled:
                    logger.warning(
                        f"Cancelled - stopped after {files_done}/{total} files."
                    )
                    break

            # Commit whatever's been planned/embedded so far
            flush_embed()
            flush_rows()
        finally:
            if self._cancelled:
                # We stopped early; a producer parked on a full queue would
                # deadlock the join. Drain until its sentinel to free it.
                while True:
                    try:
                        leftover = plan_q.get(timeout=10)
                    except queue.Empty:
                        break
                    if leftover is SENTINEL:
                        break
            producer.join(timeout=5)

        return indexed

    def remove_path(self, path: Path) -> None:
        """Remove all chunks for a deleted file."""
        self._store.delete_by_source(str(path))
        logger.debug(f"Removed {path} from index")

    def _walk_files(self, root: Path):
        """Walk directory tree, pruning skip dirs early and skipping symlinks."""
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            # Prune directories in-place so os.walk doesn't descend into them
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS
                and not d.startswith(".")
                and not Path(dirpath, d).is_symlink()
            ]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                # Skip symlinks at file level too
                if fpath.is_symlink():
                    continue
                # Canonicalize casing so reindexing with /RAGdoll vs /ragdoll
                # (case-insensitive FS) hits the same DB rows instead of
                # creating a duplicate set.
                yield _canonical_path(fpath)

    def _plan_file(
        self,
        path: Path,
        existing_hashes: dict[int, str],
        existing_vectors: dict[str, list[float]],
    ) -> FilePlan | None:
        """Chunk a file and decide what to prune / reuse / embed - no DB writes,
        no embedding. Safe to run on a background thread.

        Two-tier reuse, in order of preference:

        1. If the chunk at position `idx` still has the same content hash as in
           the DB, nothing is written - the existing row is already correct.
        2. Otherwise, if the new chunk's content hash matches *any* vector
           already stored for this file (content-anchored), the existing vector
           is reused and only the row is re-written at the new position. This
           makes insertion-at-top cheap: inserting a function above the rest
           re-indexes rows but re-embeds nothing.
        3. Only genuinely new content is sent to the embedder.
        """
        raw_chunks = chunk_file(path)
        if not raw_chunks:
            return None

        repo = _find_repo_root(path.parent)
        reused: list = []
        to_embed: list = []
        unchanged = 0
        for idx, rc in enumerate(raw_chunks):
            new_hash = VectorStore.content_hash(rc.content)
            if existing_hashes.get(idx) == new_hash:
                unchanged += 1
                continue  # row already correct - no write at all
            cached = existing_vectors.get(new_hash)
            if cached is not None:
                reused.append((idx, rc, new_hash, cached))
            else:
                to_embed.append((idx, rc, new_hash))

        removed = [idx for idx in existing_hashes if idx >= len(raw_chunks)]
        return FilePlan(
            path=path, repo=repo, raw_count=len(raw_chunks),
            reused=reused, to_embed=to_embed, removed=removed, unchanged=unchanged,
        )

    @staticmethod
    def _row(path: Path, repo: str, idx: int, rc, content_hash: str, vector) -> dict:
        """Build a chunk row dict for the store."""
        return {
            "id": VectorStore.chunk_id(str(path), idx),
            "content": rc.content,
            "content_hash": content_hash,
            "source_path": str(path),
            "repo": repo,
            "language": rc.language,
            "chunk_index": idx,
            "start_line": rc.start_line,
            "end_line": rc.end_line,
            "vector": vector,
        }

    def _index_file(
        self,
        path: Path,
        existing_hashes: dict[int, str] | None = None,
        existing_vectors: dict[str, list[float]] | None = None,
    ) -> int:
        """Index a single file, reusing unchanged chunks.

        Used for the single-file path; directory indexing uses the pipelined
        path in `_index_dir`. Shares chunking/partition logic via `_plan_file`.
        """
        if existing_hashes is None:
            existing_hashes = self._store.hashes_by_index(str(path))
        if existing_vectors is None:
            existing_vectors = self._store.vectors_by_hash(str(path))

        plan = self._plan_file(path, existing_hashes, existing_vectors)
        if plan is None:
            return 0
        # Record unchanged chunks before any early return so a no-op re-index
        # still reports "already up to date" instead of a silent zero.
        self._last_unchanged += plan.unchanged
        if plan.removed:
            self._store.delete_chunks_by_index(str(path), plan.removed)
        if not plan.reused and not plan.to_embed:
            return 0

        rows = [
            self._row(path, plan.repo, idx, rc, h, vec)
            for (idx, rc, h, vec) in plan.reused
        ]
        embedded = 0
        seen: dict[str, list] = {}  # hash -> vector; dedup identical chunks in one file
        throttle = _embed_throttle()
        for start in range(0, len(plan.to_embed), BATCH_SIZE):
            if self._cancelled:
                # Don't start another (multi-second) embed batch after Ctrl+C.
                # Partially-indexed files are fine - each chunk row is independent.
                break
            batch = plan.to_embed[start : start + BATCH_SIZE]
            need: dict[str, str] = {}  # hash -> content, unique & not yet embedded
            for (_, rc, h) in batch:
                if h not in seen and h not in need:
                    need[h] = rc.content
            if need:
                hs = list(need)
                vectors = self._embedder.embed([need[h] for h in hs])
                for h, vec in zip(hs, vectors):
                    seen[h] = vec
                embedded += len(need)
                if throttle:
                    time.sleep(throttle)
            for (idx, rc, h) in batch:
                rows.append(self._row(path, plan.repo, idx, rc, h, seen[h]))

        self._store.upsert(rows)
        self._last_reused += len(plan.reused) + (len(plan.to_embed) - embedded)
        self._last_embedded += embedded
        return len(plan.reused) + len(plan.to_embed)
