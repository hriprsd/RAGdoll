"""
Indexer — walks directories, chunks files, embeds, and stores.
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
import signal
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import git

from .chunker import chunk_file, SKIP_DIRS
from .embedder import Embedder
from .store import VectorStore

logger = logging.getLogger(__name__)

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

# --- Process lock ------------------------------------------------------------
# Prevents multiple `ragdoll index` invocations from loading ONNX models
# simultaneously. Each ONNX model load allocates ~500 MB; 5 parallel
# processes = ~2.5 GB just for model weights, plus per-batch working memory.
# On a 24 GB machine with normal app usage, that's a guaranteed OOM.

_LOCK_PATH = Path(os.environ.get("RAGDOLL_HOME", Path.home() / ".ragdoll")) / ".index.lock"


@contextmanager
def _index_lock(timeout: int = 600):
    """File-based lock so only one ragdoll index process embeds at a time.

    Other processes wait (with a timeout) rather than competing for RAM.
    Uses fcntl on Unix. On platforms without fcntl, skips locking (better
    to risk contention than to crash).
    """
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
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() > deadline:
                    logger.warning(
                        "Timed out waiting for index lock -- another ragdoll "
                        "index process may be stuck. Proceeding anyway."
                    )
                    break
                logger.info("Another ragdoll index is running -- waiting for lock...")
                time.sleep(2)
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

    macOS's filesystem is case-insensitive but case-preserving — `Path.resolve()`
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
    """Resolve the git repo root for a directory. Cached — avoids re-reading .git per file."""
    try:
        repo = git.Repo(path, search_parent_directories=True)
        root = repo.working_tree_dir
        if root is None:
            return str(path)
        return root
    except git.InvalidGitRepositoryError:
        return str(path)


class Indexer:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self._store = store
        self._embedder = embedder
        self._cancelled = False

    def cancel(self) -> None:
        """Request graceful cancellation. Honored between files and batches."""
        self._cancelled = True

    @contextmanager
    def _sigint_handler(self):
        """Install a SIGINT handler that flips the cancellation flag.

        onnxruntime's `model.embed()` is a C call that won't return mid-batch
        even when SIGINT fires — Python only delivers the signal when control
        re-enters the interpreter. We can't interrupt the in-flight batch, but
        we can guarantee the loop stops at the next safe point (between files
        or between batches) instead of running to completion.

        First Ctrl+C: set flag, log a notice, keep going to a safe stop.
        Second Ctrl+C: restore default handler so the user can hard-kill.
        """
        if signal.getsignal(signal.SIGINT) is None:
            # Non-main thread — can't install signal handlers. No-op.
            yield
            return
        hits = {"n": 0}

        def handler(signum, frame):
            hits["n"] += 1
            self._cancelled = True
            if hits["n"] == 1:
                logger.warning(
                    "Cancellation requested — finishing current batch then stopping. "
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
        # Reset cancellation flag for this run
        self._cancelled = False

        # Validate embedding model against DB on first call
        global _model_checked
        if not _model_checked:
            self._store.check_embed_model(self._embedder.model_name, self._embedder.dim)
            _model_checked = True

        # Normalize on-disk casing once for the run — `_walk_files` does this
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
                # in one query each — lets the indexer skip embedding any chunk whose
                # content is unchanged (even when its *position* in the file shifted).
                all_hashes = self._store.all_hashes_by_index()
                all_vectors = self._store.all_vectors_by_hash()

                indexed = 0
                for i, f in enumerate(files):
                    if self._cancelled:
                        logger.warning(f"Cancelled — stopped after {i}/{total} files.")
                        break
                    indexed += self._index_file(
                        f,
                        existing_hashes=all_hashes.get(str(f), {}),
                        existing_vectors=all_vectors.get(str(f), {}),
                    )
                    if progress:
                        progress(i + 1, total)

                return indexed
            finally:
                # Release ONNX model memory immediately. Without this,
                # the model weights (~500 MB) and inference buffers stay
                # resident until process exit, which is the main cause of
                # runaway memory after `ragdoll index` completes.
                self._embedder.unload()

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

    def _index_file(
        self,
        path: Path,
        existing_hashes: dict[int, str] | None = None,
        existing_vectors: dict[str, list[float]] | None = None,
    ) -> int:
        """Index a single file, reusing unchanged chunks.

        Two-tier reuse, in order of preference:

        1. If the chunk at position `idx` still has the same content hash as in
           the DB, nothing is written — the existing row is already correct.
        2. Otherwise, if the new chunk's content hash matches *any* vector
           already stored for this file (content-anchored), the existing vector
           is reused and only the row is re-written at the new position. This
           makes insertion-at-top cheap: inserting a function above the rest
           re-indexes rows but re-embeds nothing.
        3. Only genuinely new content is sent to the embedder.
        """
        raw_chunks = chunk_file(path)
        if not raw_chunks:
            return 0

        repo = _find_repo_root(path.parent)

        if existing_hashes is None:
            existing_hashes = self._store.hashes_by_index(str(path))
        if existing_vectors is None:
            existing_vectors = self._store.vectors_by_hash(str(path))

        # Partition chunks: unchanged-at-position / reusable / needs-embedding
        reused: list[tuple[int, object, str, list[float]]] = []
        to_embed: list[tuple[int, object, str]] = []
        for idx, rc in enumerate(raw_chunks):
            new_hash = VectorStore.content_hash(rc.content)
            if existing_hashes.get(idx) == new_hash:
                continue  # row already correct — no write at all
            cached = existing_vectors.get(new_hash)
            if cached is not None:
                reused.append((idx, rc, new_hash, cached))
            else:
                to_embed.append((idx, rc, new_hash))

        # Prune any chunks beyond the new file length
        removed_indices = [
            idx for idx in existing_hashes if idx >= len(raw_chunks)
        ]
        if removed_indices:
            self._store.delete_chunks_by_index(str(path), removed_indices)

        if not reused and not to_embed:
            return 0  # nothing changed at the chunk level

        all_chunks: list[dict] = []

        # Rows we can rewrite without embedding — vector comes from the DB
        for idx, rc, new_hash, vec in reused:
            all_chunks.append({
                "id": VectorStore.chunk_id(str(path), idx),
                "content": rc.content,
                "content_hash": new_hash,
                "source_path": str(path),
                "repo": repo,
                "language": rc.language,
                "chunk_index": idx,
                "start_line": rc.start_line,
                "end_line": rc.end_line,
                "vector": vec,
            })

        # Genuine new content — batch through the embedder
        for batch_start in range(0, len(to_embed), BATCH_SIZE):
            if self._cancelled:
                # Don't start another (multi-second) embed batch after Ctrl+C.
                # Whatever we already embedded for this file will be upserted
                # below; partially-indexed files are fine because each chunk
                # row is independent.
                to_embed = to_embed[:batch_start]
                break
            batch = to_embed[batch_start : batch_start + BATCH_SIZE]
            texts = [rc.content for _, rc, _ in batch]
            vectors = self._embedder.embed(texts)
            for (idx, rc, new_hash), vec in zip(batch, vectors):
                all_chunks.append({
                    "id": VectorStore.chunk_id(str(path), idx),
                    "content": rc.content,
                    "content_hash": new_hash,
                    "source_path": str(path),
                    "repo": repo,
                    "language": rc.language,
                    "chunk_index": idx,
                    "start_line": rc.start_line,
                    "end_line": rc.end_line,
                    "vector": vec,
                })

        self._store.upsert(all_chunks)
        logger.debug(
            f"Indexed {len(raw_chunks)} chunks from {path} "
            f"(embedded {len(to_embed)}, reused {len(reused)}, "
            f"unchanged {len(raw_chunks) - len(reused) - len(to_embed)})"
        )
        return len(to_embed) + len(reused)
