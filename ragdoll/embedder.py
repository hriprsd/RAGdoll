"""
Embedder — wraps FastEmbed for fully local ONNX-based embedding.

No PyTorch needed. Total dependency footprint: ~50 MB vs ~2 GB with sentence-transformers.

Default: nomic-ai/nomic-embed-text-v1.5 (768-dim, ONNX, CPU-optimised).
Fallback: BAAI/bge-base-en-v1.5 (768-dim) — same dimensions, no corruption risk.

Hardware acceleration:
  - macOS: CoreML (Apple Neural Engine / GPU) when available
  - Linux/Windows: CUDA when available
  - Everywhere: CPU fallback with thread count capped to avoid thrashing
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(vec: Any) -> "np.ndarray":
    """Scale a vector to unit length.

    FastEmbed's nomic ONNX output is NOT unit-normalized (norms ~20), which
    makes vector search magnitude-dominated and produces meaningless nearest
    neighbours. Normalizing here keeps stored and query vectors in the same
    comparable space, so cosine/L2 ranking is well-defined. Accepts numpy
    arrays or any sequence of floats.
    """
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


# Hard cap on chars sent to the model per chunk. Transformer attention is
# O(seq^2), so an oversized chunk (e.g. a minified/data file that slipped past
# the chunker) would spike onnxruntime memory. Matches the chunker's
# MAX_CHUNK_CHARS; kept here as defence-in-depth so the embedder is safe no
# matter what it's handed.
_MAX_EMBED_CHARS = 8000

# Peak ONNX working memory for a batch scales ~ batch_count * seq_len^2. Bound
# that product (in chars^2) so a batch of many large chunks can't blow up RAM.
# Indexing a big C/C++ repo previously pushed RSS into tens of GB + swap because
# 64 near-max chunks were embedded in one padded batch. ~3e8 keeps a worst-case
# batch (all 8000-char chunks) to a handful of items, while small chunks still
# batch freely. Tunable via RAGDOLL_EMBED_AREA.
_EMBED_AREA = int(os.environ.get("RAGDOLL_EMBED_AREA", 300_000_000))

# Hard cap on sequences per batch, independent of the area budget. The area
# budget alone bounds count*max_len^2, but for very short chunks max_len is tiny
# so it would permit tens of thousands of sequences in one batch — and onnxruntime
# allocates per-sequence working buffers that blow RSS into many GB. This cap
# keeps a batch sane even when callers hand us a large length-sorted pool of
# short chunks. Tunable via RAGDOLL_MAX_BATCH.
_MAX_BATCH_COUNT = int(os.environ.get("RAGDOLL_MAX_BATCH", 256))


def _memory_safe_batches(texts: list[str]):
    """Yield sub-batches bounded by both padded area (count * max_len^2) and a
    hard count cap (_MAX_BATCH_COUNT).

    Greedy: grow a batch until adding the next chunk would exceed _EMBED_AREA
    given the batch's longest member (ONNX pads every sequence to the longest),
    or the batch would exceed _MAX_BATCH_COUNT sequences. A single chunk always
    forms a batch even if it alone exceeds the area budget — but truncation to
    _MAX_EMBED_CHARS keeps that case small too.
    """
    batch: list[str] = []
    cur_max = 0
    for t in texts:
        new_max = len(t) if len(t) > cur_max else cur_max
        too_big = (len(batch) + 1) * new_max * new_max > _EMBED_AREA
        too_many = len(batch) + 1 > _MAX_BATCH_COUNT
        if batch and (too_big or too_many):
            yield batch
            batch, cur_max = [t], len(t)
        else:
            batch.append(t)
            cur_max = new_max
    if batch:
        yield batch

# Cache size for repeated query embeddings (~2MB RAM at 768-dim × 256 queries).
# Helps interactive CLI and IDE autocomplete where the same query repeats.
_QUERY_CACHE_SIZE = 256

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FALLBACK_MODEL = "BAAI/bge-base-en-v1.5"
# `--fast` model — ~3× faster on CPU, half the dimension. Tradeoff: slightly
# weaker recall on conceptual queries. Use a separate DB file (default:
# ~/.ragdoll/ragdoll-fast.db) so the 384-dim vectors don't collide with the
# 768-dim ones already on disk.
FAST_MODEL = "BAAI/bge-small-en-v1.5"

# `--quantized` model — same nomic architecture and dimension (768), but
# int8-quantized ONNX weights: ~2× faster on CPU and lighter on memory for a
# small recall cost. Because the vectors differ numerically from the fp32 model,
# it uses its own DB (default: ~/.ragdoll/ragdoll-quant.db) so the two never mix.
QUANT_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"

# Model → embedding dim. Add new models here.
MODEL_DIMS: dict[str, int] = {
    DEFAULT_MODEL: 768,
    FALLBACK_MODEL: 768,
    FAST_MODEL: 384,
    QUANT_MODEL: 768,
}

# Back-compat: kept so existing imports keep working. Reflects the default
# model only — code that supports multiple models should read `embedder.dim`.
EXPECTED_DIM = MODEL_DIMS[DEFAULT_MODEL]

# Module-level cache: model_name → TextEmbedding instance
# Prevents reloading when multiple Embedder() calls happen (e.g. add_memory).
_model_cache: dict[str, Any] = {}


def _detect_providers() -> list[str] | None:
    """Auto-detect the best ONNX execution providers for this machine.

    Returns a provider list for FastEmbed, or None to let it use defaults.
    Detection is best-effort: if onnxruntime isn't importable or a provider
    isn't available, we silently fall back to CPU.

    Note on CoreML: nomic-embed-text-v1.5 uses dynamic shapes with
    dimension 0 in its rotary embeddings, which CoreML doesn't support.
    ONNX falls back to a split execution (some ops on CoreML, rest on
    CPU) that doubles memory usage and is slower than pure CPU. We
    skip CoreML entirely for embedding models. CUDA works fine.
    """
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except (ImportError, Exception):
        return None

    providers: list[str] = []

    # CUDA for Linux/Windows with NVIDIA GPU -- works well with embedding models
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    # CoreML is intentionally skipped. The nomic/bge embedding models use
    # dynamic shapes that CoreML can't handle, causing split execution
    # (half on CoreML, half on CPU) that doubles memory and is slower.
    # See: https://github.com/microsoft/onnxruntime/issues/16455

    # CPU is the reliable default for embedding models on macOS
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    return providers if providers else None


def _cache_dir() -> Path:
    """Stable on-disk cache for embedding models.

    FastEmbed defaults to a temp dir under $TMPDIR ("fastembed_cache") when no
    cache_dir is passed and no cache env var is set. On macOS that temp dir is
    periodically purged, so the ~500 MB model re-downloads from HF on every
    fresh process. Pin it to ~/.cache/fastembed (honouring an explicit
    RAGDOLL_MODEL_CACHE override, or legacy FASTEMBED_CACHE_PATH) so every entry
    point — CLI wrapper, MCP server, git hooks, tests — shares one persistent
    cache.
    """
    raw = (
        os.environ.get("RAGDOLL_MODEL_CACHE")
        or os.environ.get("FASTEMBED_CACHE_PATH")
        or "~/.cache/fastembed"
    )
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    # FastEmbed receives cache_dir explicitly, but HuggingFace hub internals and
    # any direct fastembed use in-process read FASTEMBED_CACHE_PATH — keep them
    # pointed at the same dir so nothing re-downloads to a second location.
    os.environ["FASTEMBED_CACHE_PATH"] = str(path)
    return path


def _model_is_cached(cache_dir: Path, model_name: str) -> bool:
    """True if the model's ONNX weights are already on disk under cache_dir.

    Lets us pass local_files_only=True so FastEmbed skips the HuggingFace
    metadata round-trip (model_info / list_repo_tree) it otherwise performs on
    every load — that network call is what prints "Fetching N files" even when
    nothing actually needs downloading.
    """
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    return model_dir.is_dir() and any(model_dir.glob("**/*.onnx"))


def _performance_cores() -> int | None:
    """Number of performance cores on Apple Silicon, or None if unknown.

    Efficiency cores add little to a latency-bound matmul and can drag the
    batch down to their speed, so we prefer to size the thread pool to the
    performance cores only.
    """
    try:
        import platform
        if platform.system() != "Darwin":
            return None
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            text=True, timeout=2,
        ).strip()
        n = int(out)
        return n if n > 0 else None
    except Exception:
        return None


def _default_threads() -> int | None:
    """Pick ONNX intra-op thread count for a single index run.

    A file lock (see indexer._index_lock) serialises ONNX inference to one
    process at a time, so a single run can safely use most of the machine —
    the old `cores / 2` cap left ~half the cores idle for no reason. Prefer the
    performance-core count on Apple Silicon, else leave 2 cores for the OS/UI.

    Set RAGDOLL_THREADS to override (0 = let ONNX decide).
    """
    env = os.environ.get("RAGDOLL_THREADS")
    if env is not None:
        val = int(env)
        return val if val > 0 else None  # 0 means "no cap"
    perf = _performance_cores()
    if perf:
        return perf
    ncpu = os.cpu_count() or 4
    return max(2, ncpu - 2)


class Embedder:
    """Lightweight ONNX-based text embedder via FastEmbed.

    Models are downloaded on first use (~200-300 MB one-time) and cached
    under ~/.cache/fastembed/. Subsequent loads are instant.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._expected_dim = MODEL_DIMS.get(model_name, EXPECTED_DIM)
        self._model: Any | None = None
        # Per-instance tuple-returning cache so lists of floats become hashable
        self._cached_query = lru_cache(maxsize=_QUERY_CACHE_SIZE)(self._embed_query_uncached)

    def _load_model(self) -> Any:
        """Lazy-load the embedding model, with fallback on failure.

        Automatically selects the best ONNX execution provider:
          - CoreML on macOS (Apple Neural Engine / GPU)
          - CUDA on Linux/Windows with NVIDIA GPU
          - CPU everywhere else
        Thread count is capped to prevent thrashing when multiple
        ragdoll processes run in parallel.
        """
        if self._model is not None:
            return self._model

        # Check module-level cache first
        if self.model_name in _model_cache:
            self._model = _model_cache[self.model_name]
            return self._model

        from fastembed import TextEmbedding

        cache_dir = _cache_dir()
        providers = _detect_providers()
        threads = _default_threads()

        provider_names = [p if isinstance(p, str) else p[0] for p in (providers or [])]
        logger.info(
            f"Loading embedding model: {self.model_name} (ONNX) "
            f"providers={provider_names}, threads={threads}"
        )

        # cuda=False prevents FastEmbed's own auto-detection from picking
        # up CoreML/CUDA behind our back. We control providers explicitly.
        has_cuda = providers is not None and any(
            (p if isinstance(p, str) else p[0]) == "CUDAExecutionProvider"
            for p in providers
        )

        try:
            model = TextEmbedding(
                self.model_name,
                cache_dir=str(cache_dir),
                providers=providers,
                threads=threads,
                cuda=has_cuda,
                local_files_only=_model_is_cached(cache_dir, self.model_name),
                # Release each batch's working set instead of holding the peak
                # high-water mark for the whole run — keeps RSS low and stable
                # on memory-constrained machines.
                enable_cpu_mem_arena=False,
            )
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as exc:
            if self.model_name == FALLBACK_MODEL:
                raise RuntimeError(
                    f"Failed to load both primary and fallback embedding models. "
                    f"Ensure you have internet for the initial download, then models "
                    f"are cached locally. Error: {exc}"
                ) from exc
            # Never silently degrade to a different model when the intended
            # model's weights are already on disk. Falling back here would embed
            # with a different model than the rest of the DB — an incompatible
            # vector space that returns garbage. Fail loudly instead; the
            # fallback exists only for the genuine first-run/no-download case.
            if _model_is_cached(cache_dir, self.model_name):
                raise RuntimeError(
                    f"'{self.model_name}' is cached at {cache_dir} but failed to "
                    f"load: {exc}. Refusing to fall back to '{FALLBACK_MODEL}' — "
                    f"mixing models corrupts the vector store. Fix the load error "
                    f"(or delete the cached model to force a clean re-download) "
                    f"rather than indexing with the wrong model."
                ) from exc
            logger.warning(
                f"Failed to load '{self.model_name}': {exc}. "
                f"Falling back to '{FALLBACK_MODEL}'."
            )
            self.model_name = FALLBACK_MODEL
            try:
                model = TextEmbedding(
                    FALLBACK_MODEL,
                    cache_dir=str(cache_dir),
                    providers=providers,
                    threads=threads,
                    cuda=has_cuda,
                    local_files_only=_model_is_cached(cache_dir, FALLBACK_MODEL),
                    enable_cpu_mem_arena=False,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Failed to load fallback model '{FALLBACK_MODEL}': {fallback_exc}. "
                    f"Run `ragdoll setup` with internet access to download models."
                ) from fallback_exc

        # Validate dimensions — a mismatch here would silently corrupt the DB
        test_vec = next(iter(model.embed(["dimension check"])))
        actual_dim = len(test_vec)
        if actual_dim != self._expected_dim:
            raise RuntimeError(
                f"Model '{self.model_name}' produces {actual_dim}-dim vectors "
                f"but the registry says {self._expected_dim}-dim. "
                f"This would corrupt the vector store. Aborting."
            )

        _model_cache[self.model_name] = model
        self._model = model
        return model

    @property
    def dim(self) -> int:
        """Return the embedding dimension for this embedder's model."""
        return self._expected_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts. Returns a list of float vectors.

        Sorts the batch by length before splitting into memory-safe sub-batches
        (bounded by count * max_len^2). onnxruntime pads every sequence in a
        batch up to the longest member, so embedding mixed-length chunks
        together wastes compute padding short chunks. Length-sorting groups
        similar sizes so each sub-batch is tightly packed — a big win on repos
        with a few long chunks among many short ones. Results are scattered back
        to the caller's original order, so this is invisible to callers.
        """
        if not texts:
            return []
        model = self._load_model()
        # Truncate pathologically long chunks first (defence-in-depth).
        texts = [t[:_MAX_EMBED_CHARS] for t in texts]
        # Embed in length-sorted order, then restore the caller's order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[list[float] | None] = [None] * len(texts)
        pos = 0
        for sub in _memory_safe_batches([texts[i] for i in order]):
            for vec in model.embed(sub, batch_size=len(sub)):
                out[order[pos]] = _l2_normalize(vec).tolist()
                pos += 1
        return out  # type: ignore[return-value]

    def _embed_query_uncached(self, query: str) -> tuple[float, ...]:
        """Actual query embedding — returns a tuple so it's hashable/cacheable."""
        model = self._load_model()
        vec = next(iter(model.query_embed(query)))
        return tuple(_l2_normalize(vec).tolist())

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query. Cached — repeated queries are free."""
        return list(self._cached_query(query))

    def clear_query_cache(self) -> None:
        """Drop the query-embedding LRU cache (e.g. after a model switch)."""
        self._cached_query.cache_clear()

    def unload(self) -> None:
        """Release the ONNX model and free memory.

        ONNX Runtime holds C-level allocations (model weights, inference
        buffers, CoreML/CUDA contexts) that Python's GC won't reclaim.
        Call this after bulk operations (index) to give memory back to
        the OS immediately instead of holding it until process exit.
        """
        if self._model is not None:
            # Remove from module-level cache so it doesn't pin the object
            _model_cache.pop(self.model_name, None)
            self._model = None
            self._cached_query.cache_clear()
            # Force GC to release ONNX session and its C-level buffers
            import gc
            gc.collect()
            logger.debug("Embedding model unloaded, memory released")
