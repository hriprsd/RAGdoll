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
from typing import Any, Sequence

logger = logging.getLogger(__name__)

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

# Model → embedding dim. Add new models here.
MODEL_DIMS: dict[str, int] = {
    DEFAULT_MODEL: 768,
    FALLBACK_MODEL: 768,
    FAST_MODEL: 384,
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


def _default_threads() -> int | None:
    """Cap ONNX intra-op threads to avoid thrashing on constrained machines.

    On machines with many cores, ONNX defaults to using all of them. When
    multiple ragdoll processes run in parallel (e.g. indexing 7 repos at
    once), each one spawns N threads, causing massive context-switch
    overhead and memory pressure. We cap at half the available cores,
    minimum 2, so a single process is still fast but parallel runs don't
    fight for every core.

    Set RAGDOLL_THREADS to override (0 = let ONNX decide).
    """
    env = os.environ.get("RAGDOLL_THREADS")
    if env is not None:
        val = int(env)
        return val if val > 0 else None  # 0 means "no cap"
    ncpu = os.cpu_count() or 4
    return max(2, ncpu // 2)


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
                providers=providers,
                threads=threads,
                cuda=has_cuda,
            )
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as exc:
            if self.model_name == FALLBACK_MODEL:
                raise RuntimeError(
                    f"Failed to load both primary and fallback embedding models. "
                    f"Ensure you have internet for the initial download, then models "
                    f"are cached locally. Error: {exc}"
                ) from exc
            logger.warning(
                f"Failed to load '{self.model_name}': {exc}. "
                f"Falling back to '{FALLBACK_MODEL}'."
            )
            self.model_name = FALLBACK_MODEL
            try:
                model = TextEmbedding(
                    FALLBACK_MODEL,
                    providers=providers,
                    threads=threads,
                    cuda=has_cuda,
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

        Passes the caller's full batch through to FastEmbed in one shot
        (batch_size=len(texts)) — otherwise FastEmbed silently re-splits
        into batches of 256, which adds a second layer of per-batch overhead
        on top of our own. We've already capped batch size in the indexer.
        """
        if not texts:
            return []
        model = self._load_model()
        return [
            vec.tolist()
            for vec in model.embed(texts, batch_size=len(texts))
        ]

    def _embed_query_uncached(self, query: str) -> tuple[float, ...]:
        """Actual query embedding — returns a tuple so it's hashable/cacheable."""
        model = self._load_model()
        return tuple(next(iter(model.query_embed(query))).tolist())

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
