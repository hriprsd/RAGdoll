"""
Embedder — wraps FastEmbed for fully local ONNX-based embedding.

No PyTorch needed. Total dependency footprint: ~50 MB vs ~2 GB with sentence-transformers.

Default: nomic-ai/nomic-embed-text-v1.5 (768-dim, ONNX, CPU-optimised).
Fallback: BAAI/bge-base-en-v1.5 (768-dim) — same dimensions, no corruption risk.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

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
        """Lazy-load the embedding model, with fallback on failure."""
        if self._model is not None:
            return self._model

        # Check module-level cache first
        if self.model_name in _model_cache:
            self._model = _model_cache[self.model_name]
            return self._model

        from fastembed import TextEmbedding

        try:
            logger.info(f"Loading embedding model: {self.model_name} (ONNX) ...")
            model = TextEmbedding(self.model_name)
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
                model = TextEmbedding(FALLBACK_MODEL)
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
