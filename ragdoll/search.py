"""
Search utilities - pure functions used by the VectorStore search methods.

Split out of store.py so the storage layer stays focused on persistence.
These are deliberately stateless and connection-free; `VectorStore` calls
them with rows it fetched from SQLite.
"""

from __future__ import annotations

import struct
from typing import Literal

SearchMode = Literal["hybrid", "vector", "bm25"]

RRF_K = 60  # standard RRF constant - higher = less weight on top ranks


def pack_vector(v: list[float]) -> bytes:
    """Pack a float list into the little-endian float32 blob sqlite-vec expects."""
    return struct.pack(f"<{len(v)}f", *v)


def build_fts_query(text: str) -> str:
    """
    Convert a free-form query into a safe FTS5 MATCH expression.

    Each whitespace-delimited token is wrapped in double quotes so FTS5
    treats special chars (dots, colons, hyphens) as literal content rather
    than operators. Tokens are joined with OR for partial matching.
    """
    tokens: list[str] = []
    for t in text.split():
        if len(t) <= 1:
            continue
        escaped = t.replace('"', '""')
        tokens.append(f'"{escaped}"')
    return " OR ".join(tokens)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """
    Combine multiple ranked lists of ids using Reciprocal Rank Fusion.

    Each list is an ordered sequence of ids (best first). An id appearing in
    multiple lists gets summed - agreement across signals means higher score.

    Returns {id: fused_score}.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
