"""
RAGdoll local API server — tool-agnostic HTTP interface.

Exposes:
  POST /search          — semantic search (used by all tool adapters)
  POST /index           — manually trigger indexing of a path
  GET  /status          — health + stats
  POST /v1/embeddings   — OpenAI-compatible embeddings endpoint
                          (works with Copilot, Continue.dev, any tool that
                          supports custom OpenAI-compatible embedding endpoints)

The MCP adapter (mcp_server.py) calls /search internally.
Everything routes through this single daemon so the vector store
is never accessed by two processes simultaneously.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="RAGdoll", version="0.1.0")

# Injected at startup by cli.py via init()
_indexer = None
_embedder = None
_store = None


def init(indexer, embedder, store) -> None:
    global _indexer, _embedder, _store
    _indexer = indexer
    _embedder = embedder
    _store = store


def _require_ready() -> None:
    """Raise 503 if the stack hasn't been initialised yet."""
    if _indexer is None or _embedder is None or _store is None:
        raise HTTPException(503, "RAGdoll not initialised — call init() at startup")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

from typing import Literal as L


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    repo: Optional[str] = None
    mode: L["hybrid", "vector", "bm25"] = "hybrid"


class SearchResponse(BaseModel):
    results: list[dict]


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    _require_ready()
    vec = _embedder.embed_query(req.query)
    results = _store.search(
        query_vector=vec,
        query_text=req.query,       # ← was missing — hybrid degraded to vector-only
        top_k=req.top_k,
        repo=req.repo,
        mode=req.mode,
    )
    return SearchResponse(results=[r.model_dump() for r in results])


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    path: str


@app.post("/index")
async def index(req: IndexRequest):
    _require_ready()
    p = Path(req.path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(404, f"Path not found: {p}")
    # Refuse to index paths outside the user's home directory as a basic guard
    try:
        p.relative_to(Path.home())
    except ValueError:
        raise HTTPException(400, "Only paths inside your home directory may be indexed")
    count = _indexer.index_path(p)
    return {"indexed": count, "path": str(p)}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    if _store is None:
        return {"status": "not_initialised", "chunks": 0, "repos": []}
    repos = _store.list_repos()
    return {
        "status": "ok",
        "chunks": _store.count(),
        "repos": [
            {"repo": r.repo, "chunks": r.chunks, "last_indexed": r.last_indexed}
            for r in repos
        ],
        "db": str(_store.db_path),
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible embeddings endpoint
# Allows tools that support a custom OpenAI embeddings base URL
# (Continue.dev, some Copilot extensions, custom scripts) to use RAGdoll
# as a drop-in local embedding provider.
# ---------------------------------------------------------------------------

class OAIEmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "ragdoll"
    encoding_format: str = "float"


@app.post("/v1/embeddings")
async def oai_embeddings(req: OAIEmbeddingRequest):
    _require_ready()
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = _embedder.embed(texts)
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        # Token counting is intentionally omitted — we don't use a tokeniser here
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
