# RAGdoll

Local-only RAG memory layer for dev tools. Indexes code and docs, serves semantic search over a local vector store.

## Goal
Tool-agnostic — works with Claude Code, Cursor, Copilot, Continue.dev, or anything that can hit an HTTP API.

## Architecture

```
ragdoll serve  →  FastAPI daemon (localhost:7474)
                    ├── /search          core search endpoint (hybrid BM25+vector)
                    ├── /index           manual trigger
                    ├── /status          health check + repo info
                    └── /v1/embeddings   OpenAI-compatible (for Copilot/Continue.dev)

ragdoll mcp    →  MCP stdio server (for Claude Code + Cursor)
                    wraps /search as an MCP tool
```

## Stack
- **Vector store**: SQLite + sqlite-vec + FTS5 — single file at `~/.ragdoll/ragdoll.db`, inspectable with `sqlite3`
- **Embeddings**: nomic-ai/nomic-embed-text-v1.5 via **FastEmbed** (ONNX Runtime, no PyTorch, ~50 MB deps)
- **Search**: Hybrid (BM25 + vector via Reciprocal Rank Fusion), pure vector, or pure BM25
- **File watching**: watchdog with debounce (only used in daemon mode)
- **CLI**: typer + rich

## Key files
- `ragdoll/store.py` — SQLite + sqlite-vec + FTS5 vector store, hybrid search, memory notes, model tracking, FTS schema migration
- `ragdoll/search.py` — pure utilities: vector packing, FTS query sanitization, Reciprocal Rank Fusion
- `ragdoll/embedder.py` — FastEmbed ONNX wrapper, lazy model loading, dimension validation, LRU query cache
- `ragdoll/chunker.py` — AST-aware Python chunking, regex-based TS/JS/Go, markdown splitter, binary/secret detection
- `ragdoll/indexer.py` — per-chunk incremental indexing with content-anchored vector reuse (insertion-at-top doesn't re-embed)
- `ragdoll/watcher.py` — debounced filesystem event handler
- `ragdoll/api.py` — FastAPI routes (search, index, status, OpenAI-compatible embeddings)
- `ragdoll/mcp_server.py` — MCP stdio adapter for Claude Code / Cursor (direct store access, no daemon hop)
- `ragdoll/cli.py` — typer CLI entry point with all commands (index, search, doctor, reindex, export/import, autostart, hooks)

## Dev setup
```bash
pip install -e ".[dev]"

# Index a project (no daemon needed)
ragdoll index ~/ground/projects/RAGdoll

# Search
ragdoll search "how do I add a new chunker"

# Install git hooks so a repo auto-indexes on checkout/merge
ragdoll hooks install ~/ground/projects/RAGdoll

# Optional: start daemon for live MCP with Claude Code / Cursor
ragdoll serve --watch ~/ground/projects
```

## Conventions
- Python 3.11+
- Pydantic v2 throughout
- No cloud calls anywhere — everything runs offline
- FastEmbed (ONNX) for embeddings — no PyTorch dependency
- Don't index: .env files, secrets, node_modules, .git, build artifacts, binaries, files >1 MB
- Chunking: AST-aware (Python), regex-based (TS/JS, Go), heading-based (Markdown), line-window fallback
- Hybrid search by default (BM25 + vector via RRF), with vector-only and BM25-only modes
