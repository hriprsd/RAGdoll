# RAGdoll architecture and data flows

A visual walkthrough of how `ragdoll index`, `ragdoll search`, and the optional
daemon move data around. For the conceptual deep dive (what an embedding is, BM25
via FTS5, Reciprocal Rank Fusion, storage layout), see [how-it-works.md](how-it-works.md).

## Indexing flow

```mermaid
sequenceDiagram
    participant U as User / Git Hook
    participant CLI as ragdoll index
    participant CH as Chunker
    participant EM as Embedder<br/>(FastEmbed ONNX)
    participant DB as SQLite + sqlite-vec + FTS5<br/>~/.ragdoll/ragdoll.db

    U->>CLI: ragdoll index ~/my-project
    CLI->>CLI: walk dirs (prune node_modules,<br/>.git, symlinks, binaries)
    CLI->>CH: chunk_file(path) for each file
    CH-->>CLI: RawChunk[] (AST/regex/heading splits)
    CLI->>CLI: diff content hashes vs DB<br/>(skip unchanged files)
    CLI->>EM: embed(batch of chunk texts)
    EM-->>CLI: float32 vectors (768-dim, ONNX, local)
    CLI->>DB: upsert chunks + vectors + FTS
    DB-->>CLI: done
    CLI-->>U: "42 chunks indexed"
```

## Search flow

```mermaid
sequenceDiagram
    participant U as User / Tool
    participant CLI as ragdoll search<br/>or MCP / HTTP
    participant EM as Embedder
    participant DB as SQLite + sqlite-vec + FTS5

    U->>CLI: "how do we handle auth?"
    CLI->>EM: embed_query(query)
    EM-->>CLI: query vector

    par Hybrid search (default)
        CLI->>DB: vec_chunks KNN (cosine)
        DB-->>CLI: vector results + ranks
        CLI->>DB: fts_chunks MATCH (BM25)
        DB-->>CLI: keyword results + ranks
    end

    CLI->>CLI: Reciprocal Rank Fusion (k=60)
    CLI-->>U: ranked results with file + line refs
```

## Live daemon flow (optional, for MCP integration)

```mermaid
sequenceDiagram
    participant CC as Claude Code / Cursor
    participant MCP as ragdoll mcp<br/>(stdio)
    participant API as ragdoll serve<br/>(localhost:7474)
    participant EM as Embedder
    participant DB as SQLite + sqlite-vec + FTS5
    participant FS as File Watcher

    CC->>MCP: search_codebase("rate limiting")
    MCP->>API: POST /search {mode: "hybrid"}
    API->>EM: embed_query(...)
    EM-->>API: vector
    API->>DB: KNN + BM25 → RRF
    DB-->>API: chunks
    API-->>MCP: JSON results
    MCP-->>CC: formatted code blocks

    Note over FS,DB: In parallel, debounced watcher re-indexes on save
    FS->>API: file changed event (500ms debounce)
    API->>DB: delete + re-embed file
```

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│                     Your machine                          │
│                                                           │
│  ┌──────────────┐   ┌──────────────────────────────┐    │
│  │  Dev tools   │   │      ragdoll daemon           │    │
│  │              │   │   (optional, port 7474)       │    │
│  │ Claude Code ─┼───┤► MCP stdio adapter            │    │
│  │ Cursor      ─┼───┘  FastAPI HTTP server          │    │
│  │ Copilot     ─┼─────► POST /v1/embeddings         │    │
│  │ Continue.dev─┼─────► POST /search                │    │
│  └──────────────┘   └────────────┬─────────────────┘    │
│                                   │                       │
│  ┌────────────────────────────────▼──────────────────┐   │
│  │              ragdoll CLI (no daemon needed)        │   │
│  │                                                    │   │
│  │  ragdoll index   →  Chunker + Embedder             │   │
│  │  ragdoll search  →  Embedder + VectorStore         │   │
│  │  ragdoll context →  Search + token-budgeted pack   │   │
│  │  ragdoll hooks   →  git post-checkout/merge        │   │
│  └────────────────────────────────┬──────────────────┘   │
│                                   │                       │
│  ┌────────────────────────────────▼──────────────────┐   │
│  │         ~/.ragdoll/ragdoll.db                      │   │
│  │                                                    │   │
│  │  chunks      - content, path, repo, lang, hash     │   │
│  │  vec_chunks  - sqlite-vec 768-dim float32 vectors  │   │
│  │  fts_chunks  - FTS5 BM25 index over content        │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```
