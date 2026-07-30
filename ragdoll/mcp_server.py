"""
MCP server adapter for RAGdoll.

Exposes RAGdoll's search capability as an MCP tool so that
Claude Code and Cursor (both support MCP) can call it natively.

To register with Claude Code, add to ~/.claude/settings.json:
  "mcpServers": {
    "ragdoll": {
      "command": "ragdoll",
      "args": ["mcp"]
    }
  }

To register with Cursor, add to .cursor/mcp.json in your project:
  {
    "mcpServers": {
      "ragdoll": {
        "command": "ragdoll",
        "args": ["mcp"]
      }
    }
  }

The MCP server opens the vector store directly - no daemon required.
This removes a dead process risk: the watcher is a separate concern.
If the daemon is running (for live file watching), the MCP server
still just reads the same SQLite DB independently.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .embedder import Embedder
from .store import VectorStore

logger = logging.getLogger(__name__)
server = Server("ragdoll")


def _db_path() -> Path:
    """Resolve the DB the MCP server should read from, respecting RAGDOLL_DB."""
    env = os.environ.get("RAGDOLL_DB")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".ragdoll" / "ragdoll.db"


# Lazily constructed on first tool call
_store: VectorStore | None = None
_embedder: Embedder | None = None


def _ensure_ready() -> tuple[VectorStore, Embedder]:
    global _store, _embedder
    if _store is None:
        _store = VectorStore(_db_path())
    if _embedder is None:
        _embedder = Embedder()
    return _store, _embedder


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_codebase",
            description=(
                "Search your local codebase and past work using semantic similarity. "
                "Use this to find relevant code, patterns, or decisions from any indexed project. "
                "Returns the most relevant code chunks with file paths and line numbers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you're looking for - natural language or code snippet",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 8,
                        "description": "Number of results to return",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Filter to a specific repo root path (optional)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "vector", "bm25"],
                        "default": "hybrid",
                        "description": "Search mode: hybrid (default), vector (semantic), bm25 (keyword)",
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "search_codebase":
        raise ValueError(f"Unknown tool: {name}")

    try:
        store, embedder = _ensure_ready()
        query = arguments.get("query", "")
        if not query:
            return [TextContent(type="text", text="Empty query.")]

        vec = embedder.embed_query(query)
        results = store.search(
            query_vector=vec,
            query_text=query,
            top_k=int(arguments.get("top_k", 8)),
            repo=arguments.get("repo"),
            mode=arguments.get("mode", "hybrid"),
        )
    except Exception as exc:
        logger.exception("RAGdoll MCP tool failed")
        return [TextContent(type="text", text=f"RAGdoll error: {exc}")]

    if not results:
        return [TextContent(type="text", text="No relevant results found.")]

    lines: list[str] = []
    for r in results:
        lines.append(f"### {r.source_path} (lines {r.start_line}-{r.end_line})")
        lines.append(f"```{r.language}")
        lines.append(r.content)
        lines.append("```")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)
