"""
Vector store — SQLite + sqlite-vec + FTS5 for local, file-based persistence.

Single file at ~/.ragdoll/ragdoll.db. Inspect any time with:
  sqlite3 ~/.ragdoll/ragdoll.db

Schema:
  chunks      — metadata (source_path, repo, language, lines, content, hash, indexed_at)
  vec_chunks  — sqlite-vec virtual table, float32 embeddings
  fts_chunks  — FTS5 virtual table, BM25 full-text index over content

Search modes:
  vector  — pure KNN cosine similarity (good for semantic / natural language)
  bm25    — pure FTS5 BM25 (good for exact identifiers, error messages)
  hybrid  — Reciprocal Rank Fusion of both (best overall, default)
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import sqlite3
import sqlite_vec
from pydantic import BaseModel

from .search import (
    RRF_K,
    SearchMode,
    build_fts_query as _build_fts_query,
    pack_vector as _pack_vector,
    reciprocal_rank_fusion,
)

if TYPE_CHECKING:
    from .embedder import Embedder

logger = logging.getLogger(__name__)

EMBED_DIM = 768  # nomic-embed-text-v1.5 / bge-base-en-v1.5 (fallback)

# Bump when the FTS schema changes so existing DBs get rebuilt automatically
FTS_SCHEMA_VERSION = "2"  # v2: porter stemmer for better prose recall
FTS_TOKENIZE = "porter unicode61 remove_diacritics 1"

# Distance metric for the sqlite-vec table. cosine is magnitude-invariant, so
# ranking stays correct even for vectors that aren't perfectly unit-length.
# sqlite-vec defaults to L2, which only ranks correctly on normalized vectors —
# a silent footgun that produced garbage results on unnormalized embeddings.
VEC_DISTANCE_METRIC = "cosine"
# Bump when the vec0 table definition changes (e.g. distance metric). Vectors
# live only in vec_chunks and can't be backfilled from the chunks table, so a
# bump forces a rebuild and requires `ragdoll reindex` to re-embed.
VEC_SCHEMA_VERSION = "2"  # v2: cosine distance metric + unit-normalized vectors

MEMORY_REPO   = "ragdoll://memory"
MEMORY_PREFIX = "ragdoll://memory/"


class SearchResult(BaseModel):
    content: str
    source_path: str
    repo: str
    language: str
    start_line: int
    end_line: int
    score: float


class RepoSummary(BaseModel):
    repo: str
    chunks: int
    languages: list[str]
    last_indexed: Optional[str]   # ISO timestamp or None for legacy rows


class VectorStore:
    def __init__(self, db_path: Path, dim: int = EMBED_DIM):
        """Open (creating if needed) the vector store at `db_path`.

        `dim` is the embedding dimension the vec0 table is built with. It
        defaults to 768 (nomic / bge-base) so existing callers are unaffected;
        the CLI passes 384 for `--fast` (bge-small) into its separate
        ragdoll-fast.db so the two dimensions never share a table.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._dim = dim
        self._init_db()
        self._migrate_db()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path)
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        con.row_factory = sqlite3.Row
        # Keep FTS and vec tables in sync with main table on cascade
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=5000")  # wait up to 5s for locks
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id           TEXT PRIMARY KEY,
                    content      TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_path  TEXT NOT NULL,
                    repo         TEXT NOT NULL,
                    language     TEXT NOT NULL,
                    chunk_index  INTEGER NOT NULL,
                    start_line   INTEGER NOT NULL,
                    end_line     INTEGER NOT NULL,
                    indexed_at   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_repo   ON chunks(repo);

                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    id     TEXT PRIMARY KEY,
                    vector float[{self._dim}] distance_metric={VEC_DISTANCE_METRIC}
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    id      UNINDEXED,
                    content,
                    tokenize = '{FTS_TOKENIZE}'
                );
            """)

    def _migrate_db(self) -> None:
        """Apply incremental schema migrations for users upgrading from older builds."""
        with self._conn() as con:
            # 1. Add indexed_at if missing (pre-timestamp builds)
            existing_cols = {
                r[1] for r in con.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "indexed_at" not in existing_cols:
                con.execute("ALTER TABLE chunks ADD COLUMN indexed_at TEXT")
                logger.info("Migrated: added indexed_at column")

            # 2. Backfill FTS table if it's empty but chunks exist
            fts_count = con.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]
            chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if fts_count == 0 and chunk_count > 0:
                con.execute(
                    "INSERT INTO fts_chunks(id, content) SELECT id, content FROM chunks"
                )
                logger.info(f"Migrated: backfilled FTS index for {chunk_count} existing chunks")

            # 3. Create ragdoll_meta table for tracking embed model info
            con.execute("""
                CREATE TABLE IF NOT EXISTS ragdoll_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # 4. If FTS schema version changed, drop + recreate FTS table
            # and backfill from chunks. Vectors are untouched.
            stored_fts_version = con.execute(
                "SELECT value FROM ragdoll_meta WHERE key = 'fts_schema_version'"
            ).fetchone()
            current_version = stored_fts_version[0] if stored_fts_version else None
            if current_version != FTS_SCHEMA_VERSION:
                logger.info(
                    f"Rebuilding FTS index (schema {current_version} -> {FTS_SCHEMA_VERSION})"
                )
                con.execute("DROP TABLE IF EXISTS fts_chunks")
                con.execute(f"""
                    CREATE VIRTUAL TABLE fts_chunks USING fts5(
                        id      UNINDEXED,
                        content,
                        tokenize = '{FTS_TOKENIZE}'
                    )
                """)
                con.execute(
                    "INSERT INTO fts_chunks(id, content) SELECT id, content FROM chunks"
                )
                con.execute(
                    "INSERT OR REPLACE INTO ragdoll_meta(key, value) VALUES (?, ?)",
                    ("fts_schema_version", FTS_SCHEMA_VERSION),
                )

            # 5. If the vec schema changed (e.g. distance metric), recreate the
            # vec table. Vectors live only here and can't be backfilled from the
            # chunks table, so this empties them — the user must run
            # `ragdoll reindex` to re-embed. Recreate eagerly so any new writes
            # land in a correctly-configured (cosine) table.
            stored_vec_version = con.execute(
                "SELECT value FROM ragdoll_meta WHERE key = 'vec_schema_version'"
            ).fetchone()
            vec_version = stored_vec_version[0] if stored_vec_version else None
            if vec_version != VEC_SCHEMA_VERSION:
                vec_count = con.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
                con.execute("DROP TABLE IF EXISTS vec_chunks")
                con.execute(f"""
                    CREATE VIRTUAL TABLE vec_chunks USING vec0(
                        id     TEXT PRIMARY KEY,
                        vector float[{self._dim}] distance_metric={VEC_DISTANCE_METRIC}
                    )
                """)
                con.execute(
                    "INSERT OR REPLACE INTO ragdoll_meta(key, value) VALUES (?, ?)",
                    ("vec_schema_version", VEC_SCHEMA_VERSION),
                )
                if vec_count > 0:
                    logger.warning(
                        f"Vector index rebuilt for schema v{vec_version} -> "
                        f"v{VEC_SCHEMA_VERSION} ({vec_count} vectors dropped). "
                        f"Run `ragdoll reindex` to re-embed your repos."
                    )

    # ------------------------------------------------------------------
    # Model version tracking
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        """Read a value from the ragdoll_meta key-value table."""
        with self._conn() as con:
            row = con.execute(
                "SELECT value FROM ragdoll_meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Write a value to the ragdoll_meta key-value table."""
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO ragdoll_meta(key, value) VALUES (?, ?)",
                (key, value),
            )

    def check_embed_model(self, model_name: str, dim: int) -> None:
        """Record or validate the embedding model used for this DB.

        On first use, stores the model name and dimension.
        On subsequent calls, warns (via logger) if there's a mismatch —
        this means the user switched models and vectors are inconsistent.
        """
        stored_model = self.get_meta("embed_model")
        stored_dim = self.get_meta("embed_dim")

        if not stored_model:
            # First use, or cleared by `reindex` (which blanks these to "").
            # Treat empty/None alike, otherwise the blanked value never gets
            # re-recorded and check_embed_model warns "mismatch" on every run.
            self.set_meta("embed_model", model_name)
            self.set_meta("embed_dim", str(dim))
            return

        if stored_model != model_name:
            logger.warning(
                f"Embedding model mismatch! DB was built with '{stored_model}' "
                f"but current model is '{model_name}'. "
                f"Run 'ragdoll reindex' to rebuild with the new model."
            )
        if stored_dim and int(stored_dim) != dim:
            logger.warning(
                f"Embedding dimension mismatch! DB has {stored_dim}-dim vectors "
                f"but current model produces {dim}-dim. "
                f"Run 'ragdoll reindex' to rebuild."
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as con:
            for c in chunks:
                con.execute("""
                    INSERT OR REPLACE INTO chunks
                        (id, content, content_hash, source_path, repo, language,
                         chunk_index, start_line, end_line, indexed_at)
                    VALUES
                        (:id, :content, :content_hash, :source_path, :repo, :language,
                         :chunk_index, :start_line, :end_line, :indexed_at)
                """, {**c, "indexed_at": now})
                # vec0 and fts5 virtual tables don't honour INSERT OR REPLACE
                # predictably — use delete-then-insert for both
                con.execute("DELETE FROM vec_chunks WHERE id = ?", (c["id"],))
                con.execute(
                    "INSERT INTO vec_chunks(id, vector) VALUES (?, ?)",
                    (c["id"], _pack_vector(c["vector"])),
                )
                con.execute("DELETE FROM fts_chunks WHERE id = ?", (c["id"],))
                con.execute(
                    "INSERT INTO fts_chunks(id, content) VALUES (?, ?)",
                    (c["id"], c["content"]),
                )

    def delete_by_source(self, source_path: str) -> None:
        with self._conn() as con:
            ids = [
                r[0] for r in con.execute(
                    "SELECT id FROM chunks WHERE source_path = ?", (source_path,)
                ).fetchall()
            ]
            if not ids:
                return
            self._delete_ids(con, ids)

    def delete_by_prefix(self, path_prefix: str) -> int:
        """Remove all chunks whose source_path starts with path_prefix. Returns count removed."""
        with self._conn() as con:
            ids = [
                r[0] for r in con.execute(
                    "SELECT id FROM chunks WHERE source_path LIKE ?",
                    (f"{path_prefix}%",),
                ).fetchall()
            ]
            if not ids:
                return 0
            self._delete_ids(con, ids)
        return len(ids)

    def _delete_ids(self, con: sqlite3.Connection, ids: list[str]) -> None:
        placeholders = ",".join("?" * len(ids))
        con.execute(f"DELETE FROM chunks     WHERE id IN ({placeholders})", ids)
        con.execute(f"DELETE FROM vec_chunks WHERE id IN ({placeholders})", ids)
        con.execute(f"DELETE FROM fts_chunks WHERE id IN ({placeholders})", ids)

    # ------------------------------------------------------------------
    # Memory (explicit notes stored alongside code)
    # ------------------------------------------------------------------

    def add_memory(
        self,
        text: str,
        tags: list[str] | None = None,
        embedder: "Embedder | None" = None,
    ) -> str:
        """Store a free-text note. Returns the memory ID.

        Args:
            embedder: An Embedder instance. Pass one in — creating a new one per
                      call works but wastes time on the lazy model load check.
        """
        if embedder is None:
            from .embedder import Embedder
            embedder = Embedder()

        memory_id = hashlib.sha256(
            f"{text}{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]
        source_path = f"{MEMORY_PREFIX}{memory_id}"
        tag_line = f"[tags: {', '.join(tags)}]\n" if tags else ""
        full_content = f"{tag_line}{text}"

        vector = embedder.embed([full_content])[0]

        self.upsert([{
            "id": self.chunk_id(source_path, 0),
            "content": full_content,
            "content_hash": self.content_hash(full_content),
            "source_path": source_path,
            "repo": MEMORY_REPO,
            "language": "note",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 1,
            "vector": vector,
        }])
        return memory_id

    def list_memories(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute("""
                SELECT source_path, content, indexed_at
                FROM   chunks
                WHERE  repo = ?
                ORDER  BY indexed_at DESC
            """, (MEMORY_REPO,)).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, memory_id: str) -> bool:
        source_path = f"{MEMORY_PREFIX}{memory_id}"
        with self._conn() as con:
            ids = [
                r[0] for r in con.execute(
                    "SELECT id FROM chunks WHERE source_path = ?", (source_path,)
                ).fetchall()
            ]
            if not ids:
                return False
            self._delete_ids(con, ids)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        query_text: str = "",
        top_k: int = 10,
        repo: Optional[str] = None,
        mode: SearchMode = "hybrid",
        include_memories: bool = True,
    ) -> list[SearchResult]:
        if mode == "vector":
            return self._vector_search(query_vector, top_k, repo, include_memories)
        if mode == "bm25":
            return self._bm25_search(query_text, top_k, repo, include_memories)
        return self._hybrid_search(query_vector, query_text, top_k, repo, include_memories)

    def _vector_search(
        self, query_vector: list[float], top_k: int, repo: Optional[str], include_memories: bool
    ) -> list[SearchResult]:
        fetch_k = top_k * 5 if (repo or not include_memories) else top_k
        with self._conn() as con:
            rows = con.execute("""
                SELECT c.content, c.source_path, c.repo, c.language,
                       c.start_line, c.end_line, v.distance AS score
                FROM (
                    SELECT id, distance
                    FROM   vec_chunks
                    WHERE  vector MATCH ?
                    AND    k = ?
                ) v
                JOIN chunks c ON c.id = v.id
                ORDER BY v.distance
            """, (_pack_vector(query_vector), fetch_k)).fetchall()
        return self._filter_and_limit(rows, top_k, repo, include_memories, score_key="score")

    def _bm25_search(
        self, query_text: str, top_k: int, repo: Optional[str], include_memories: bool
    ) -> list[SearchResult]:
        fts_query = _build_fts_query(query_text)
        if not fts_query:
            return []
        fetch_k = top_k * 5 if (repo or not include_memories) else top_k
        with self._conn() as con:
            rows = con.execute("""
                SELECT c.content, c.source_path, c.repo, c.language,
                       c.start_line, c.end_line,
                       -fts.rank AS score
                FROM fts_chunks fts
                JOIN chunks c ON c.id = fts.id
                WHERE fts_chunks MATCH ?
                ORDER BY fts.rank
                LIMIT ?
            """, (fts_query, fetch_k)).fetchall()
        return self._filter_and_limit(rows, top_k, repo, include_memories, score_key="score")

    def _hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        repo: Optional[str],
        include_memories: bool,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of vector + BM25 results."""
        fetch_k = top_k * 5

        with self._conn() as con:
            # Vector results
            vec_rows = con.execute("""
                SELECT id, distance
                FROM   vec_chunks
                WHERE  vector MATCH ?
                AND    k = ?
                ORDER  BY distance
            """, (_pack_vector(query_vector), fetch_k)).fetchall()

            # BM25 results
            fts_query = _build_fts_query(query_text)
            bm25_rows = []
            if fts_query:
                try:
                    bm25_rows = con.execute("""
                        SELECT id, rank
                        FROM   fts_chunks
                        WHERE  fts_chunks MATCH ?
                        ORDER  BY rank
                        LIMIT  ?
                    """, (fts_query, fetch_k)).fetchall()
                except sqlite3.OperationalError:
                    # Malformed FTS query — degrade gracefully to vector-only
                    logger.debug(f"FTS query failed for '{fts_query}', using vector only")

            # RRF fusion
            rrf_scores = reciprocal_rank_fusion(
                [[r[0] for r in vec_rows], [r[0] for r in bm25_rows]],
                k=RRF_K,
            )

            # Fetch metadata for merged candidate ids
            if not rrf_scores:
                return []
            candidate_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:fetch_k]
            placeholders = ",".join("?" * len(candidate_ids))
            meta = {
                r["id"]: r for r in con.execute(
                    f"SELECT * FROM chunks WHERE id IN ({placeholders})", candidate_ids
                ).fetchall()
            }

        results: list[SearchResult] = []
        for chunk_id in candidate_ids:
            if chunk_id not in meta:
                continue
            r = meta[chunk_id]
            if repo and r["repo"] != repo:
                continue
            if not include_memories and r["repo"] == MEMORY_REPO:
                continue
            results.append(SearchResult(
                content=r["content"],
                source_path=r["source_path"],
                repo=r["repo"],
                language=r["language"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                score=rrf_scores[chunk_id],
            ))
            if len(results) >= top_k:
                break
        return results

    def _filter_and_limit(
        self, rows, top_k: int, repo: Optional[str], include_memories: bool, score_key: str
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for r in rows:
            if repo and r["repo"] != repo:
                continue
            if not include_memories and r["repo"] == MEMORY_REPO:
                continue
            results.append(SearchResult(
                content=r["content"],
                source_path=r["source_path"],
                repo=r["repo"],
                language=r["language"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                score=float(r[score_key]),
            ))
            if len(results) >= top_k:
                break
        return results

    def explain(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        repo: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search with per-result scoring breakdown for debugging.

        Returns list of dicts with: source_path, start_line, end_line, language,
        content, vector_rank, bm25_rank, rrf_score.
        """
        fetch_k = top_k * 5
        with self._conn() as con:
            vec_rows = con.execute(
                """
                SELECT id, distance FROM vec_chunks
                WHERE vector MATCH ? AND k = ?
                ORDER BY distance
                """,
                (_pack_vector(query_vector), fetch_k),
            ).fetchall()

            fts_query = _build_fts_query(query_text)
            bm25_rows = []
            if fts_query:
                try:
                    bm25_rows = con.execute(
                        """
                        SELECT id, rank FROM fts_chunks
                        WHERE fts_chunks MATCH ?
                        ORDER BY rank LIMIT ?
                        """,
                        (fts_query, fetch_k),
                    ).fetchall()
                except sqlite3.OperationalError:
                    pass

            vec_rank = {row[0]: i + 1 for i, row in enumerate(vec_rows)}
            bm25_rank = {row[0]: i + 1 for i, row in enumerate(bm25_rows)}

            rrf_scores = reciprocal_rank_fusion(
                [[r[0] for r in vec_rows], [r[0] for r in bm25_rows]],
                k=RRF_K,
            )
            if not rrf_scores:
                return []
            top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:fetch_k]
            placeholders = ",".join("?" * len(top_ids))
            meta = {
                r["id"]: r
                for r in con.execute(
                    f"SELECT * FROM chunks WHERE id IN ({placeholders})", top_ids
                ).fetchall()
            }

        out: list[dict] = []
        for cid in top_ids:
            if cid not in meta:
                continue
            r = meta[cid]
            if repo and r["repo"] != repo:
                continue
            out.append({
                "source_path": r["source_path"],
                "language": r["language"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "content": r["content"],
                "vector_rank": vec_rank.get(cid),
                "bm25_rank": bm25_rank.get(cid),
                "rrf_score": rrf_scores[cid],
            })
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
        """Remove overlapping chunks from the same file, keeping the higher-scored one.

        Two chunks overlap if they share the same source_path and their line
        ranges intersect (a.start <= b.end and b.start <= a.end).
        """
        kept: list[SearchResult] = []
        for r in results:
            is_dup = False
            for existing in kept:
                if (
                    r.source_path == existing.source_path
                    and r.start_line <= existing.end_line
                    and existing.start_line <= r.end_line
                ):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(r)
        return kept

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_repos(self) -> list[RepoSummary]:
        with self._conn() as con:
            rows = con.execute("""
                SELECT
                    repo,
                    COUNT(*)                              AS chunks,
                    GROUP_CONCAT(DISTINCT language)       AS languages,
                    MAX(indexed_at)                       AS last_indexed
                FROM chunks
                WHERE repo != ?
                GROUP BY repo
                ORDER BY last_indexed DESC NULLS LAST
            """, (MEMORY_REPO,)).fetchall()
        return [
            RepoSummary(
                repo=r["repo"],
                chunks=r["chunks"],
                languages=sorted((r["languages"] or "").split(",")),
                last_indexed=r["last_indexed"],
            )
            for r in rows
        ]

    def stats_breakdown(self) -> list[dict]:
        with self._conn() as con:
            return [
                dict(r) for r in con.execute("""
                    SELECT
                        repo,
                        language,
                        COUNT(*) AS chunks,
                        MAX(indexed_at) AS last_indexed
                    FROM chunks
                    WHERE repo != ?
                    GROUP BY repo, language
                    ORDER BY repo, chunks DESC
                """, (MEMORY_REPO,)).fetchall()
            ]

    def known_hashes(self, source_path: str) -> set[str]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT content_hash FROM chunks WHERE source_path = ?", (source_path,)
            ).fetchall()
        return {r[0] for r in rows}

    def hashes_by_index(self, source_path: str) -> dict[int, str]:
        """Return {chunk_index: content_hash} for a given source file.

        Used for per-chunk incremental indexing — skip embedding chunks whose
        content hash is unchanged at the same position.
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT chunk_index, content_hash FROM chunks WHERE source_path = ?",
                (source_path,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_chunks_by_index(
        self, source_path: str, indices: list[int]
    ) -> None:
        """Delete specific chunks of a file by their chunk_index values."""
        if not indices:
            return
        with self._conn() as con:
            ids = [
                self.chunk_id(source_path, i) for i in indices
            ]
            self._delete_ids(con, ids)

    def all_hashes(self) -> dict[str, set[str]]:
        """Preload all content hashes, keyed by source_path.

        Use this before a bulk indexing run so we don't hit the DB per file.
        """
        result: dict[str, set[str]] = {}
        with self._conn() as con:
            rows = con.execute("SELECT source_path, content_hash FROM chunks").fetchall()
        for r in rows:
            result.setdefault(r[0], set()).add(r[1])
        return result

    def all_hashes_by_index(self) -> dict[str, dict[int, str]]:
        """Preload {source_path: {chunk_index: content_hash}} for bulk indexing.

        Lets the indexer do per-chunk incremental updates without one query per file.
        """
        result: dict[str, dict[int, str]] = {}
        with self._conn() as con:
            rows = con.execute(
                "SELECT source_path, chunk_index, content_hash FROM chunks"
            ).fetchall()
        for r in rows:
            result.setdefault(r[0], {})[r[1]] = r[2]
        return result

    def vectors_by_hash(self, source_path: str) -> dict[str, list[float]]:
        """Return {content_hash: vector} for every chunk in a file.

        Used to reuse embeddings when a chunk's *content* is unchanged but its
        *position* in the file has shifted (e.g. a new function inserted above).
        """
        import struct as _struct
        result: dict[str, list[float]] = {}
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT c.content_hash, v.vector
                FROM   chunks c
                JOIN   vec_chunks v ON v.id = c.id
                WHERE  c.source_path = ?
                """,
                (source_path,),
            ).fetchall()
        for r in rows:
            blob = r["vector"]
            n = len(blob) // 4
            result[r["content_hash"]] = list(_struct.unpack(f"<{n}f", blob))
        return result

    def all_vectors_by_hash(self) -> dict[str, dict[str, list[float]]]:
        """Preload {source_path: {content_hash: vector}} for the entire DB.

        Expensive on large DBs but a one-shot cost per indexing run.
        """
        import struct as _struct
        result: dict[str, dict[str, list[float]]] = {}
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT c.source_path, c.content_hash, v.vector
                FROM   chunks c
                JOIN   vec_chunks v ON v.id = c.id
                """
            ).fetchall()
        for r in rows:
            blob = r["vector"]
            n = len(blob) // 4
            result.setdefault(r["source_path"], {})[r["content_hash"]] = \
                list(_struct.unpack(f"<{n}f", blob))
        return result

    def iter_all_chunks(self):
        """Yield every chunk row with its vector. Used by export."""
        with self._conn() as con:
            rows = con.execute("""
                SELECT c.id, c.content, c.content_hash, c.source_path, c.repo,
                       c.language, c.chunk_index, c.start_line, c.end_line,
                       c.indexed_at, v.vector
                FROM   chunks c
                JOIN   vec_chunks v ON v.id = c.id
            """).fetchall()
        for r in rows:
            vec_blob = r["vector"]
            # Unpack the float32 blob
            import struct as _struct
            n = len(vec_blob) // 4
            vec = list(_struct.unpack(f"<{n}f", vec_blob))
            yield {
                "id": r["id"],
                "content": r["content"],
                "content_hash": r["content_hash"],
                "source_path": r["source_path"],
                "repo": r["repo"],
                "language": r["language"],
                "chunk_index": r["chunk_index"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "indexed_at": r["indexed_at"],
                "vector": vec,
            }

    def count(self) -> int:
        with self._conn() as con:
            return con.execute(
                "SELECT COUNT(*) FROM chunks WHERE repo != ?", (MEMORY_REPO,)
            ).fetchone()[0]

    def chunk_counts_by_file(self) -> dict[str, int]:
        """Return {source_path: chunk_count} for all indexed files."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT source_path, COUNT(*) FROM chunks "
                "WHERE repo != ? GROUP BY source_path",
                (MEMORY_REPO,),
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_id(source_path: str, chunk_index: int) -> str:
        return hashlib.sha256(f"{source_path}:{chunk_index}".encode()).hexdigest()

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
