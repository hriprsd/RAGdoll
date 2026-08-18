"""
Integration tests — index the fixture repo and verify search quality.

Uses a deterministic fake embedder so tests run fast without downloading
the 200MB ONNX model. The fake embedder produces vectors based on simple
word overlap (bag-of-words cosine), which is enough to verify the pipeline
works end-to-end. BM25 tests don't need embeddings at all.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path

import pytest

from ragdoll.store import VectorStore
from ragdoll.indexer import Indexer
from ragdoll.chunker import chunk_file

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fake embedder — deterministic bag-of-words vectors
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    """Produce 768-dim vectors via hashed word counts. Deterministic and fast."""

    model_name = "test/fake-embedder"
    dim = 768

    def unload(self) -> None:
        """No-op for tests — no ONNX session to release."""
        pass

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * 768
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 768
            vec[idx] += 1.0
        # L2-normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vectorize(query)


@pytest.fixture()
def indexed_store(tmp_path):
    """Index the test fixtures into a temp DB and return (store, embedder)."""
    db_path = tmp_path / "test.db"
    store = VectorStore(db_path)
    embedder = _FakeEmbedder()
    indexer = Indexer(store, embedder)
    n = indexer.index_path(FIXTURES)
    assert n > 0, "Fixture indexing produced no chunks"
    return store, embedder


# ---------------------------------------------------------------------------
# Pipeline sanity
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_fixtures_produce_chunks(self):
        """Every fixture file should produce at least one chunk."""
        for f in FIXTURES.iterdir():
            if f.name.startswith(".") or f.name == "__pycache__":
                continue
            chunks = chunk_file(f)
            assert chunks, f"{f.name} produced no chunks"

    def test_index_populates_db(self, indexed_store):
        store, _ = indexed_store
        assert store.count() > 10, "Expected at least 10 chunks from fixtures"

    def test_multiple_languages_indexed(self, indexed_store):
        store, _ = indexed_store
        repos = store.list_repos()
        assert repos, "No repos in store after indexing"
        languages = repos[0].languages
        # We have .py, .go, .ts, .yaml, .md files
        assert len(languages) >= 3, f"Expected 3+ languages, got {languages}"


# ---------------------------------------------------------------------------
# BM25 search quality — exact identifier matches
# ---------------------------------------------------------------------------

class TestBM25:
    def test_exact_function_name(self, indexed_store):
        store, embedder = indexed_store
        vec = embedder.embed_query("validate_token")
        results = store.search(vec, query_text="validate_token", top_k=5, mode="bm25")
        assert results, "BM25 found nothing for 'validate_token'"
        # The auth.py file should be in results
        paths = [r.source_path for r in results]
        assert any("auth.py" in p for p in paths), f"auth.py not in BM25 results: {paths}"

    def test_exact_class_name(self, indexed_store):
        store, embedder = indexed_store
        vec = embedder.embed_query("PaymentProcessor")
        results = store.search(vec, query_text="PaymentProcessor", top_k=5, mode="bm25")
        assert results, "BM25 found nothing for 'PaymentProcessor'"
        paths = [r.source_path for r in results]
        assert any("payments.py" in p for p in paths), f"payments.py not in results: {paths}"

    def test_cross_language_identifier(self, indexed_store):
        """'RateLimitMiddleware' exists in both Go and Python — BM25 should find both."""
        store, embedder = indexed_store
        vec = embedder.embed_query("RateLimitMiddleware")
        results = store.search(vec, query_text="RateLimitMiddleware", top_k=10, mode="bm25")
        paths = [r.source_path for r in results]
        # Go file has the exact identifier
        assert any("middleware.go" in p for p in paths), f"middleware.go not in results: {paths}"


# ---------------------------------------------------------------------------
# Hybrid search quality
# ---------------------------------------------------------------------------

class TestHybrid:
    def test_conceptual_query(self, indexed_store):
        """A natural language query should find relevant code."""
        store, embedder = indexed_store
        vec = embedder.embed_query("how do we handle authentication")
        results = store.search(vec, query_text="how do we handle authentication", top_k=5)
        assert results, "Hybrid found nothing for auth query"
        # At least one result should be from auth-related files
        content_joined = " ".join(r.content.lower() for r in results)
        assert "token" in content_joined or "auth" in content_joined, \
            "No auth-related content in hybrid results"

    def test_payment_query(self, indexed_store):
        store, embedder = indexed_store
        vec = embedder.embed_query("charge refund payment processing")
        results = store.search(vec, query_text="charge refund payment processing", top_k=5)
        assert results, "No results for payment query"
        paths = [r.source_path for r in results]
        assert any("payment" in p.lower() for p in paths), \
            f"No payment file in results: {paths}"

    def test_pagination_query(self, indexed_store):
        store, embedder = indexed_store
        vec = embedder.embed_query("pagination offset limit")
        results = store.search(vec, query_text="pagination offset limit", top_k=5)
        assert results, "No results for pagination query"
        content_joined = " ".join(r.content.lower() for r in results)
        assert "offset" in content_joined or "limit" in content_joined or "paginate" in content_joined


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_overlapping_chunks_removed(self):
        """Overlapping chunks from the same file should be deduped."""
        from ragdoll.store import SearchResult
        results = [
            SearchResult(content="a", source_path="f.py", repo="r", language="python",
                         start_line=1, end_line=20, score=0.9),
            SearchResult(content="b", source_path="f.py", repo="r", language="python",
                         start_line=10, end_line=30, score=0.7),  # overlaps with first
            SearchResult(content="c", source_path="g.py", repo="r", language="python",
                         start_line=1, end_line=20, score=0.6),  # different file — keep
        ]
        deduped = VectorStore.deduplicate(results)
        assert len(deduped) == 2
        assert deduped[0].content == "a"
        assert deduped[1].content == "c"


# ---------------------------------------------------------------------------
# Memory notes
# ---------------------------------------------------------------------------

class TestMemory:
    def test_remember_and_search(self, indexed_store):
        store, embedder = indexed_store
        mid = store.add_memory(
            "we use JWT not sessions because mobile client cannot handle cookies",
            tags=["auth", "decisions"],
            embedder=embedder,
        )
        assert mid
        vec = embedder.embed_query("JWT sessions cookies auth decision")
        results = store.search(vec, query_text="JWT sessions cookies", top_k=5)
        # Memory should appear in results
        assert any(r.language == "note" for r in results), "Memory note not found in search"

    def test_no_memories_flag(self, indexed_store):
        store, embedder = indexed_store
        store.add_memory("test note for exclusion", embedder=embedder)
        vec = embedder.embed_query("test note for exclusion")
        results = store.search(
            vec, query_text="test note for exclusion",
            top_k=5, include_memories=False,
        )
        assert all(r.language != "note" for r in results), \
            "Memory appeared despite include_memories=False"


# ---------------------------------------------------------------------------
# Model version tracking
# ---------------------------------------------------------------------------

class TestIncrementalIndexing:
    def test_unchanged_file_skips_embedding(self, tmp_path):
        """A file re-indexed with no changes should not re-embed any chunks."""
        db_path = tmp_path / "inc.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)

        src = tmp_path / "sample.py"
        src.write_text(
            "def one():\n    return 1\n\n"
            "def two():\n    return 2\n\n"
            "def three():\n    return 3\n"
        )
        first = indexer.index_path(src)
        assert first > 0

        # Count embed calls on second pass
        call_count = {"n": 0}
        original_embed = embedder.embed
        def counting_embed(texts):
            call_count["n"] += len(texts)
            return original_embed(texts)
        embedder.embed = counting_embed

        second = indexer.index_path(src)
        assert second == 0, "Unchanged file should produce 0 new chunks"
        assert call_count["n"] == 0, "No chunks should be embedded on unchanged file"

    def test_one_chunk_edit_only_re_embeds_that_chunk(self, tmp_path):
        """Editing one function should only re-embed that function, not the whole file."""
        db_path = tmp_path / "inc.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)

        src = tmp_path / "sample.py"
        src.write_text(
            "def alpha():\n    return 1\n\n"
            "def beta():\n    return 2\n\n"
            "def gamma():\n    return 3\n"
        )
        indexer.index_path(src)

        # Edit only beta
        src.write_text(
            "def alpha():\n    return 1\n\n"
            "def beta():\n    return 222\n\n"
            "def gamma():\n    return 3\n"
        )

        call_count = {"n": 0}
        original = embedder.embed
        def counting(texts):
            call_count["n"] += len(texts)
            return original(texts)
        embedder.embed = counting

        indexer.index_path(src)
        # Only one chunk changed — should embed exactly one
        assert call_count["n"] == 1, \
            f"Expected 1 embed call, got {call_count['n']}"

    def test_insert_at_top_reuses_existing_vectors(self, tmp_path):
        """Inserting a new function ABOVE existing ones must not re-embed the rest.

        Without content-anchored reuse, position-based diffing would mark every
        downstream chunk as "changed" and re-embed them. The indexer should match
        unchanged content by hash and reuse the stored vector.
        """
        db_path = tmp_path / "inc.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)

        src = tmp_path / "sample.py"
        src.write_text(
            "def alpha():\n    return 1\n\n"
            "def beta():\n    return 2\n\n"
            "def gamma():\n    return 3\n"
        )
        indexer.index_path(src)
        before = store.count()

        # Insert a new function at the top — every downstream chunk_index shifts
        src.write_text(
            "def zero():\n    return 0\n\n"
            "def alpha():\n    return 1\n\n"
            "def beta():\n    return 2\n\n"
            "def gamma():\n    return 3\n"
        )

        call_count = {"n": 0}
        original = embedder.embed
        def counting(texts):
            call_count["n"] += len(texts)
            return original(texts)
        embedder.embed = counting

        indexer.index_path(src)
        # Only `zero` is genuinely new content — the other three shift positions
        # but their hashes are unchanged, so vectors must come from the DB.
        assert call_count["n"] == 1, \
            f"Expected 1 embed call for insertion-at-top, got {call_count['n']}"
        assert store.count() == before + 1

    def test_removed_chunks_are_pruned(self, tmp_path):
        """Deleting a function from a file should remove its chunk from the DB."""
        db_path = tmp_path / "inc.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)

        src = tmp_path / "sample.py"
        src.write_text(
            "def alpha():\n    return 1\n\n"
            "def beta():\n    return 2\n\n"
            "def gamma():\n    return 3\n"
        )
        indexer.index_path(src)
        initial = store.count()

        # Drop gamma
        src.write_text(
            "def alpha():\n    return 1\n\n"
            "def beta():\n    return 2\n"
        )
        indexer.index_path(src)
        after = store.count()
        assert after < initial, f"Expected chunk count to drop after removal ({initial} -> {after})"


class TestExportImport:
    def test_roundtrip(self, indexed_store, tmp_path):
        """Export then re-import into a fresh DB should preserve chunk count."""
        store, embedder = indexed_store
        src_count = store.count()

        out = tmp_path / "export.jsonl"
        with open(out, "w") as fh:
            import json
            for c in store.iter_all_chunks():
                fh.write(json.dumps(c) + "\n")

        assert out.exists()
        line_count = sum(1 for _ in open(out))
        # exported chunks include memory chunks; store.count() excludes them
        assert line_count >= src_count

        # Fresh DB
        new_db = tmp_path / "restored.db"
        new_store = VectorStore(new_db)
        batch = []
        with open(out) as fh:
            for line in fh:
                import json as _j
                batch.append(_j.loads(line))
        new_store.upsert(batch)
        assert new_store.count() == src_count


class TestExplain:
    def test_explain_returns_ranks(self, indexed_store):
        store, embedder = indexed_store
        vec = embedder.embed_query("validate_token")
        rows = store.explain(vec, "validate_token", top_k=5)
        assert rows
        # Top result for an exact identifier should have a BM25 rank
        assert any(r["bm25_rank"] is not None for r in rows)
        # rrf_score should be sorted descending
        scores = [r["rrf_score"] for r in rows]
        assert scores == sorted(scores, reverse=True)


class TestPorterStemming:
    def test_stemming_matches_inflected_forms(self, indexed_store):
        """Porter stemmer should match 'expire' for 'expires' or 'expired'."""
        store, embedder = indexed_store
        # fixtures contain 'expired' (past tense) in auth.py
        vec = embedder.embed_query("expire")
        results = store.search(vec, query_text="expire", top_k=10, mode="bm25")
        # Should find auth.py because 'expire' stems match 'expired'
        paths = [r.source_path for r in results]
        assert any("auth.py" in p for p in paths), \
            f"Porter stemming didn't match 'expire' -> 'expired' in auth.py: {paths}"


class TestQueryCache:
    def test_repeated_queries_hit_cache(self, monkeypatch):
        """Second call to embed_query with same input should return cached result."""
        from ragdoll.embedder import Embedder

        emb = Embedder()

        # Monkeypatch _load_model so we don't download a real ONNX model
        call_count = {"n": 0}

        class _Arr(list):
            def tolist(self):
                return list(self)

        class FakeModel:
            def query_embed(self, q):
                call_count["n"] += 1
                yield _Arr(float(i) + hash(q) % 100 for i in range(768))

        monkeypatch.setattr(emb, "_load_model", lambda: FakeModel())

        v1 = emb.embed_query("repeat me")
        v2 = emb.embed_query("repeat me")
        v3 = emb.embed_query("something else")

        assert v1 == v2  # cached
        assert v1 != v3  # different query
        # Model invoked only twice — repeat query hit cache
        assert call_count["n"] == 2, f"Expected 2 calls, got {call_count['n']}"


class TestModelTracking:
    def test_records_model_on_first_use(self, tmp_path):
        db_path = tmp_path / "meta.db"
        store = VectorStore(db_path)
        store.check_embed_model("test/model-v1", 768)
        assert store.get_meta("embed_model") == "test/model-v1"
        assert store.get_meta("embed_dim") == "768"

    def test_warns_on_model_mismatch(self, tmp_path, caplog):
        import logging
        db_path = tmp_path / "meta.db"
        store = VectorStore(db_path)
        store.check_embed_model("test/model-v1", 768)
        with caplog.at_level(logging.WARNING):
            store.check_embed_model("test/model-v2", 768)
        assert "mismatch" in caplog.text.lower()


# ---------------------------------------------------------------------------
# HTTP API — exercise the FastAPI routes so regressions on query_text,
# /v1/embeddings, and /status are caught by the test suite.
# ---------------------------------------------------------------------------

class TestHTTPApi:
    def _client(self, indexed_store):
        from fastapi.testclient import TestClient
        from ragdoll import api

        store, embedder = indexed_store
        api.init(indexer=Indexer(store, embedder), embedder=embedder, store=store)
        return TestClient(api.app), store, embedder

    def test_search_hybrid_passes_query_text(self, indexed_store):
        """Regression: /search must forward `query` as `query_text` so the
        BM25 half of the hybrid fusion actually runs. If only the vector
        path ran, a pure identifier query like 'PaymentProcessor' could
        still match loosely, so we assert the exact-identifier file wins."""
        client, _, _ = self._client(indexed_store)
        resp = client.post(
            "/search",
            json={"query": "PaymentProcessor", "top_k": 5, "mode": "hybrid"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["results"], "Hybrid /search returned nothing"
        paths = [r["source_path"] for r in payload["results"]]
        assert any("payments.py" in p for p in paths), \
            f"payments.py not in hybrid /search results — BM25 half likely silent: {paths}"

    def test_search_bm25_mode(self, indexed_store):
        client, _, _ = self._client(indexed_store)
        resp = client.post(
            "/search",
            json={"query": "validate_token", "top_k": 5, "mode": "bm25"},
        )
        assert resp.status_code == 200
        paths = [r["source_path"] for r in resp.json()["results"]]
        assert any("auth.py" in p for p in paths)

    def test_status_reports_chunks(self, indexed_store):
        client, store, _ = self._client(indexed_store)
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["chunks"] == store.count()
        assert body["repos"]

    def test_openai_embeddings_single_and_batch(self, indexed_store):
        client, _, embedder = self._client(indexed_store)

        single = client.post("/v1/embeddings", json={"input": "hello world"})
        assert single.status_code == 200
        body = single.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert len(body["data"][0]["embedding"]) == 768

        batch = client.post(
            "/v1/embeddings",
            json={"input": ["one", "two", "three"], "model": "ragdoll"},
        )
        assert batch.status_code == 200
        data = batch.json()["data"]
        assert [d["index"] for d in data] == [0, 1, 2]

    def test_uninitialised_returns_503(self):
        from fastapi.testclient import TestClient
        from ragdoll import api

        # Force uninitialised state so _require_ready raises
        api._indexer = api._embedder = api._store = None
        client = TestClient(api.app, raise_server_exceptions=False)
        resp = client.post("/search", json={"query": "x"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Git hooks — block markers, registry, RAGDOLL_DB capture
# ---------------------------------------------------------------------------

class TestHooks:
    def _make_repo(self, tmp_path):
        import subprocess
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        return repo

    def test_install_writes_bracketed_block_and_registers(self, tmp_path, monkeypatch):
        from ragdoll.cli import hooks_install, hooks_uninstall, _hook_registry_path
        # Re-home the registry so we don't touch the real ~/.ragdoll
        monkeypatch.setenv("HOME", str(tmp_path))
        # _hook_registry_path() reads Path.home() at call time — patch it too
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))

        repo = self._make_repo(tmp_path)
        hooks_install(repo)

        post_checkout = repo / ".git" / "hooks" / "post-checkout"
        assert post_checkout.exists()
        body = post_checkout.read_text()
        assert "# >>> ragdoll hook >>>" in body
        assert "# <<< ragdoll hook <<<" in body

        registry = _hook_registry_path()
        assert registry.exists()
        assert str(repo) in registry.read_text()

        # Uninstall removes both the block and the registry entry
        hooks_uninstall(repo)
        assert not post_checkout.exists()
        assert not registry.exists() or str(repo) not in registry.read_text()

    def test_install_preserves_existing_user_hook(self, tmp_path, monkeypatch):
        from ragdoll.cli import hooks_install, hooks_uninstall
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
        repo = self._make_repo(tmp_path)
        hook = repo / ".git" / "hooks" / "post-checkout"
        hook.write_text("#!/bin/sh\necho 'user hook ran'\n")
        hook.chmod(0o755)

        hooks_install(repo)
        body = hook.read_text()
        assert "echo 'user hook ran'" in body
        assert "# >>> ragdoll hook >>>" in body

        hooks_uninstall(repo)
        # User content survives uninstall
        assert hook.exists()
        assert "echo 'user hook ran'" in hook.read_text()
        assert "# >>> ragdoll hook >>>" not in hook.read_text()

    def test_install_captures_ragdoll_db_env(self, tmp_path, monkeypatch):
        from ragdoll.cli import hooks_install
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
        monkeypatch.setenv("RAGDOLL_DB", "/tmp/work-profile.db")
        repo = self._make_repo(tmp_path)
        hooks_install(repo)
        body = (repo / ".git" / "hooks" / "post-checkout").read_text()
        assert '--db "/tmp/work-profile.db"' in body, \
            "RAGDOLL_DB env was not captured into the hook script"


# ---------------------------------------------------------------------------
# Import — model/dim mismatch refusal
# ---------------------------------------------------------------------------

class TestChunkSizeCap:
    def test_giant_class_is_split(self, tmp_path):
        """Regression: a real-world Python class can hit thousands of lines.
        The embedder OOM-kills the process if any single chunk exceeds the
        model's token window. Every chunker must enforce MAX_CHUNK_CHARS."""
        from ragdoll.chunker import chunk_file, MAX_CHUNK_CHARS

        # Synthesize a class with one huge method body — ~50 KB, well over
        # the 8 KB cap. Without _split_oversized this produced one chunk.
        big = tmp_path / "big.py"
        body = "\n".join(f"    x_{i} = {i}  # filler line {i}" for i in range(2000))
        big.write_text(f"class Huge:\n    def method(self):\n{body}\n")

        chunks = chunk_file(big)
        assert chunks, "Chunker returned nothing for a real file"
        max_chars = max(len(c.content) for c in chunks)
        assert max_chars <= MAX_CHUNK_CHARS, \
            f"Chunk exceeded cap: {max_chars} > {MAX_CHUNK_CHARS}"


class TestImportSafety:
    def test_dim_mismatch_refused(self, tmp_path):
        import json
        import typer
        from ragdoll.cli import import_cmd

        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({
            "id": "x", "content": "hello", "content_hash": "abc",
            "source_path": "f.py", "repo": "r", "language": "python",
            "chunk_index": 0, "start_line": 1, "end_line": 1,
            "vector": [0.0] * 384,  # wrong dim — current build expects 768
        }) + "\n")

        with pytest.raises(typer.Exit) as exc:
            import_cmd(src=bad, db=tmp_path / "fresh.db", replace=False)
        assert exc.value.exit_code == 2


# ---------------------------------------------------------------------------
# Cross-file content-hash dedup — embed identical chunks once per run
# ---------------------------------------------------------------------------

class TestCrossFileDedup:
    def _boiler(self) -> str:
        # A chunk that will be byte-identical across every file.
        return "# Copyright ACME\n# Licensed MIT\n" + "\n".join(f"k{i}=1" for i in range(30))

    def test_identical_chunks_across_files_embed_once(self, tmp_path):
        """Boilerplate repeated across files should be embedded once, reused elsewhere."""
        db_path = tmp_path / "dedup.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)

        boiler = self._boiler()
        for i in range(3):
            (tmp_path / f"f{i}.py").write_text(boiler + f"\n\ndef uniq_{i}():\n    return {i}\n")

        call_count = {"n": 0}
        original = embedder.embed
        def counting(texts):
            call_count["n"] += len(texts)
            return original(texts)
        embedder.embed = counting

        written = indexer.index_path(tmp_path)
        # 3 files x (1 shared boiler chunk + 1 unique chunk) = 6 rows written
        assert written == 6
        # Shared chunk embedded once; 2 duplicate copies reused, not re-embedded.
        assert indexer._last_embedded == 4
        assert indexer._last_reused == 2
        assert call_count["n"] == 4, f"model saw {call_count['n']} texts, expected 4"

    def test_reindex_after_dedup_embeds_nothing(self, tmp_path):
        """A clean re-index of a deduped tree must not re-embed anything."""
        db_path = tmp_path / "dedup.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)
        boiler = self._boiler()
        for i in range(3):
            (tmp_path / f"f{i}.py").write_text(boiler + f"\n\ndef uniq_{i}():\n    return {i}\n")
        indexer.index_path(tmp_path)

        call_count = {"n": 0}
        original = embedder.embed
        def counting(texts):
            call_count["n"] += len(texts)
            return original(texts)
        embedder.embed = counting

        written = indexer.index_path(tmp_path)
        assert written == 0
        assert call_count["n"] == 0
        assert indexer._last_unchanged == 6


# ---------------------------------------------------------------------------
# Per-run stats — embedded vs reused vs unchanged reporting
# ---------------------------------------------------------------------------

class TestIndexStats:
    def test_stats_reset_and_report_on_fresh_index(self, tmp_path):
        db_path = tmp_path / "stats.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)
        src = tmp_path / "s.py"
        src.write_text("def a():\n    return 1\n\ndef b():\n    return 2\n")

        n = indexer.index_path(src)
        assert n == indexer._last_embedded + indexer._last_reused
        assert indexer._last_embedded > 0
        assert indexer._last_unchanged == 0

        # Second pass: all unchanged, stats reset, nothing embedded/reused.
        indexer.index_path(src)
        assert indexer._last_embedded == 0
        assert indexer._last_reused == 0
        assert indexer._last_unchanged > 0


# ---------------------------------------------------------------------------
# Throttle knob
# ---------------------------------------------------------------------------

class TestThrottle:
    def test_env_parsing(self, monkeypatch):
        from ragdoll.indexer import _embed_throttle
        monkeypatch.delenv("RAGDOLL_THROTTLE_MS", raising=False)
        assert _embed_throttle() == 0.0
        monkeypatch.setenv("RAGDOLL_THROTTLE_MS", "250")
        assert _embed_throttle() == 0.25
        monkeypatch.setenv("RAGDOLL_THROTTLE_MS", "garbage")
        assert _embed_throttle() == 0.0

    def test_throttle_sleeps_between_batches(self, tmp_path, monkeypatch):
        db_path = tmp_path / "t.db"
        store = VectorStore(db_path)
        embedder = _FakeEmbedder()
        indexer = Indexer(store, embedder)
        src = tmp_path / "s.py"
        src.write_text("def a():\n    return 1\n")

        slept = {"total": 0.0}
        import ragdoll.indexer as ind
        monkeypatch.setattr(ind.time, "sleep", lambda s: slept.__setitem__("total", slept["total"] + s))
        monkeypatch.setenv("RAGDOLL_THROTTLE_MS", "100")

        indexer.index_path(src)
        assert slept["total"] >= 0.1, "throttle should sleep after embedding"


# ---------------------------------------------------------------------------
# Quantized model wiring
# ---------------------------------------------------------------------------

class TestQuantizedModel:
    def test_dim_and_name(self):
        from ragdoll.embedder import QUANT_MODEL, MODEL_DIMS
        assert MODEL_DIMS[QUANT_MODEL] == 768

    def test_active_model_and_db_selection(self, monkeypatch):
        import ragdoll.cli as cli
        from ragdoll.embedder import QUANT_MODEL, FAST_MODEL, DEFAULT_MODEL
        monkeypatch.delenv("RAGDOLL_DB", raising=False)

        monkeypatch.setattr(cli, "_QUANT_MODE", True, raising=False)
        monkeypatch.setattr(cli, "_FAST_MODE", False, raising=False)
        assert cli._active_model_name() == QUANT_MODEL
        assert cli._default_db(quant=True).name == "ragdoll-quant.db"

        monkeypatch.setattr(cli, "_QUANT_MODE", False, raising=False)
        assert cli._active_model_name() == DEFAULT_MODEL
        assert cli._default_db().name == "ragdoll.db"


# ---------------------------------------------------------------------------
# Doctor — unindexed-files detection (partial *repo* index)
# ---------------------------------------------------------------------------

class TestDoctorUnindexed:
    def test_flags_files_not_in_index(self, tmp_path):
        # Build a repo with two chunkable files but index only one of them.
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("def a():\n    return 1\n")
        (repo / "b.py").write_text("def b():\n    return 2\n")

        db_path = tmp_path / "doc.db"
        store = VectorStore(db_path)
        indexer = Indexer(store, _FakeEmbedder())
        indexer.index_path(repo / "a.py")  # only a.py indexed; b.py missing

        from ragdoll.cli import _scan_unindexed
        by_repo, capped = _scan_unindexed(store)
        assert not capped
        assert by_repo.get("repo") == 1, f"expected b.py flagged, got {by_repo}"

    def test_complete_index_flags_nothing(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("def a():\n    return 1\n")
        (repo / "b.py").write_text("def b():\n    return 2\n")

        db_path = tmp_path / "doc.db"
        store = VectorStore(db_path)
        indexer = Indexer(store, _FakeEmbedder())
        indexer.index_path(repo)  # index everything

        from ragdoll.cli import _scan_unindexed
        by_repo, capped = _scan_unindexed(store)
        assert by_repo == {}, f"complete index should flag nothing, got {by_repo}"


# ---------------------------------------------------------------------------
# Sticky model selection — DB remembers its model; flag choice is persisted
# ---------------------------------------------------------------------------

class TestStickyModel:
    def test_config_roundtrip_and_fallback(self, tmp_path, monkeypatch):
        import ragdoll.cli as cli
        monkeypatch.setattr(cli, "_CONFIG_PATH", tmp_path / "config.json")
        assert cli._read_default_mode() == "standard"  # no file yet
        cli._write_default_mode("fast")
        assert cli._read_default_mode() == "fast"
        cli._write_default_mode("bogus")  # invalid ignored
        assert cli._read_default_mode() == "fast"

    def _stamped_db(self, tmp_path):
        """Create a DB indexed with the fake embedder so its model is recorded.

        Resets the process-global `_model_checked` so the stamp actually happens
        even when an earlier test already flipped it.
        """
        import ragdoll.indexer as ind
        ind._model_checked = False
        db_path = tmp_path / "rec.db"
        store = VectorStore(db_path)
        src = tmp_path / "s.py"
        src.write_text("def a():\n    return 1\n")
        Indexer(store, _FakeEmbedder()).index_path(src)
        return db_path

    def test_db_records_model(self, tmp_path):
        import ragdoll.cli as cli
        db_path = self._stamped_db(tmp_path)
        assert cli._db_recorded_model(db_path) == "test/fake-embedder"
        assert cli._db_recorded_model(tmp_path / "nope.db") is None

    def test_resolve_adopts_recorded_model(self, tmp_path, monkeypatch):
        import ragdoll.cli as cli
        from ragdoll.embedder import DEFAULT_MODEL
        db_path = self._stamped_db(tmp_path)
        # No explicit flag → existing index's model wins over the default.
        monkeypatch.setattr(cli, "_MODE", "standard", raising=False)
        monkeypatch.setattr(cli, "_MODE_EXPLICIT", False, raising=False)
        assert cli._resolve_index_model(db_path) == "test/fake-embedder"
        # Fresh DB with no record → requested (default) model.
        assert cli._resolve_index_model(tmp_path / "fresh.db") == DEFAULT_MODEL

    def test_resolve_errors_on_conflicting_flag(self, tmp_path, monkeypatch):
        import ragdoll.cli as cli
        import typer
        db_path = self._stamped_db(tmp_path)
        # Explicit --fast against a DB built with a different model → hard error.
        monkeypatch.setattr(cli, "_MODE", "fast", raising=False)
        monkeypatch.setattr(cli, "_MODE_EXPLICIT", True, raising=False)
        with pytest.raises(typer.Exit) as exc:
            cli._resolve_index_model(db_path)
        assert exc.value.exit_code == 2

    def test_callback_persists_and_reads_default(self, tmp_path, monkeypatch):
        import ragdoll.cli as cli
        monkeypatch.setattr(cli, "_CONFIG_PATH", tmp_path / "config.json")
        monkeypatch.delenv("RAGDOLL_DB", raising=False)
        # Explicit flag persists as the new default.
        cli._root(fast=True, quantized=False, standard=False)
        assert cli._MODE == "fast" and cli._read_default_mode() == "fast"
        # Bare invocation now inherits the remembered default.
        cli._root(fast=False, quantized=False, standard=False)
        assert cli._MODE == "fast" and cli._MODE_EXPLICIT is False
        # --standard overrides back.
        cli._root(fast=False, quantized=False, standard=True)
        assert cli._MODE == "standard" and cli._read_default_mode() == "standard"


class TestIndexStatusHeartbeat:
    """The heartbeat file that powers `ragdoll status` live progress."""

    def test_write_and_clear_roundtrip(self, tmp_path, monkeypatch):
        import json
        import os
        import ragdoll.indexer as ind
        sp = tmp_path / "index.status"
        monkeypatch.setattr(ind, "_STATUS_PATH", sp)

        ind._write_status("myrepo", 3, 10)
        data = json.loads(sp.read_text())
        assert data["repo"] == "myrepo"
        assert data["done"] == 3 and data["total"] == 10
        assert data["pid"] == os.getpid()

        ind._clear_status()
        assert not sp.exists()

    def test_clear_missing_is_safe(self, tmp_path, monkeypatch):
        import ragdoll.indexer as ind
        monkeypatch.setattr(ind, "_STATUS_PATH", tmp_path / "nope.status")
        ind._clear_status()  # must not raise

    def test_status_dead_pid_is_cleaned(self, tmp_path, monkeypatch):
        import json
        import time
        import ragdoll.indexer as ind
        from ragdoll.cli import _print_running_index
        sp = tmp_path / "index.status"
        monkeypatch.setattr(ind, "_STATUS_PATH", sp)
        # PID 999999 is not a real process -> stale marker gets removed.
        sp.write_text(json.dumps(
            {"pid": 999999, "repo": "x", "done": 1, "total": 2, "updated_at": time.time()}
        ))
        _print_running_index()
        assert not sp.exists()

    def test_status_live_pid_is_kept(self, tmp_path, monkeypatch):
        import json
        import os
        import time
        import ragdoll.indexer as ind
        from ragdoll.cli import _print_running_index
        sp = tmp_path / "index.status"
        monkeypatch.setattr(ind, "_STATUS_PATH", sp)
        sp.write_text(json.dumps(
            {"pid": os.getpid(), "repo": "x", "done": 1, "total": 2, "updated_at": time.time()}
        ))
        _print_running_index()
        assert sp.exists()
