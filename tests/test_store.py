"""Tests for the vector store module."""

import pytest
from pathlib import Path
from ragdoll.store import VectorStore, _build_fts_query, EMBED_DIM


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore backed by a temp DB."""
    return VectorStore(tmp_path / "test.db")


class TestFTSQuery:
    def test_basic_query(self):
        result = _build_fts_query("hello world")
        assert '"hello"' in result
        assert '"world"' in result
        assert "OR" in result

    def test_preserves_dots(self):
        result = _build_fts_query("auth.py error")
        assert '"auth.py"' in result

    def test_preserves_colons(self):
        result = _build_fts_query("TypeError: cannot read")
        assert '"TypeError:"' in result

    def test_single_char_tokens_stripped(self):
        result = _build_fts_query("a is b")
        assert '"a"' not in result
        assert '"b"' not in result
        assert '"is"' in result

    def test_empty_query(self):
        assert _build_fts_query("") == ""
        assert _build_fts_query("a b c") == ""  # all single chars


class TestVectorStore:
    def test_upsert_and_count(self, store):
        store.upsert([{
            "id": "test-1",
            "content": "hello world",
            "content_hash": "abc123",
            "source_path": "/test/file.py",
            "repo": "/test",
            "language": "python",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 5,
            "vector": [0.1] * EMBED_DIM,
        }])
        assert store.count() == 1

    def test_delete_by_source(self, store):
        store.upsert([{
            "id": "test-1",
            "content": "hello world",
            "content_hash": "abc123",
            "source_path": "/test/file.py",
            "repo": "/test",
            "language": "python",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 5,
            "vector": [0.1] * EMBED_DIM,
        }])
        store.delete_by_source("/test/file.py")
        assert store.count() == 0

    def test_known_hashes(self, store):
        store.upsert([{
            "id": "test-1",
            "content": "hello world",
            "content_hash": "abc123",
            "source_path": "/test/file.py",
            "repo": "/test",
            "language": "python",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 5,
            "vector": [0.1] * EMBED_DIM,
        }])
        hashes = store.known_hashes("/test/file.py")
        assert "abc123" in hashes

    def test_all_hashes(self, store):
        store.upsert([{
            "id": "test-1",
            "content": "hello",
            "content_hash": "hash1",
            "source_path": "/a.py",
            "repo": "/test",
            "language": "python",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 1,
            "vector": [0.1] * EMBED_DIM,
        }, {
            "id": "test-2",
            "content": "world",
            "content_hash": "hash2",
            "source_path": "/b.py",
            "repo": "/test",
            "language": "python",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 1,
            "vector": [0.2] * EMBED_DIM,
        }])
        all_h = store.all_hashes()
        assert "hash1" in all_h["/a.py"]
        assert "hash2" in all_h["/b.py"]
