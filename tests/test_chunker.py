"""Tests for the chunker module."""

from pathlib import Path
from ragdoll.chunker import (
    chunk_file,
    should_skip,
    detect_language,
    _chunk_python,
    _chunk_markdown,
    _chunk_by_lines,
    _is_binary,
)


class TestShouldSkip:
    def test_skips_node_modules(self, tmp_path):
        f = tmp_path / "node_modules" / "pkg" / "index.js"
        f.parent.mkdir(parents=True)
        f.write_text("const x = 1;")
        assert should_skip(f) is True

    def test_skips_git_dir(self, tmp_path):
        f = tmp_path / ".git" / "config"
        f.parent.mkdir(parents=True)
        f.write_text("[core]")
        assert should_skip(f) is True

    def test_skips_env_file(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SECRET=abc")
        assert should_skip(f) is True

    def test_skips_pem_file(self, tmp_path):
        f = tmp_path / "server.pem"
        f.write_text("-----BEGIN CERTIFICATE-----")
        assert should_skip(f) is True

    def test_skips_lock_file(self, tmp_path):
        f = tmp_path / "package-lock.json"
        f.write_text("{}")
        assert should_skip(f) is True

    def test_allows_python_file(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("print('hello')")
        assert should_skip(f) is False

    def test_allows_markdown(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Hello")
        assert should_skip(f) is False


class TestDetectLanguage:
    def test_python(self):
        assert detect_language(Path("foo.py")) == "python"

    def test_typescript(self):
        assert detect_language(Path("foo.ts")) == "typescript"

    def test_yaml(self):
        assert detect_language(Path("config.yaml")) == "yaml"

    def test_toml(self):
        assert detect_language(Path("pyproject.toml")) == "toml"

    def test_unknown(self):
        assert detect_language(Path("file.xyz")) == "unknown"


class TestChunkPython:
    def test_splits_functions(self):
        code = '''def foo():
    return 1

def bar():
    return 2
'''
        chunks = _chunk_python(code, "python")
        assert len(chunks) == 2
        assert "foo" in chunks[0].content
        assert "bar" in chunks[1].content

    def test_splits_class_methods(self):
        code = '''class MyClass:
    def method_a(self):
        pass

    def method_b(self):
        pass
'''
        chunks = _chunk_python(code, "python")
        # Should have class + 2 methods
        assert len(chunks) == 3

    def test_falls_back_on_syntax_error(self):
        code = "def broken(\n  this is not valid python"
        chunks = _chunk_python(code, "python")
        assert len(chunks) > 0  # falls back to line chunks


class TestChunkMarkdown:
    def test_splits_on_headings(self):
        md = '''# Introduction
Some intro text.

## Details
More details here.

## Conclusion
The end.
'''
        chunks = _chunk_markdown(md, "markdown")
        assert len(chunks) == 3
        assert "Introduction" in chunks[0].content
        assert "Details" in chunks[1].content


class TestBinaryDetection:
    def test_detects_binary(self, tmp_path):
        f = tmp_path / "binary.dat"
        f.write_bytes(b"\x00\x01\x02\x03")
        assert _is_binary(f) is True

    def test_detects_text(self, tmp_path):
        f = tmp_path / "text.py"
        f.write_text("print('hello')")
        assert _is_binary(f) is False
