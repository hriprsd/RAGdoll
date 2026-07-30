"""
Chunker - splits files into semantically meaningful chunks.
- Python: AST-aware, splits on top-level and class-level definitions
- Markdown/RST/text: heading-based paragraph splits
- Everything else: sliding line-window fallback

Per-project exclusions:
  Drop a .ragdollignore file anywhere in your project (same syntax as .gitignore).
  RAGdoll walks up from each file to find the nearest .ragdollignore and applies it.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CHUNK_SIZE = 40      # lines per chunk for fallback splitter
CHUNK_OVERLAP = 8    # overlapping lines between chunks

# Hard ceiling on a single chunk's size, in characters. Any chunk produced by
# any splitter gets post-split if it exceeds this.
#
# Why: the embedding model (nomic-embed-text-v1.5) has an 8192-token window.
# One token ~= 4 chars on English/code, so ~8000 chars ~ 2000 tokens keeps us
# well inside the window AND keeps onnxruntime memory bounded - a single
# 28 KB chunk will OOM-kill the process during batched inference.
MAX_CHUNK_CHARS = 8000

CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
    ".sql": "sql",
    ".lua": "lua",
    ".r": "r",
    ".scala": "scala",
    ".zig": "zig",
    ".ex": "elixir",
    ".exs": "elixir",
    ".php": "php",
}

DOC_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
}

CONFIG_EXTENSIONS: dict[str, str] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".ini": "ini",
    ".cfg": "config",
    ".conf": "config",
}

# Max file size to index (1 MB). Larger files are almost always generated/vendored.
MAX_FILE_SIZE = 1_048_576

# Directory names that should never be indexed
SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".tox", ".mypy_cache",
})

# Glob patterns matched against the filename only
_SKIP_FILE_PATTERNS: tuple[str, ...] = (
    # Secrets / credentials
    ".env", ".env.*",
    "*.pem", "*.key", "*.crt", "*.cer", "*.p12", "*.pfx",
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "*.keystore", "*.jks",
    "credentials.json", "service-account*.json",
    ".aws", ".npmrc", ".pypirc",
    # Generated / vendored
    "*.lock",
    "*-lock.json", "*-lock.yaml", "*-lock.yml",     # package-lock.json, pnpm-lock.yaml
    "Pipfile.lock", "poetry.lock", "uv.lock", "Cargo.lock",
    "*.min.js", "*.min.css",
    "*.pyc", "*.pyo",
    "*.map",       # source maps
    "*.bundle.js",
    # Binary files that sneak past extension checks
    "*.wasm", "*.so", "*.dylib", "*.dll",
    "*.exe", "*.bin",
    # Data files
    "*.sqlite", "*.sqlite3", "*.db",
    "*.parquet", "*.arrow", "*.feather",
    "*.pkl", "*.pickle",
    # Images / media
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg",
    "*.mp3", "*.mp4", "*.wav", "*.webm",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
    # Archives
    "*.zip", "*.tar", "*.gz", "*.bz2", "*.xz", "*.7z",
)


@dataclass
class RawChunk:
    content: str
    start_line: int   # 1-indexed
    end_line: int     # 1-indexed, inclusive
    language: str


@lru_cache(maxsize=256)
def _load_ragdollignore(directory: Path) -> tuple[str, ...]:
    """
    Load and cache .ragdollignore patterns for a directory.
    Returns a tuple of glob patterns (empty if no file found).
    Walks up to the filesystem root looking for the nearest .ragdollignore.
    """
    current = directory
    while True:
        ignore_file = current / ".ragdollignore"
        if ignore_file.exists():
            try:
                lines = ignore_file.read_text(encoding="utf-8").splitlines()
                patterns = tuple(
                    line.strip()
                    for line in lines
                    if line.strip() and not line.startswith("#")
                )
                return patterns
            except OSError:
                return ()
        parent = current.parent
        if parent == current:   # reached filesystem root
            break
        current = parent
    return ()


def _matches_ragdollignore(path: Path) -> bool:
    """Check path against any .ragdollignore found in its directory tree."""
    patterns = _load_ragdollignore(path.parent)
    if not patterns:
        return False
    name = path.name
    # Match against filename and relative path segments
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        # Also support directory-level patterns like "tests/fixtures/"
        for part in path.parts:
            if fnmatch.fnmatch(part, pattern.rstrip("/")):
                return True
    return False


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    name = path.name
    for pattern in _SKIP_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    # Skip hidden files except useful dotfiles we want to index
    if name.startswith(".") and path.suffix not in {".md", ".mdx", ".toml", ".rst"}:
        return True
    if _matches_ragdollignore(path):
        return True
    return False


def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    return (
        CODE_EXTENSIONS.get(ext)
        or DOC_EXTENSIONS.get(ext)
        or CONFIG_EXTENSIONS.get(ext)
        or "unknown"
    )


def _is_binary(path: Path, sample_size: int = 8192) -> bool:
    """Quick binary detection - read first N bytes and look for null bytes."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except (OSError, PermissionError):
        return True  # if we can't read it, skip it


def _split_oversized(chunks: list[RawChunk]) -> list[RawChunk]:
    """Break any chunk above MAX_CHUNK_CHARS into smaller line-window pieces.

    Preserves the original start_line so search results still point at the
    right spot in the source file. Applied as a final pass over every
    splitter's output so the AST/regex paths never emit monster chunks
    (e.g. a 500-line class body) that OOM the embedder.
    """
    out: list[RawChunk] = []
    for ch in chunks:
        if len(ch.content) <= MAX_CHUNK_CHARS:
            out.append(ch)
            continue
        lines = ch.content.splitlines()
        i = 0
        while i < len(lines):
            end = min(i + CHUNK_SIZE, len(lines))
            piece = "\n".join(lines[i:end])
            # If even a CHUNK_SIZE window is too big (very long lines), hard-cap
            # on characters as a last resort - slicing mid-line is ugly but
            # guaranteed to fit in the model's window.
            if len(piece) > MAX_CHUNK_CHARS:
                piece = piece[:MAX_CHUNK_CHARS]
            if piece.strip():
                out.append(RawChunk(
                    content=piece,
                    start_line=ch.start_line + i,
                    end_line=ch.start_line + end - 1,
                    language=ch.language,
                ))
            i += CHUNK_SIZE - CHUNK_OVERLAP
    return out


def chunk_file(path: Path) -> list[RawChunk]:
    """Split a file into chunks. Returns [] if the file should be skipped."""
    if should_skip(path):
        return []
    # Skip large files (likely generated/vendored)
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return []
    except OSError:
        return []
    # Skip binary files
    if _is_binary(path):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, PermissionError):
        return []
    if not text.strip():
        return []

    language = detect_language(path)
    if language == "python":
        chunks = _chunk_python(text, language)
    elif language in ("markdown", "mdx", "rst", "text"):
        chunks = _chunk_markdown(text, language)
    elif language in ("typescript", "javascript"):
        chunks = _chunk_ts_js(text, language)
    elif language == "go":
        chunks = _chunk_go(text, language)
    elif language == "rust":
        chunks = _chunk_rust(text, language)
    else:
        chunks = _chunk_by_lines(text, language)

    return _split_oversized(chunks)


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def _chunk_by_lines(text: str, language: str) -> list[RawChunk]:
    lines = text.splitlines()
    chunks: list[RawChunk] = []
    i = 0
    while i < len(lines):
        end = min(i + CHUNK_SIZE, len(lines))
        content = "\n".join(lines[i:end])
        if content.strip():
            chunks.append(RawChunk(
                content=content,
                start_line=i + 1,
                end_line=end,       # end is already the count of lines taken
                language=language,
            ))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _chunk_python(text: str, language: str) -> list[RawChunk]:
    """
    Split by top-level definitions (functions, classes) using the AST.
    For each ClassDef also emit its methods as sub-chunks so large classes
    don't become one giant chunk.
    Falls back to line-window chunking on SyntaxError.
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _chunk_by_lines(text, language)

    lines = text.splitlines()
    chunks: list[RawChunk] = []

    def emit(node: ast.AST) -> None:
        start = node.lineno                              # 1-indexed
        end = getattr(node, "end_lineno", None) or start
        content = "\n".join(lines[start - 1 : end])     # slice is 0-indexed
        if content.strip():
            chunks.append(RawChunk(
                content=content,
                start_line=start,
                end_line=end,
                language=language,
            ))

    # Emit module preamble (imports, constants, type aliases, __all__)
    # - everything before the first function/class definition.
    first_def_line = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def_line = node.lineno
            break

    if first_def_line and first_def_line > 1:
        preamble = "\n".join(lines[: first_def_line - 1]).strip()
        if preamble and len(preamble.splitlines()) > 2:  # skip trivial preambles
            chunks.append(RawChunk(
                content=preamble,
                start_line=1,
                end_line=first_def_line - 1,
                language=language,
            ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node)
        elif isinstance(node, ast.ClassDef):
            emit(node)
            # Also emit each method individually so large classes stay searchable
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    emit(item)

    # If AST gave us nothing (e.g. file is only top-level statements), fall back
    return chunks or _chunk_by_lines(text, language)


def _chunk_markdown(text: str, language: str) -> list[RawChunk]:
    """Split on headings (lines starting with #). Falls back to line chunks."""
    lines = text.splitlines()
    chunks: list[RawChunk] = []
    current: list[str] = []
    current_start = 1

    for i, line in enumerate(lines, start=1):
        if line.startswith("#") and current:
            content = "\n".join(current).strip()
            if content:
                chunks.append(RawChunk(
                    content=content,
                    start_line=current_start,
                    end_line=i - 1,
                    language=language,
                ))
            current = [line]
            current_start = i
        else:
            current.append(line)

    if current:
        content = "\n".join(current).strip()
        if content:
            chunks.append(RawChunk(
                content=content,
                start_line=current_start,
                end_line=len(lines),
                language=language,
            ))

    return chunks or _chunk_by_lines(text, language)


# ---------------------------------------------------------------------------
# Regex-based chunkers for TS/JS and Go
# Not as precise as AST, but far better than blind line-window splitting.
# ---------------------------------------------------------------------------

# TS/JS: match top-level declarations (functions, classes, interfaces, types, enums)
_TS_JS_PATTERN = re.compile(
    r"^(?:export\s+)?(?:default\s+)?"
    r"(?:"
    r"(?:async\s+)?function\s+\w+"          # function foo / async function foo
    r"|class\s+\w+"                          # class Foo
    r"|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\(|function)"  # const foo = ( / function
    r"|interface\s+\w+"                      # interface Foo
    r"|type\s+\w+\s*="                       # type Foo =
    r"|enum\s+\w+"                           # enum Foo
    r")",
    re.MULTILINE,
)

# Go: match top-level func/type/var/const declarations
_GO_PATTERN = re.compile(
    r"^(?:"
    r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+"  # func Foo / func (r *T) Foo
    r"|type\s+\w+\s+(?:struct|interface)"    # type Foo struct/interface
    r"|var\s+(?:\w+|[(])"                    # var x / var (
    r"|const\s+(?:\w+|[(])"                  # const x / const (
    r")",
    re.MULTILINE,
)

# Rust: match top-level declarations
_RUST_PATTERN = re.compile(
    r"^(?:pub(?:\s*\(crate\))?\s+)?(?:async\s+)?"
    r"(?:fn|struct|enum|trait|impl|mod|type|const|static)\s",
    re.MULTILINE,
)


def _chunk_by_regex(text: str, language: str, pattern: re.Pattern) -> list[RawChunk]:
    """Split source into chunks at regex-matched declaration boundaries."""
    lines = text.splitlines()
    matches = list(pattern.finditer(text))

    if not matches:
        return _chunk_by_lines(text, language)

    # Convert char offsets to line numbers
    line_starts: list[int] = []
    for m in matches:
        line_num = text[:m.start()].count("\n") + 1
        line_starts.append(line_num)

    chunks: list[RawChunk] = []

    for i, start_line in enumerate(line_starts):
        end_line = (line_starts[i + 1] - 1) if i + 1 < len(line_starts) else len(lines)
        content = "\n".join(lines[start_line - 1 : end_line])
        if content.strip():
            chunks.append(RawChunk(
                content=content,
                start_line=start_line,
                end_line=end_line,
                language=language,
            ))

    # If file has content before the first match, include it
    if line_starts and line_starts[0] > 1:
        preamble = "\n".join(lines[: line_starts[0] - 1]).strip()
        if preamble:
            chunks.insert(0, RawChunk(
                content=preamble,
                start_line=1,
                end_line=line_starts[0] - 1,
                language=language,
            ))

    return chunks or _chunk_by_lines(text, language)


def _chunk_ts_js(text: str, language: str) -> list[RawChunk]:
    return _chunk_by_regex(text, language, _TS_JS_PATTERN)


def _chunk_go(text: str, language: str) -> list[RawChunk]:
    return _chunk_by_regex(text, language, _GO_PATTERN)


def _chunk_rust(text: str, language: str) -> list[RawChunk]:
    return _chunk_by_regex(text, language, _RUST_PATTERN)
