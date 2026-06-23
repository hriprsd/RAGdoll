#!/usr/bin/env bash
# =============================================================================
# RAGdoll — install script
# Local RAG memory for your dev tools. No cloud. No daemon required.
#
# Works with bash 3.2 (stock macOS) and bash 4+. Use --yes for non-interactive.
# =============================================================================
set -euo pipefail

# --- portable helpers --------------------------------------------------------
# macOS still ships bash 3.2 at /bin/bash — avoid ${var,,} and similar bash-4-isms.
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# Non-interactive opt-in: --yes (or $RAGDOLL_YES=1) answers "y" to every prompt.
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help)
            echo "Usage: install.sh [--yes]"
            echo "  --yes    Answer 'y' to every prompt (safe default branches)"
            exit 0
            ;;
    esac
done
[[ "${RAGDOLL_YES:-0}" == "1" ]] && ASSUME_YES=1

# If stdin isn't a TTY (curl | bash), fall back to --yes behaviour so reads
# don't silently return empty and the installer doesn't appear to hang.
[[ ! -t 0 ]] && ASSUME_YES=1

ask() {
    # $1 = prompt, $2 = var name to assign. Defaults to "n" when non-interactive.
    local __prompt="$1"
    local __var="$2"
    if [[ $ASSUME_YES -eq 1 ]]; then
        printf -v "$__var" '%s' "y"
        echo "$__prompt y  ${DIM:-}[auto: --yes]${RESET:-}"
    else
        read -rp "$__prompt " "$__var" || printf -v "$__var" '%s' "n"
    fi
}

path_contains() {
    # Exact-entry match on :-separated PATH — avoids substring false matches.
    local needle="$1"
    local p
    IFS=':' read -ra _parts <<< "${PATH:-}"
    for p in "${_parts[@]}"; do
        [[ "$p" == "$needle" ]] && return 0
    done
    return 1
}

# --- colours -----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# --- logging -----------------------------------------------------------------
log()     { echo -e "${BLUE}[ragdoll]${RESET} $*"; }
success() { echo -e "${GREEN}[ragdoll]${RESET} ${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}[ragdoll]${RESET} ${YELLOW}⚠${RESET}  $*"; }
error()   { echo -e "${RED}[ragdoll]${RESET} ${RED}✗${RESET} $*" >&2; }
section() { echo -e "\n${BOLD}━━━ $* ${RESET}"; }
die()     { error "$*"; exit 1; }

# --- config ------------------------------------------------------------------
RAGDOLL_DIR="${RAGDOLL_DIR:-$HOME/.ragdoll}"
VENV_DIR="$RAGDOLL_DIR/venv"
DB_PATH="$RAGDOLL_DIR/ragdoll.db"
LOG_DIR="$RAGDOLL_DIR/logs"
MIN_PYTHON_MINOR=11   # Python 3.11+

# Pin the embedding-model cache to a stable location. FastEmbed otherwise
# defaults to the OS temp dir (e.g. /var/folders/... on macOS), which the OS
# purges automatically — leaving a half-downloaded model.onnx and a confusing
# "NO_SUCHFILE ... model.onnx" crash on the next run. Respect a user override.
MODEL_CACHE_DIR="${FASTEMBED_CACHE_PATH:-$HOME/.cache/fastembed}"
# Export for this install session so the prefetch below lands here; the wrapper
# script and shell rc (further down) make it stick for every future invocation.
export FASTEMBED_CACHE_PATH="$MODEL_CACHE_DIR"

# Detect repo root (script lives in scripts/, repo root is one level up)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =============================================================================
section "RAGdoll installer"
# =============================================================================
echo -e "${DIM}Install dir : $RAGDOLL_DIR${RESET}"
echo -e "${DIM}Source      : $REPO_ROOT${RESET}"
echo ""

# =============================================================================
section "Checking system requirements"
# =============================================================================

# --- Python ------------------------------------------------------------------
log "Looking for Python 3.11+..."
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        # Use arithmetic comparison (( )) — avoids string comparison pitfalls
        major=$("$candidate" -c "import sys; print(sys.version_info.major)" 2>/dev/null) || continue
        ver=$("$candidate"   -c "import sys; print(sys.version_info.minor)" 2>/dev/null) || continue
        if (( major == 3 && ver >= MIN_PYTHON_MINOR )); then
            PYTHON="$candidate"
            full_ver=$("$candidate" --version 2>&1)
            success "Found $full_ver at $(command -v "$candidate")"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    error "Python 3.$MIN_PYTHON_MINOR+ not found."
    echo ""
    echo "  Install it via:"
    echo "    brew install python@3.11    # macOS"
    echo "    sudo apt install python3.11  # Ubuntu/Debian"
    echo "    https://python.org/downloads"
    die "Aborting — Python requirement not met."
fi

# --- pip ---------------------------------------------------------------------
log "Checking pip..."
"$PYTHON" -m pip --version &>/dev/null || die "pip not found. Run: $PYTHON -m ensurepip"
success "pip OK"

# --- sqlite3 (should always be present) -------------------------------------
log "Checking sqlite3..."
"$PYTHON" -c "import sqlite3; print(sqlite3.sqlite_version)" &>/dev/null || die "sqlite3 not available in Python build"
SQLITE_VER=$("$PYTHON" -c "import sqlite3; print(sqlite3.sqlite_version)")
success "sqlite3 $SQLITE_VER"

# --- repo sanity -------------------------------------------------------------
# Prevent confusing failure if this script has been copied somewhere else and
# REPO_ROOT doesn't actually point at a RAGdoll checkout.
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]] || ! grep -q 'name = "ragdoll"' "$REPO_ROOT/pyproject.toml" 2>/dev/null; then
    die "No RAGdoll pyproject.toml at $REPO_ROOT. Run this script from the repo's scripts/ dir."
fi
success "Source checkout looks correct"

# =============================================================================
section "Creating RAGdoll directories"
# =============================================================================

for dir in "$RAGDOLL_DIR" "$LOG_DIR" "$MODEL_CACHE_DIR"; do
    if [[ -d "$dir" ]]; then
        log "Directory exists: $dir"
    else
        mkdir -p "$dir"
        success "Created $dir"
    fi
done

# =============================================================================
section "Setting up Python virtual environment"
# =============================================================================

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR"
    # Default to KEEP on --yes — recreating wipes working state unexpectedly.
    if [[ $ASSUME_YES -eq 1 ]]; then
        recreate="n"
        log "Non-interactive mode — keeping existing venv"
    else
        read -rp "  Recreate it? [y/N] " recreate || recreate="n"
    fi
    if [[ "$(lower "$recreate")" == "y" ]]; then
        rm -rf "$VENV_DIR"
        log "Removed old venv"
    else
        log "Keeping existing venv"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

log "Upgrading pip inside venv..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
success "pip upgraded"

# =============================================================================
section "Installing RAGdoll"
# =============================================================================

log "Installing RAGdoll and dependencies (FastEmbed — no PyTorch needed)..."
log "${DIM}Embedding model (ONNX, ~200MB) will be downloaded on first use, not now.${RESET}"
echo ""

# Run pip in the background and show a spinner with the latest pip output line.
# This gives useful feedback ("Collecting onnxruntime...", "Downloading...")
# without the wall-of-text a plain pip install produces.
PIP_LOG="$(mktemp -t ragdoll-pip.XXXXXX)"
"$VENV_PIP" install "$REPO_ROOT" > "$PIP_LOG" 2>&1 &
PIP_PID=$!

spinner() {
    local pid="$1"
    local log="$2"
    local frames='|/-\'
    local i=0
    local start=$SECONDS
    # Only render the live line if attached to a TTY — in CI just wait.
    if [[ ! -t 1 ]]; then
        wait "$pid"
        return $?
    fi
    # Hide cursor for the duration
    printf '\e[?25l'
    while kill -0 "$pid" 2>/dev/null; do
        local frame="${frames:$((i % 4)):1}"
        local elapsed=$(( SECONDS - start ))
        # Grab the last meaningful line from pip (skip blanks / progress-bar CRs)
        local last
        last=$(tail -n 5 "$log" 2>/dev/null \
               | tr '\r' '\n' \
               | awk 'NF' \
               | tail -n 1 \
               | cut -c1-80)
        printf '\r  %s  %3ds  %s\e[K' "$frame" "$elapsed" "${last:-working...}"
        i=$((i + 1))
        sleep 0.15
    done
    wait "$pid"
    local rc=$?
    # Clear the spinner line + restore cursor
    printf '\r\e[K\e[?25h'
    return $rc
}

if ! spinner "$PIP_PID" "$PIP_LOG"; then
    error "pip install failed — last 40 lines of output:"
    tail -n 40 "$PIP_LOG" >&2
    rm -f "$PIP_LOG"
    die "Installation aborted."
fi
rm -f "$PIP_LOG"

# Verify the install
"$VENV_DIR/bin/ragdoll" --help &>/dev/null || die "Installation failed — 'ragdoll' binary not working"
success "RAGdoll installed successfully"

# =============================================================================
section "Creating wrapper script"
# =============================================================================
# We install a thin wrapper at /usr/local/bin/ragdoll (or ~/bin/ragdoll)
# so users don't need to activate the venv manually.

WRAPPER_CONTENT="#!/usr/bin/env bash
# RAGdoll wrapper — activates venv and forwards all args
# Pin the embedding-model cache so it never lands in a purgeable OS temp dir
# (prevents the half-downloaded-model NO_SUCHFILE crash). User override wins.
export FASTEMBED_CACHE_PATH=\"\${FASTEMBED_CACHE_PATH:-$MODEL_CACHE_DIR}\"
exec \"$VENV_DIR/bin/ragdoll\" \"\$@\"
"

# Bin-dir preference order:
#   1. /opt/homebrew/bin on Apple Silicon Macs (Homebrew arm64 default)
#   2. /usr/local/bin    on Intel Macs / Linux with writable prefix
#   3. ~/.local/bin      (PEP 370 user-site, common on Linux)
#   4. ~/bin             (last-resort personal bin)
INSTALL_BIN=""
ARCH="$(uname -m 2>/dev/null || echo unknown)"
BIN_CANDIDATES=()
if [[ "$(uname -s 2>/dev/null || echo)" == "Darwin" && "$ARCH" == "arm64" ]]; then
    BIN_CANDIDATES+=("/opt/homebrew/bin")
fi
BIN_CANDIDATES+=("/usr/local/bin" "$HOME/.local/bin")

for candidate in "${BIN_CANDIDATES[@]}"; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
        INSTALL_BIN="$candidate/ragdoll"
        break
    fi
done

if [[ -z "$INSTALL_BIN" ]]; then
    mkdir -p "$HOME/bin"
    INSTALL_BIN="$HOME/bin/ragdoll"
fi

if ! echo "$WRAPPER_CONTENT" > "$INSTALL_BIN"; then
    die "Failed to write wrapper to $INSTALL_BIN — check permissions"
fi
if ! chmod +x "$INSTALL_BIN"; then
    die "Failed to make $INSTALL_BIN executable"
fi
# Smoke-test the wrapper before declaring success
"$INSTALL_BIN" --help &>/dev/null || die "Wrapper at $INSTALL_BIN is not executable"
success "Wrapper installed at $INSTALL_BIN"

# Record where we installed so uninstall.sh knows
echo "$INSTALL_BIN" > "$RAGDOLL_DIR/bin_path"

# =============================================================================
section "Checking PATH"
# =============================================================================

BIN_DIR="$(dirname "$INSTALL_BIN")"
if ! path_contains "$BIN_DIR"; then
    warn "$BIN_DIR is not in your PATH"
    echo ""
    SHELL_RC=""
    if [[ "${SHELL:-}" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    elif [[ "${SHELL:-}" == *"bash"* ]]; then
        SHELL_RC="$HOME/.bashrc"
    fi

    if [[ -n "$SHELL_RC" ]]; then
        echo "  Add this to $SHELL_RC:"
        echo -e "  ${BOLD}export PATH=\"$BIN_DIR:\$PATH\"${RESET}"
        ask "  Add it automatically now? [y/N]" addpath
        if [[ "$(lower "$addpath")" == "y" ]]; then
            # Bracket our edits with unique markers so uninstall removes exactly
            # what we added, no more. Using sed with a substring match is
            # too loose and has burned users before.
            {
                echo ""
                echo "# >>> ragdoll install (do not edit) >>>"
                echo "export PATH=\"$BIN_DIR:\$PATH\""
                echo "export FASTEMBED_CACHE_PATH=\"\${FASTEMBED_CACHE_PATH:-$MODEL_CACHE_DIR}\""
                echo "# <<< ragdoll install <<<"
            } >> "$SHELL_RC"
            success "Added to $SHELL_RC — restart your shell or run: source $SHELL_RC"
        fi
    fi
else
    success "$BIN_DIR is already in PATH"
fi

# =============================================================================
section "Verifying installation"
# =============================================================================

log "Running: ragdoll status"
if "$INSTALL_BIN" status &>/dev/null; then
    success "ragdoll status OK"
else
    rc=$?
    # Exit 0 means OK; an empty-DB message also exits 0 in our CLI.
    # Anything else is a real failure — surface it instead of burying it.
    if [[ $rc -ne 0 ]]; then
        warn "ragdoll status exited $rc — re-running with output:"
        "$INSTALL_BIN" status || true
    fi
fi

# =============================================================================
section "Optional: prefetch the embedding model (~200 MB)"
# =============================================================================
# The first search/index command otherwise blocks 2–5 min downloading from
# Hugging Face with no progress bar. Doing it here is a much nicer UX.

# Delete any half-finished downloads first. An interrupted fetch leaves a
# 0-byte "*.incomplete" blob and no model.onnx, which surfaces later as the
# dreaded "NO_SUCHFILE ... model.onnx ... File doesn't exist" crash. Wiping
# them forces a clean re-download instead of tripping over the corpse.
clean_partial_downloads() {
    [[ -d "$MODEL_CACHE_DIR" ]] || return 0
    find "$MODEL_CACHE_DIR" -name '*.incomplete' -type f -delete 2>/dev/null || true
}

# Download in the background and reuse the shared spinner so the user sees
# progress instead of a multi-minute hang — and so a failure shows a friendly
# message rather than a raw Python traceback.
prefetch_model() {
    PREFETCH_LOG="$(mktemp -t ragdoll-prefetch.XXXXXX)"
    "$VENV_PYTHON" -c "from ragdoll.embedder import Embedder; Embedder()._load_model()" \
        > "$PREFETCH_LOG" 2>&1 &
    spinner "$!" "$PREFETCH_LOG"
}

ask "Download the embedding model now? [y/N]" prefetch
if [[ "$(lower "$prefetch")" == "y" ]]; then
    log "Fetching nomic-embed-text-v1.5 into $MODEL_CACHE_DIR (this may take a few minutes)..."
    clean_partial_downloads
    if prefetch_model; then
        success "Model cached ($MODEL_CACHE_DIR)"
        rm -f "${PREFETCH_LOG:-}"
    else
        warn "Download didn't complete — clearing the partial file and retrying once..."
        clean_partial_downloads
        if prefetch_model; then
            success "Model cached on retry ($MODEL_CACHE_DIR)"
            rm -f "${PREFETCH_LOG:-}"
        else
            error "Model prefetch failed twice."
            echo ""
            echo "  This is almost always the network blocking Hugging Face's model CDN"
            echo "  (corporate proxy / VPN / firewall): the small config files download but"
            echo "  the large model.onnx gets truncated. Options, then re-run the installer"
            echo "  or just run 'ragdoll index':"
            echo ""
            echo -e "    ${BOLD}export HF_ENDPOINT=https://hf-mirror.com${RESET}    # use a mirror"
            echo -e "    ${BOLD}export HTTPS_PROXY=<your-corp-proxy>${RESET}       # if behind a proxy"
            echo ""
            echo "  Last lines of the attempt:"
            tail -n 8 "${PREFETCH_LOG:-/dev/null}" 2>/dev/null | sed 's/^/    /' || true
            rm -f "${PREFETCH_LOG:-}"
            warn "Skipping for now — RAGdoll will retry automatically on first search/index."
        fi
    fi
else
    log "Skipping prefetch — model will download on first use."
fi

# =============================================================================
section "Optional: install git hooks into current directory"
# =============================================================================

if command -v git &>/dev/null && git -C "$(pwd)" rev-parse --git-dir &>/dev/null; then
    CURRENT_REPO="$(git -C "$(pwd)" rev-parse --show-toplevel)"
    ask "Install git hooks into $CURRENT_REPO? (auto-index on checkout/merge) [y/N]" dohooks
    if [[ "$(lower "$dohooks")" == "y" ]]; then
        "$INSTALL_BIN" hooks install "$CURRENT_REPO"
    fi
else
    log "Not inside a git repo — skipping git hooks (run 'ragdoll hooks install <repo>' later)"
fi

# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}━━━ RAGdoll is ready ━━━${RESET}"
echo ""
echo -e "  ${BOLD}Index a project:${RESET}   ragdoll index ~/your/project"
echo -e "  ${BOLD}Search:${RESET}            ragdoll search \"how do I handle auth\""
echo -e "  ${BOLD}Git hooks:${RESET}         ragdoll hooks install <repo>"
echo -e "  ${BOLD}MCP (optional):${RESET}    ragdoll mcp  (add to Claude Code / Cursor settings)"
echo -e "  ${BOLD}Uninstall:${RESET}         $SCRIPT_DIR/uninstall.sh"
echo ""
echo -e "${DIM}DB: $DB_PATH${RESET}"
echo -e "${DIM}Venv: $VENV_DIR${RESET}"
echo ""
