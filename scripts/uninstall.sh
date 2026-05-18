#!/usr/bin/env bash
# =============================================================================
# RAGdoll — uninstall script
# Removes everything install.sh put on your machine.
#
# Safe with bash 3.2 (stock macOS). Use --yes for non-interactive runs.
# =============================================================================
set -euo pipefail

# --- portable helpers --------------------------------------------------------
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

ASSUME_YES=0
KEEP_DB=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes)     ASSUME_YES=1 ;;
        --keep-db)    KEEP_DB=1    ;;
        -h|--help)
            echo "Usage: uninstall.sh [--yes] [--keep-db]"
            echo "  --yes       Answer 'y' to prompts (still keeps DB unless --force-db is set)"
            echo "  --keep-db   Never prompt to delete ~/.ragdoll/ragdoll.db"
            exit 0
            ;;
    esac
done
[[ "${RAGDOLL_YES:-0}" == "1" ]] && ASSUME_YES=1
[[ ! -t 0 ]] && ASSUME_YES=1

ask() {
    local __prompt="$1"
    local __var="$2"
    local __default="${3:-n}"
    if [[ $ASSUME_YES -eq 1 ]]; then
        printf -v "$__var" '%s' "$__default"
    else
        read -rp "$__prompt " "$__var" || printf -v "$__var" '%s' "$__default"
    fi
}

# --- colours -----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

log()     { echo -e "${BLUE}[ragdoll]${RESET} $*"; }
success() { echo -e "${GREEN}[ragdoll]${RESET} ${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}[ragdoll]${RESET} ${YELLOW}⚠${RESET}  $*"; }
error()   { echo -e "${RED}[ragdoll]${RESET} ${RED}✗${RESET} $*" >&2; }
section() { echo -e "\n${BOLD}━━━ $* ${RESET}"; }
skip()    { echo -e "${DIM}[ragdoll] skipped: $*${RESET}"; }

RAGDOLL_DIR="${RAGDOLL_DIR:-$HOME/.ragdoll}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
section "RAGdoll uninstaller"
# =============================================================================
echo -e "${DIM}This will remove the ragdoll binary, virtual environment, and optionally your index DB.${RESET}"
echo ""

# =============================================================================
section "Confirming"
# =============================================================================
ask "Are you sure you want to uninstall RAGdoll? [y/N]" confirm "y"
if [[ "$(lower "$confirm")" != "y" ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
section "Stopping launchd agent (macOS)"
# =============================================================================
# Do this BEFORE removing the binary so launchctl can cleanly stop the process.
# Without this the plist survives uninstall and keeps trying to relaunch a
# deleted binary — the single most common "ghost process" uninstall bug.
PLIST_PATH="$HOME/Library/LaunchAgents/com.ragdoll.daemon.plist"
if [[ "$(uname -s 2>/dev/null || echo)" == "Darwin" && -f "$PLIST_PATH" ]]; then
    if command -v launchctl &>/dev/null; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
    fi
    rm -f "$PLIST_PATH"
    success "Removed launchd agent: $PLIST_PATH"
else
    skip "no launchd agent to remove"
fi

# =============================================================================
section "Removing git hooks from known repos"
# =============================================================================
# `ragdoll hooks install` writes an entry to ~/.ragdoll/hook_registry so we
# can reliably clean up every repo that got hooks, not just the current one.
HOOK_MARKER="# RAGdoll"
HOOKS_REMOVED=0

_remove_hook_if_ours() {
    local hook_path="$1"
    [[ -f "$hook_path" ]] || return 0
    if grep -q "$HOOK_MARKER" "$hook_path" 2>/dev/null; then
        # If the file is ONLY our shebang + our ragdoll line, delete it.
        # Otherwise strip the RAGdoll-bracketed block and keep the user's content.
        if grep -q "# >>> ragdoll hook >>>" "$hook_path"; then
            # Remove the bracketed block only
            # Use a tempfile — sed -i differs between GNU and BSD.
            local tmp
            tmp="$(mktemp)"
            awk '
                /# >>> ragdoll hook >>>/ { skip=1; next }
                /# <<< ragdoll hook <<</ { skip=0; next }
                !skip { print }
            ' "$hook_path" > "$tmp"
            # If the result has anything besides a shebang + blanks, keep it.
            if grep -qE '^[^#[:space:]]' "$tmp" 2>/dev/null; then
                mv "$tmp" "$hook_path"
                chmod +x "$hook_path"
                success "Cleaned RAGdoll block from $hook_path (kept user hook)"
            else
                rm -f "$tmp" "$hook_path"
                success "Removed hook: $hook_path"
            fi
        else
            # Legacy hook written entirely by RAGdoll — safe to delete whole file.
            rm -f "$hook_path"
            success "Removed hook: $hook_path"
        fi
        HOOKS_REMOVED=$((HOOKS_REMOVED + 1))
    fi
}

HOOK_REGISTRY="$RAGDOLL_DIR/hook_registry"
if [[ -f "$HOOK_REGISTRY" ]]; then
    while IFS= read -r repo_path; do
        [[ -z "$repo_path" ]] && continue
        if [[ -d "$repo_path/.git/hooks" ]]; then
            _remove_hook_if_ours "$repo_path/.git/hooks/post-checkout"
            _remove_hook_if_ours "$repo_path/.git/hooks/post-merge"
        fi
    done < "$HOOK_REGISTRY"
    rm -f "$HOOK_REGISTRY"
fi

# Also check the current repo as a safety net (pre-registry installs).
if command -v git &>/dev/null && git -C "$(pwd)" rev-parse --git-dir &>/dev/null 2>&1; then
    GIT_DIR="$(git -C "$(pwd)" rev-parse --git-dir)"
    _remove_hook_if_ours "$GIT_DIR/hooks/post-checkout"
    _remove_hook_if_ours "$GIT_DIR/hooks/post-merge"
fi

if [[ $HOOKS_REMOVED -eq 0 ]]; then
    skip "no RAGdoll git hooks found"
fi

# =============================================================================
section "Removing wrapper binary"
# =============================================================================

BIN_PATH=""

# Check recorded path first
if [[ -f "$RAGDOLL_DIR/bin_path" ]]; then
    BIN_PATH="$(cat "$RAGDOLL_DIR/bin_path")"
fi

# Fallback: find it in PATH
if [[ -z "$BIN_PATH" ]] || [[ ! -f "$BIN_PATH" ]]; then
    BIN_PATH="$(command -v ragdoll 2>/dev/null || true)"
fi

if [[ -n "$BIN_PATH" ]] && [[ -f "$BIN_PATH" ]]; then
    # Only remove if it's our wrapper (points into the venv)
    if grep -q "$RAGDOLL_DIR" "$BIN_PATH" 2>/dev/null; then
        rm -f "$BIN_PATH"
        success "Removed binary: $BIN_PATH"
    else
        warn "Binary at $BIN_PATH doesn't look like RAGdoll's wrapper — skipping"
        warn "Remove it manually if needed: rm $BIN_PATH"
    fi
else
    skip "ragdoll binary not found (already removed?)"
fi

# =============================================================================
section "Removing virtual environment"
# =============================================================================

VENV_DIR="$RAGDOLL_DIR/venv"
if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
    success "Removed venv: $VENV_DIR"
else
    skip "venv not found at $VENV_DIR"
fi

# =============================================================================
section "Removing index database"
# =============================================================================

DB_PATH="$RAGDOLL_DIR/ragdoll.db"
if [[ -f "$DB_PATH" ]]; then
    DB_SIZE=$(du -sh "$DB_PATH" 2>/dev/null | cut -f1 || echo "unknown size")
    echo ""
    if [[ $KEEP_DB -eq 1 ]]; then
        log "Keeping database at $DB_PATH ($DB_SIZE) — --keep-db given"
        deldb="n"
    else
        # Default answer is "keep" — losing an indexed DB is the worst possible
        # uninstall accident. --yes also chooses "keep" unless the user passed
        # something stronger.
        ask "Delete index database ($DB_PATH, $DB_SIZE)? [y/N]" deldb "n"
    fi
    if [[ "$(lower "$deldb")" == "y" ]]; then
        rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
        success "Deleted database"
    else
        log "Keeping database at $DB_PATH"
        log "You can delete it manually later: rm $DB_PATH"
    fi
else
    skip "database not found at $DB_PATH"
fi

# --- log directory + stray bookkeeping ---------------------------------------
LOG_DIR="$RAGDOLL_DIR/logs"
if [[ -d "$LOG_DIR" ]]; then
    rm -rf "$LOG_DIR"
    success "Removed $LOG_DIR"
fi
# install.sh drops a bin_path marker — clean it up so the dir can be rmdir'd.
rm -f "$RAGDOLL_DIR/bin_path"
# Also remove the model-download marker if any was written
rm -f "$RAGDOLL_DIR/model_prefetched"

# =============================================================================
section "Removing RAGdoll directory"
# =============================================================================

if [[ -d "$RAGDOLL_DIR" ]]; then
    REMAINING=$(find "$RAGDOLL_DIR" -mindepth 1 | wc -l | tr -d ' ')
    if [[ "$REMAINING" -eq 0 ]]; then
        rmdir "$RAGDOLL_DIR"
        success "Removed $RAGDOLL_DIR (empty)"
    else
        warn "$RAGDOLL_DIR still has $REMAINING item(s) — leaving it in place"
        log "Contents:"
        find "$RAGDOLL_DIR" -mindepth 1 -maxdepth 2 | while IFS= read -r f; do
            echo -e "  ${DIM}$f${RESET}"
        done
        log "Remove manually when ready: rm -rf $RAGDOLL_DIR"
    fi
else
    skip "$RAGDOLL_DIR not found"
fi

# =============================================================================
section "Cleaning up PATH entry"
# =============================================================================

for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
    [[ -f "$rc" ]] || continue
    # New installs bracket their PATH block with unique markers so we can
    # remove EXACTLY what we added. Fall back to the legacy pattern only for
    # rc files written by older installers.
    if grep -q "# >>> ragdoll install" "$rc" 2>/dev/null; then
        tmp="$(mktemp)"
        awk '
            /# >>> ragdoll install/ { skip=1; next }
            /# <<< ragdoll install/ { skip=0; next }
            !skip { print }
        ' "$rc" > "$tmp" && mv "$tmp" "$rc"
        success "Removed RAGdoll block from $rc"
    elif grep -q "^# RAGdoll$" "$rc" 2>/dev/null; then
        # Legacy (pre-marker) layout: exact "# RAGdoll" header + the very next
        # line's export. Match the export literally to avoid clobbering user
        # lines that happen to mention ragdoll + PATH.
        tmp="$(mktemp)"
        awk '
            /^# RAGdoll$/ { skip=2; next }
            skip > 0    { skip--; next }
            { print }
        ' "$rc" > "$tmp" && mv "$tmp" "$rc"
        success "Cleaned legacy PATH entry from $rc"
    fi
done

# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}━━━ RAGdoll uninstalled ━━━${RESET}"
echo ""
echo -e "${DIM}If you kept the database and want to reinstall later, run install.sh again —"
echo -e "your index will still be there.${RESET}"
echo ""
