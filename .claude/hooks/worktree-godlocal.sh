#!/usr/bin/env bash
# .claude/hooks/worktree-godlocal.sh
# Overrides Claude Code --worktree: sibling dirs + Ghostty + Lazygit + Yazi
# Adapted from @dani_avila7 claude-code-templates/worktree-ghostty
# GodLocal: each agent/track gets own branch in ../worktrees/<branch>

set -euo pipefail

BRANCH="${1:-}"
BASE="${2:-main}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
WORKTREES_DIR="$(dirname "$REPO_ROOT")/worktrees"

if [[ -z "$BRANCH" ]]; then
  echo "Usage: worktree-godlocal.sh <branch-name> [base-branch]" >&2
  exit 1
fi

WORKTREE_PATH="$WORKTREES_DIR/$BRANCH"
mkdir -p "$WORKTREES_DIR"

if [[ -d "$WORKTREE_PATH" ]]; then
  echo "⚡ Worktree exists: $WORKTREE_PATH"
else
  echo "🌿 Creating worktree: $BRANCH → $WORKTREE_PATH"
  git -C "$REPO_ROOT" worktree add -b "$BRANCH" "$WORKTREE_PATH" "$BASE" 2>/dev/null \
    || git -C "$REPO_ROOT" worktree add "$WORKTREE_PATH" "$BRANCH"
fi

# Copy .env to worktree if available
if [[ -f "$REPO_ROOT/.env" && ! -f "$WORKTREE_PATH/.env" ]]; then
  cp "$REPO_ROOT/.env" "$WORKTREE_PATH/.env"
  echo "🔑 .env copied"
fi

# Ghostty: 3-pane layout (shell + lazygit + yazi)
if command -v ghostty &>/dev/null; then
  ghostty --title="GodLocal: $BRANCH" \
    --working-directory="$WORKTREE_PATH" \
    --command="bash -c 'echo GodLocal — Branch: $BRANCH; bash'" &
  echo "🟢 Ghostty launched"
elif command -v kitty &>/dev/null; then
  kitty @ new-window --cwd "$WORKTREE_PATH" --title "GodLocal: $BRANCH" bash &
  echo "🟡 Kitty fallback"
elif command -v wezterm &>/dev/null; then
  wezterm start --cwd "$WORKTREE_PATH" -- bash &
  echo "🟡 WezTerm fallback"
else
  osascript -e "tell application \"Terminal\" to do script \"cd '$WORKTREE_PATH'\"" 2>/dev/null || true
  echo "🟡 Terminal.app fallback"
fi

cat <<EOF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 GodLocal Worktree Ready
   Branch : $BRANCH
   Path   : $WORKTREE_PATH
   Base   : $BASE

   Start GodLocal:
     python godlocal_v6.py --verbose
     python sleep_scheduler_v6.py
     cargo build --release -p xzero_swap_cli
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
