#!/usr/bin/env bash
# scripts/godlocal-term.sh
# GodLocal multi-track terminal launcher
# Opens 4 parallel worktrees for GodLocal development tracks
#
# Usage: ./scripts/godlocal-term.sh [godlocal|x100|roblox|xzero|all|<branch>]

set -euo pipefail

TRACK="${1:-all}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
HOOK="$REPO_ROOT/.claude/hooks/worktree-godlocal.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠 GodLocal Terminal Launcher — $TRACK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A TRACK_MAP=(
  ["godlocal"]="feat/godlocal-dev"
  ["x100"]="feat/x100-oasis"
  ["roblox"]="feat/roblox-x100"
  ["xzero"]="feat/xzero-trading"
)

if [[ "$TRACK" == "all" ]]; then
  echo "🌿 Spawning all 4 tracks..."
  for t in godlocal x100 roblox xzero; do
    branch="${TRACK_MAP[$t]}"
    echo "  ▶ $t → $branch"
    bash "$HOOK" "$branch" "main" &
  done
  wait
  echo ""
  echo "✅ All tracks ready"
elif [[ -v TRACK_MAP["$TRACK"] ]]; then
  bash "$HOOK" "${TRACK_MAP[$TRACK]}" "main"
else
  bash "$HOOK" "$TRACK" "main"
fi
