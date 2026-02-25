#!/usr/bin/env bash
# .claude/hooks/post-tool-use.sh
# SparkNet capture: save tool outcome as Spark after Claude Code tool use
# Maps to: SparkNetConnector.post_action_capture()

set -euo pipefail

TOOL_NAME="${CLAUDE_TOOL_NAME:-unknown}"
AGENT="${CLAUDE_AGENT_ID:-autogenesis}"
EXIT_CODE="${CLAUDE_TOOL_EXIT_CODE:-0}"
OUTPUT="${CLAUDE_TOOL_OUTPUT:-}"

[[ "$EXIT_CODE" != "0" ]] && exit 0
case "$TOOL_NAME" in
  write_file|edit_file|bash|str_replace_editor) ;;
  *) exit 0 ;;
esac

SPARK=$(echo "$OUTPUT" | python3 -c "
import sys,re
t=re.sub(r'\s+',' ',sys.stdin.read(500)).strip()
print(t[:200])
" 2>/dev/null || echo "$TOOL_NAME completed")

[[ -z "$SPARK" ]] && exit 0

GODLOCAL_BASE="${GODLOCAL_BASE_URL:-http://localhost:8000}"
curl -sf -X POST "$GODLOCAL_BASE/spark/evoke" \
  -H "Content-Type: application/json" \
  -d "{\"agent\":\"$AGENT\",\"content\":\"[$TOOL_NAME] $SPARK\",\"tags\":[\"autogenesis\",\"$TOOL_NAME\"]}" \
  &>/dev/null || true
