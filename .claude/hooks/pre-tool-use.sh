#!/usr/bin/env bash
# .claude/hooks/pre-tool-use.sh
# SparkNet evoke: inject relevant agent memories before Claude Code tool use
# Maps to: SparkNetConnector.pre_action_evoke()

set -euo pipefail

TOOL_NAME="${CLAUDE_TOOL_NAME:-unknown}"
AGENT="${CLAUDE_AGENT_ID:-autogenesis}"
CONTEXT="${CLAUDE_TOOL_INPUT:-}"

# Only evoke for write/execute tools
case "$TOOL_NAME" in
  write_file|edit_file|bash|str_replace_editor|computer) ;;
  *) exit 0 ;;
esac

GODLOCAL_BASE="${GODLOCAL_BASE_URL:-http://localhost:8000}"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1][:200]))" "$CONTEXT" 2>/dev/null || echo "")
SPARKS=$(curl -sf "$GODLOCAL_BASE/spark/evoke?agent=$AGENT&context=$ENCODED" 2>/dev/null || echo "[]")
COUNT=$(echo "$SPARKS" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

if [[ "$COUNT" -gt "0" ]]; then
  echo ""
  echo "⚡ SparkNet: $COUNT memories for [$TOOL_NAME]"
  echo "$SPARKS" | python3 -c "
import json,sys
for s in json.load(sys.stdin)[:3]:
    print(f'  • {s[\"content\"][:120]}')
" 2>/dev/null || true
fi
