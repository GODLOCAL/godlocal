#!/usr/bin/env bash
# GodLocal Skills Installer
# Installs all skills listed in SKILLS_INDEX.md from their original sources
# Usage: bash .claude/skills/install.sh

set -euo pipefail

SKILLS_DIR"${SKILLS_DIR:-.claude/skills}"
mkdir -p "$SKILLS_DIR"
echo "🚂 Installing GodLocal Skills into $SKILLS_DIR..."

# Antigravity awesome-skills (930+ skills via npx)
echo "✅ Antigravity skills (930+)..."
npx "antigravity-awesome-skills@0.0.1" --claude --prefix="$SKILLS_DIR" 2>/dev/null || \
  npx "antigravity-awesome-skills" --claude --prefix="$SKILLS_DIR" 2>/dev/null || \
  echo "  Skipped antigravity (npx not available)"

# VoltAgent/awesome-agent-skills (curated originals)
echo "✅ VoltAgent curated skills..."
CURATED_SKILLS=(
  "trailofbits/building-secure-contracts"
  "trailofbits/insecure-defaults"
  "trailofbits/static-analysis"
  "trailofbits/modern-python"
  "openai/security-best-practices"
  "openai/security-threat-model"
  "cloudflare/agents-sdk"
  "cloudflare/durable-objects"
  "cloudflare/building-mcp-server-on-cloudflare"
  "huggingface/hugging-face-model-trainer"
  "huggingface/hugging-face-evaluation"
  "huggingface/hugging-face-jobs"
  "AvdLee/swiftui-expert-skill"
  "efremidze/swift-patterns-skill"
  "conorluddy/ios-simulator-skill"
  "hashicorp/terraform-code-generation"
  "microsoft/pydantic-models-py"
  "microsoft/fastapi-router-py"
  "vercel-labs/react-best-practices"
  "vercel-labs/next-best-practices"
  "vercel-labs/vercel-deploy-claimable"
  "getsentry/code-review"
  "getsentry/find-bugs"
  "getsentry/create-pr"
  "openai/yeet"
  "openai/gh-fix-ci"
  "replicate/replicate"
)

for skill in "${CURATED_SKILLS[@]}"; do
  SKILL_NAME"${skill#/*}"
  SKILL_PATH"$SKILLS_DIR/$SKILL_NAME"
  if [[ -d "$SKILL_PATH" ]]; then
    echo "  Skip (exists): $SKILL_NAME"
  else
    URL="https://raw.githubusercontent.com/VoltAgent/awesome-agent-skills/main/skills/${skill}/SKILL.md"
    mkdir -p "$SKILL_PATH"
    if curl -sf "$URL" -o "$SKILL_PATH/SKILL.md" 2>/dev/null; then
      echo "  Installed: $SKILL_NAME"
    else
      rmdir "$SKILL_PATH" 2>/dev/null || true
      echo "  Failed: $SKILL_NAME"
    fi
  fi
done

# blader/humanizer (v2.2.0 — 24 AI writing patterns)
echo "  Installing humanizer (blader/humanizer)..."
HUMANIZER_PATH="$SKILLS_DIR/humanizer"
if [[ -d "$HUMANIZER_PATH" ]]; then
  echo "  Skip (exists): humanizer"
else
  mkdir -p "$HUMANIZER_PATH"
  if curl -sf "https://raw.githubusercontent.com/blader/humanizer/main/SKILL.md" \
       -o "$HUMANIZER_PATH/SKILL.md" 2>/dev/null; then
    echo "  Installed: humanizer"
  else
    rmdir "$HUMANIZER_PATH" 2>/dev/null || true
    echo "  Failed: humanizer"
  fi
fi


echo ""
echo "🚂 Skills installed to: $SKILLS_DIR"
echo "Usage: @humanizer, @ai-agents-architect, @security-audit, etc."
