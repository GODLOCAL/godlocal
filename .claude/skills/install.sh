#!/usr/bin/env bash
# GodLocal Agent Skills Installer v2
# Source A: VoltAgent/awesome-agent-skills
# Source B: sickn33/antigravity-awesome-skills (930+ skills)
set -euo pipefail

SKILLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install_skill() {
  local owner=$1; local repo=$2
  local target="$SKILLS_DIR/$owner/$repo"
  if [ -d "$target" ]; then
    echo "⟳  $owner/$repo (update)"
    git -C "$target" pull --ff-only -q
  else
    echo "↓  $owner/$repo"
    mkdir -p "$SKILLS_DIR/$owner"
    git clone -q --depth 1 "https://github.com/$owner/$repo.git" "$target"
  fi
}

# ── Block A: VoltAgent curated skills ─────────────────────────────────────────

# Security
install_skill trailofbits building-secure-contracts
install_skill trailofbits insecure-defaults
install_skill trailofbits static-analysis
install_skill trailofbits modern-python
install_skill openai security-best-practices
install_skill openai security-threat-model

# Cloud / Edge
install_skill cloudflare agents-sdk
install_skill cloudflare durable-objects
install_skill cloudflare building-mcp-server-on-cloudflare

# ML / Models
install_skill huggingface hugging-face-model-trainer
install_skill huggingface hugging-face-evaluation
install_skill huggingface hugging-face-jobs

# Mobile / Swift
install_skill AvdLee swiftui-expert-skill
install_skill efremidze swift-patterns-skill
install_skill conorluddy ios-simulator-skill

# DevOps / Infra
install_skill hashicorp terraform-code-generation
install_skill microsoft pydantic-models-py
install_skill microsoft fastapi-router-py

# Vercel / Next.js
install_skill vercel-labs react-best-practices
install_skill vercel-labs next-best-practices
install_skill vercel-labs vercel-deploy-claimable

# Productivity / Code Quality
install_skill getsentry code-review
install_skill getsentry find-bugs
install_skill getsentry create-pr
install_skill openai yeet
install_skill openai gh-fix-ci

# AI Models / Generative
install_skill replicate replicate

# ── Block B: antigravity-awesome-skills (930+, Claude Code path) ─────────────
echo ""
echo "↓  antigravity-awesome-skills (930+ skills via npx)"
npx antigravity-awesome-skills --claude 2>/dev/null || {
  echo "  npx failed, falling back to git clone..."
  git clone -q --depth 1 https://github.com/sickn33/antigravity-awesome-skills.git "$SKILLS_DIR/antigravity" 2>/dev/null ||     git -C "$SKILLS_DIR/antigravity" pull --ff-only -q
}

echo ""
echo "✅ All skills installed in $SKILLS_DIR"
echo "   Block A (curated): 27 skills"
echo "   Block B (antigravity): 930+ skills"
echo ""
echo "Usage:"
echo "  Claude Code: /skill-name <your task>"
echo "  Agents:      @skill-name in prompt"
