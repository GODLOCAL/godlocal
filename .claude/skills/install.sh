#!/usr/bin/env bash
# .claude/skills/install.sh
# Install selected Agent Skills from VoltAgent/awesome-agent-skills
# Run from repo root: bash .claude/skills/install.sh
set -e
BASE="https://raw.githubusercontent.com"
SKILLS_DIR=".claude/skills"
mkdir -p "$SKILLS_DIR"

install_skill() {
  local org=$1 slug=$2
  local dest="$SKILLS_DIR/$slug"
  if [ -d "$dest" ]; then
    echo "  [skip] $org/$slug already installed"
    return
  fi
  echo "  [install] $org/$slug"
  mkdir -p "$dest"
  # Try to fetch CLAUDE.md (most skills use this name)
  for fname in CLAUDE.md README.md skill.md; do
    url="$BASE/$org/$slug/main/$fname"
    if curl -sf "$url" -o "$dest/$fname" 2>/dev/null; then
      echo "    -> $fname"
      break
    fi
  done
}

echo "=== GodLocal Agent Skills Installer ==="
echo "Source: github.com/VoltAgent/awesome-agent-skills"
echo

# Security
install_skill trailofbits building-secure-contracts
install_skill trailofbits insecure-defaults
install_skill trailofbits static-analysis
install_skill trailofbits modern-python
install_skill openai       security-best-practices
install_skill openai       security-threat-model

# Cloud / Edge
install_skill cloudflare   agents-sdk
install_skill cloudflare   durable-objects

# ML / Models
install_skill huggingface  hugging-face-model-trainer
install_skill huggingface  hugging-face-evaluation

# Mobile / Swift
install_skill AvdLee       swiftui-expert-skill
install_skill efremidze    swift-patterns-skill

# DevOps
install_skill hashicorp    terraform-code-generation
install_skill microsoft    fastapi-router-py

# Vercel / Next.js
install_skill vercel-labs  next-best-practices
install_skill vercel-labs  vercel-deploy-claimable

# Code Quality
install_skill getsentry    code-review
install_skill getsentry    find-bugs
install_skill openai       yeet
install_skill openai       gh-fix-ci

echo
echo "=== Done. Skills installed to $SKILLS_DIR/ ==="
echo "Usage in Claude Code: @<slug>  e.g. @security-threat-model"
