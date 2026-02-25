#!/usr/bin/env bash
# Install Claude Code CLI for local Ollama routing
# Usage: bash scripts/install_claude_code.sh

set -e

echo "=== Installing Claude Code CLI ==="

# Try npm first (preferred)
if command -v npm &>/dev/null; then
    echo "Installing via npm..."
    npm install -g @anthropic-ai/claude-code
    echo "✅ claude-code installed: $(claude --version)"
    exit 0
fi

# Fallback: official install script
if command -v curl &>/dev/null; then
    echo "Installing via curl..."
    curl -fsSL https://claude.ai/install.sh | bash
    echo "✅ claude-code installed"
    exit 0
fi

echo "❌ Neither npm nor curl found. Install Node.js first: https://nodejs.org"
exit 1
