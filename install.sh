#!/usr/bin/env bash
# GodLocal v5 — One-command installer
# Usage: bash install.sh
# Supports: macOS, Linux, SteamOS/Arch
set -e

VENV_DIR="${HOME}/godlocal-env"
SOUL_TEMPLATE="god_soul.example.md"
SOUL_FILE="god_soul.md"
DATA_DIR="godlocal_data"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GodLocal v5 — Sovereign AI Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Python check
if ! command -v python3 &>/dev/null; then
    echo "❌  python3 not found. Install Python 3.10+ first."
    exit 1
fi
PYTHON_VER=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✓  Python ${PYTHON_VER} detected"

# 2. Create virtualenv
if [ ! -d "$VENV_DIR" ]; then
    echo "→  Creating virtualenv at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# 3. Install dependencies
echo "→  Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓  Dependencies installed"

# 4. Ollama check
if ! command -v ollama &>/dev/null; then
    echo "→  Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✓  Ollama installed"
else
    echo "✓  Ollama already present: $(ollama --version 2>/dev/null || echo 'version unknown')"
fi

# 5. Pull default model (fallback chain)
echo "→  Pulling default model (z-lab/Qwen3-4B-PARO ~1.8GB)..."
ollama pull z-lab/Qwen3-4B-PARO 2>/dev/null \
    || ollama pull qwen3:4b 2>/dev/null \
    || ollama pull qwen2.5:3b 2>/dev/null \
    || echo "⚠  Could not pull model — run manually: ollama pull qwen2.5:3b"

# 6. Create data directories
mkdir -p "${DATA_DIR}/souls"
echo "✓  Data directories ready: ${DATA_DIR}/"

# 7. Soul template
if [ ! -f "${DATA_DIR}/souls/${SOUL_FILE}" ]; then
    cp "$SOUL_TEMPLATE" "${DATA_DIR}/souls/${SOUL_FILE}"
    echo "✓  Soul template copied → ${DATA_DIR}/souls/${SOUL_FILE}"
    echo "   ✏️  Edit ${DATA_DIR}/souls/${SOUL_FILE} to personalise your AI"
else
    echo "✓  Soul file already exists: ${DATA_DIR}/souls/${SOUL_FILE}"
fi

# 8. .env check
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✓  .env created from .env.example"
    echo "   ⚠  Set TELEGRAM_TOKEN in .env before starting"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  GodLocal is ready to launch!"
echo ""
echo "  Activate env: source ${VENV_DIR}/bin/activate"
echo ""
echo "  Terminal 1 (server):  python godlocal_v5.py"
echo "  Terminal 2 (telegram): python godlocal_telegram.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
