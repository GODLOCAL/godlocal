#!/usr/bin/env bash
# setup_picobot.sh — первоначальная настройка Picobot VPS
# Запускать ОДИН раз с iPhone через Termius / Blink Shell:
#   ssh root@YOUR_PICOBOT_IP
#   curl -sSL https://raw.githubusercontent.com/GODLOCAL/godlocal/main/setup_picobot.sh | bash

set -euo pipefail

echo "╔═════════════════════════════════════╗"
echo "║   GodLocal Picobot Setup v7.0.8    ║"
echo "╚═════════════════════════════════════╝"

# ── 1. Docker ────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "[1/5] Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker && systemctl start docker
else
  echo "[1/5] Docker ✓"
fi

if ! docker compose version &>/dev/null; then
  echo "  Installing docker compose plugin..."
  apt-get install -y docker-compose-plugin 2>/dev/null || pip install docker-compose
fi

# ── 2. Clone repo ────────────────────────────────────────────────────────────
echo "[2/5] Cloning GODLOCAL/godlocal..."
if [ -d ~/godlocal ]; then
  cd ~/godlocal && git pull origin main
else
  git clone https://github.com/GODLOCAL/godlocal ~/godlocal
  cd ~/godlocal
fi

# ── 3. Env file ─────────────────────────────────────────────────────────────
echo "[3/5] Creating .env..."
if [ ! -f ~/godlocal/.env ]; then
  cat > ~/godlocal/.env <<EOF
# GodLocal environment — edit these!
XZERO_PRIVKEY=YOUR_BASE58_KEYPAIR
HELIUS_API_KEY=YOUR_HELIUS_KEY
MOONPAY_API_KEY=YOUR_MOONPAY_KEY
MOONPAY_SECRET_KEY=YOUR_MOONPAY_SECRET
EOF
  echo "  ⚠  Edit ~/godlocal/.env and add your keys!"
else
  echo "  .env exists — skipping"
fi

# ── 4. Build + start ─────────────────────────────────────────────────────────
echo "[4/5] Building and starting GodLocal..."
cd ~/godlocal
docker compose --env-file .env up -d --build

# ── 5. Pull LLM ─────────────────────────────────────────────────────────────
echo "[5/5] Pulling Qwen3:0.5b (CPU, fits $5 VPS)..."
sleep 5
docker compose exec -T ollama ollama pull qwen3:0.5b || echo "  Will pull on first request"

echo ""
echo "✅ GodLocal running on Picobot!"
VPS_IP=$(curl -s ifconfig.me)
echo "  Backend: http://$VPS_IP:8000"
echo "  Health:  http://$VPS_IP:8000/health"
echo ""
echo "iPhone setup:"
echo "  1. Open GodLocal app"
echo "  2. Settings → Backend URL → http://$VPS_IP:8000"
echo "  3. Done — no Mac needed 🚀"
