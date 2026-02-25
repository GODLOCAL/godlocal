#!/usr/bin/env bash
# setup_mobile_o.sh — Download Mobile-O CoreML weights for GodLocal
# arXiv:2602.20161 · FastVLM-0.5B + SANA-600M-512 + MCP
# Usage: ./setup_mobile_o.sh [iphone-sim|iphone-device]
# Mirrors setup_nexa.sh pattern

set -euo pipefail

TARGET="${1:-iphone-device}"
DEST="$HOME/Documents/MobileO"

echo "╔══════════════════════════════════════╗"
echo "║   Mobile-O CoreML Setup (v7.0.6)    ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Target : $TARGET"
echo "Dest   : $DEST"
echo ""

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
  echo "[1/4] Installing huggingface_hub CLI..."
  pip install -q huggingface_hub
else
  echo "[1/4] huggingface-cli ✓"
fi

# ── 2. Create destination ────────────────────────────────────────────────────
mkdir -p "$DEST"
echo "[2/4] Destination ready: $DEST"

# ── 3. Download CoreML packages from HuggingFace ────────────────────────────
echo "[3/4] Downloading Mobile-O CoreML packages..."
echo "      Repo: amshaker/Mobile-O-CoreML (community conversion)"
echo "      Size: ~1.6GB total · 4 packages"
echo ""

HF_REPO="amshaker/Mobile-O-CoreML"

packages=(
  "MobileO_VLM.mlpackage"
  "MobileO_DiT.mlpackage"
  "MobileO_VAE.mlpackage"
  "MobileO_MCP.mlpackage"
)

for pkg in "${packages[@]}"; do
  echo "  → $pkg"
  huggingface-cli download "$HF_REPO" "$pkg" \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False \
    2>/dev/null || {
      echo "  ⚠  $pkg not yet on HF — will fall back to App Store path"
      echo "     See step 3b below."
    }
done

# ── 3b. App Store fallback ───────────────────────────────────────────────────
# If HF packages not yet available (model is new), use App Store + export:
#   1. Install from App Store: https://apps.apple.com/app/id6759238106
#   2. Run: python3 scripts/export_mobile_o_coreml.py (ships in this repo)
#      This script pulls PyTorch weights → torch.export → coremltools → .mlpackage

if [ ! -f "$DEST/MobileO_VLM.mlpackage/Manifest.json" ]; then
  echo ""
  echo "[3b] HF packages unavailable — running PyTorch → CoreML export..."
  if [ -f "$(dirname "$0")/export_mobile_o_coreml.py" ]; then
    python3 "$(dirname "$0")/export_mobile_o_coreml.py" --output "$DEST"
  else
    echo "     Export script not found. Options:"
    echo "     a) Wait for community HF upload (amshaker/Mobile-O-CoreML)"
    echo "     b) Convert manually: see mobile/README.md § Mobile-O Install"
    echo "     c) iOS App Store ID 6759238106 — weights bundled in-app"
  fi
fi

# ── 4. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying packages..."
ALL_OK=true
for pkg in "${packages[@]}"; do
  if [ -f "$DEST/$pkg/Manifest.json" ]; then
    echo "  ✅  $pkg"
  else
    echo "  ❌  $pkg — missing"
    ALL_OK=false
  fi
done

echo ""
if $ALL_OK; then
  echo "✅  Mobile-O weights ready at: $DEST"
  echo ""
  echo "Next steps (Xcode):"
  echo "  1. Open GodLocal.xcodeproj"
  echo "  2. MobileOBridge auto-loads from Documents/MobileO/ on first call"
  echo "  3. No Embed & Sign needed (runtime load, not bundled)"
  echo ""
  echo "Test in app:"
  echo "  let mo = MobileOBridge()"
  echo "  let img = try await mo.generate(prompt: \"neon city at night\")"
  echo "  // → 512×512 UIImage in ~3s on iPhone 17 Pro"
else
  echo "⚠  Some packages missing. See options in step 3b above."
  echo "   Runtime stub mode active — app compiles without weights."
fi
