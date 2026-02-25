# GodLocal Mobile — iPhone On-Device AI

Run LFM2 24B (Liquid AI), PARO 4B, or Qwen3 entirely on your iPhone 17 Pro.  
No cloud. No API key. ~40–60 toks/s on A19 Pro ANE.

---

## Requirements

| | |
|---|---|
| Mac | macOS 15+ (Sequoia) |
| Xcode | 16+ — [download](https://developer.apple.com/xcode/) |
| iPhone | 17 Pro (A19 Pro / 12GB RAM) or any iPhone with A17+ |
| iOS | 17.0+ |
| Apple ID | Free account — enough for personal device |

---

## Setup (2 steps)

### Step 1 — Terminal (Mac)

```bash
git clone https://github.com/GODLOCAL/godlocal.git
cd godlocal/mobile
chmod +x setup_nexa.sh
./setup_nexa.sh
```

The script:
- Downloads `NexaSdk.xcframework` (~300MB) from [docs.nexa.ai](https://docs.nexa.ai/en/nexa-sdk-ios/quickstart)
- Activates the SDK in `LLMBridgeNexa.swift` (removes stubs, uncomments real code)

Takes ~2 minutes.

---

### Step 2 — Xcode

1. Open `godlocal/mobile/GodLocal.xcodeproj` in Xcode
2. **Drag** `NexaSdk.xcframework` (just downloaded) into the Project Navigator
3. In **Frameworks, Libraries, and Embedded Content** → set to **Embed & Sign**
4. Select your iPhone 17 Pro as target
5. **Signing & Capabilities** → add your Apple ID as Team
6. **Cmd + R**

First launch: tap a model → SDK downloads weights once (~5–25 min depending on model).

---

## Models

| Model | Size | Speed (A19 Pro) | Best for |
|-------|------|-----------------|---------|
| **PARO 4B** | 1.8 GB | ~60 tok/s | Daily use, fast |
| **LFM2 24B** | 4.8 GB | ~40 tok/s | Complex reasoning |
| Qwen3 4B | 2.4 GB | ~55 tok/s | Coding |
| Qwen3 8B | 4.9 GB | ~35 tok/s | Balanced |

LFM2 24B is a Mixture-of-Experts model — only 2B parameters active per token, so it fits in 12GB RAM while matching much larger dense models.

---

## Files

| File | Description |
|------|-------------|
| `LLMBridgeNexa.swift` | NexaSDK wrapper — model loading, ANE/GPU/CPU backend, streaming |
| `NexaView.swift` | SwiftUI chat UI — model picker, stream bubble, tok/s badge, 🎙 mic button |
| `OasisApp.swift` | App entry point with TabView |
| `setup_nexa.sh` | One-command setup script |
| `LLMBridgeNexa_activated.swift` | Pre-activated reference (no stubs) |
| `IPHONE17PRO_GUIDE.md` | Detailed 11-step guide with all links |
| `AudioBridgeMLX.swift` | MLX-Audio-Swift TTS+STT wrapper |

---

## Troubleshooting

**Build error: `Cannot find type 'Llm'`**  
→ xcframework not added to Xcode yet. Complete Step 2.

**Build error: `Module 'NexaSdk' not found`**  
→ Set Embed to **Embed & Sign**, not "Do Not Embed".

**App crashes on model load**  
→ Not enough RAM. Use PARO 4B (1.8GB) instead of LFM2 24B.

**Trust error on iPhone**  
→ Settings → General → VPN & Device Management → your Apple ID → Trust.

---

## Links

- NexaSDK docs: https://docs.nexa.ai/en/nexa-sdk-ios/overview
- xcframework direct: https://nexa-model-hub-bucket.s3.us-west-1.amazonaws.com/public/ios/latest/NexaSdk.xcframework.zip
- Nexa model hub: https://nexa.ai/models
- LFM2 24B (Liquid AI): https://liquid.ai/lfm2
- NexaAI GitHub: https://github.com/NexaAI/nexa-sdk

---

## Audio — On-Device TTS + STT (Optional)

`AudioBridgeMLX.swift` integrates [MLX-Audio-Swift](https://github.com/Prince_Canuma/MLX-Audio-Swift) for native voice on Apple Silicon.

**Mic button** in NexaView: tap 🎙 → live transcription (Qwen3-ASR-0.6B) → auto-fills input field.

| Capability | Models |
|-----------|--------|
| **TTS** (text → speech) | Qwen3-TTS, Marvis, Soprano, Pocket — with streaming |
| **STT** (speech → text) | **Qwen3-ASR-0.6B** *(default)*, LFM-2.5-Audio, Voxtral Realtime, Parakeet |

### STT model comparison

| Model | Size | iPhone 15 Pro | Notes |
|-------|------|--------------|-------|
| **Qwen3-ASR-0.6B (4bit)** | ~240 MB | realtime | Default. Fastest, lightest — WisprFlow alternative |
| LFM-2.5-Audio | ~1.2 GB | realtime | Liquid AI, high accuracy |
| Voxtral Realtime | ~0.8 GB | realtime | Streaming-optimized |
| Parakeet | ~0.6 GB | realtime | English-only |

> Source: [@Prince_Canuma](https://x.com/Prince_Canuma) — live transcription of Dwarkesh×Karpathy interview on M3 Max and iPhone 15 Pro.

**Install (Xcode only — no script needed):**

1. In Xcode: **File → Add Package Dependencies...**
2. Paste URL: `https://github.com/Prince_Canuma/MLX-Audio-Swift`
3. Select version `0.1.0` → **Add Package**
4. Done. `AudioBridgeMLX.swift` auto-activates (stub `#else` branch disabled).

> Without this package the file compiles fine in stub mode — LLM chat still works normally.

---

## Mobile-O — On-Device Unified Multimodal AI (Optional)

`MobileOBridge.swift` integrates [arXiv:2602.20161](https://arxiv.org/abs/2602.20161) — the first unified model for both **image understanding** (VQA, OCR, captioning) and **text-to-image generation** running entirely on-device.

| Capability | Model | Speed (iPhone 17 Pro) |
|-----------|-------|----------------------|
| **Understanding** (VQA/OCR/caption) | FastVLM-0.5B + MCP | TTFT 248ms |
| **Generation** (text → 512×512) | SANA-600M-512 DiT | ~3s (20 steps) |

**Stats**: 1.6B params total · <2GB RAM · 6–11× faster than Show-O / JanusFlow · CoreML + MLX

**Install weights** (first launch):
- iOS App Store: [id 6759238106](https://apps.apple.com/app/id6759238106)  
- Or HuggingFace: `Mobile-O-0.5B` → place `.mlpackage` files in `Documents/MobileO/`

**Usage in GodLocal** (combine with NexaView):
```swift
// Speak to PARO 4B → get text → generate image from response
let answer = try await nexaBridge.generate(prompt: userQuery)
let image  = try await mobileO.generate(prompt: answer)
```

> Source: [@HuggingPapers](https://x.com/HuggingPapers) · paper: "Unified Multimodal Understanding and Generation on Mobile Device"
