# NexaSDK Setup Guide for GodLocal Mobile
## v6.8 — On-Device LLM via NexaSDK (iOS 17+)

### What this enables
- **LFM2 24B-A2B** (Liquid AI MoE) — 2B active params, ~35 tok/s on A17 Pro
- **PARO 4B/8B** — GodLocal's primary quantized models
- **Qwen3 4B/8B** — general purpose
- All running **fully on-device** via Apple Neural Engine / Metal GPU

---

### 1. Download NexaSDK xcframework

```bash
curl -L https://nexa-model-hub-bucket.s3.us-west-1.amazonaws.com/public/ios/latest/NexaSdk.xcframework.zip -o NexaSdk.xcframework.zip
unzip NexaSdk.xcframework.zip
```

### 2. Add to Xcode

1. Open `mobile/` folder as Xcode project (or workspace)
2. Drag `NexaSdk.xcframework` into the project navigator
3. In target settings → **Frameworks, Libraries, and Embedded Content** → **Embed & Sign**
4. Minimum Deployment: **iOS 17.0**

### 3. Uncomment SDK code

In `LLMBridgeNexa.swift`:
- Uncomment `import NexaSdk`
- Uncomment `loadModel()` implementation block
- Uncomment `generate()` streaming block
- Remove stub `Task.sleep` / demo token lines

### 4. LFM2 24B specifics

The LFM2 24B-A2B uses **MoE architecture** — only 2B params activate per token.
- Disk: ~4.8 GB (quantized)
- RAM required: ~5–6 GB
- Best on: **iPhone 16 Pro / iPhone 15 Pro** (8GB RAM, A17/A18)
- Backend: `.ane` (Apple Neural Engine) for best speed

Model ID on Nexa Hub: `liquid/lfm2-24b-a2b`

For Snapdragon devices (Android): use **NexaSDK Android** + Qualcomm Hexagon NPU path.

### 5. Run

```swift
// In OasisApp.swift or your entry point, add NexaView to tab bar:
TabItem { NexaView() } label: {
    Label("AI", systemImage: "brain.head.profile")
}
```

### Docs
- NexaSDK iOS: https://docs.nexa.ai/en/nexa-sdk-ios/overview
- Liquid AI LEAP SDK: https://docs.liquid.ai/leap/edge-sdk/ios/ios-quick-start-guide
- GitHub: https://github.com/NexaAI/nexa-sdk
