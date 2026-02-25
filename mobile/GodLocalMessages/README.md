# GodLocal iMessage Extension

AI agent natively inside Messages.app — no separate app needed.

## Architecture
```
Messages.app
 └── GodLocal Extension (MessagesViewController)
      └── GodLocalChatView (SwiftUI)
           └── GodLocalConfig.Client.think() → POST http://PICOBOT_IP:8000/think
```

## Files
- `MessagesViewController.swift` — iOS Messages extension entry point + full SwiftUI chat UI
- `Info.plist` — extension manifest (NSExtensionPointIdentifier: com.apple.message-payload-provider)

## Xcode Setup (3 steps)
1. **Add Extension Target**
   - File → New → Target → iMessage Extension
   - Product Name: `GodLocalMessages`
   - Delete generated files, replace with these

2. **Add to main target**
   - GodLocalMessages target → Build Phases → Dependencies → add GodLocal app
   - Embed `GodLocalMessages.appex` in main app

3. **Share Config.swift**
   - `mobile/Config.swift` must be in BOTH targets (GodLocal + GodLocalMessages)
   - Or extract to a shared framework

## Usage
1. Open Messages → tap any conversation
2. Tap `⋯` → App Drawer → GodLocal
3. Type prompt → response appears in thread + auto-inserted as iMessage bubble
4. Other person sees the AI response inline

## On-device fallback (no Picobot)
When `isAlive()` returns false, status dot turns red.
Wire `MobileOBridge.understand()` as fallback — 100% offline.

## Brand
- Background: #000000
- Text / accents: #00FF41 (neon green), #00E5FF (cyan), #7B2FFF (purple)
- Font: SF Mono / monospaced
