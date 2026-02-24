#!/bin/bash
# setup_nexa.sh — GodLocal NexaSDK one-command setup
# Run from: godlocal/mobile/
# After this: open Xcode → drag NexaSdk.xcframework → Embed & Sign → Cmd+R

set -e
GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}[1/3] Downloading NexaSdk.xcframework...${NC}"
curl -L --progress-bar \
  https://nexa-model-hub-bucket.s3.us-west-1.amazonaws.com/public/ios/latest/NexaSdk.xcframework.zip \
  -o NexaSdk.xcframework.zip
unzip -q -o NexaSdk.xcframework.zip
rm NexaSdk.xcframework.zip
echo -e "${GREEN}✓ NexaSdk.xcframework ready${NC}"

echo -e "${CYAN}[2/3] Activating NexaSDK code in LLMBridgeNexa.swift...${NC}"
SWIFT_FILE="LLMBridgeNexa.swift"

# 1. Uncomment: import NexaSdk
sed -i '' 's|// import NexaSdk|import NexaSdk|' "$SWIFT_FILE"

# 2. Uncomment SDK loadModel block (remove leading // from the 8-line do/catch block)
sed -i '' \
  -e 's|        // do {|        do {|' \
  -e 's|        //     llm = try Llm|        llm = try Llm|' \
  -e 's|        //     let modelURL = try await NexaModelHub|        let modelURL = try await NexaModelHub|' \
  -e 's|        //     try await llm?.load(from: modelURL)|        try await llm?.load(from: modelURL)|' \
  -e 's|        //     isLoaded = true|        isLoaded = true|' \
  -e 's|        //     loadedModel = model|        loadedModel = model|' \
  -e 's|        //     statusMessage = "\(model.displayName) ready (\(backend.rawValue))"|        statusMessage = "\(model.displayName) ready (\(backend.rawValue))"|' \
  -e 's|        // } catch {|        } catch {|' \
  -e 's|        //     statusMessage = "Load failed: \(error.localizedDescription)"|        statusMessage = "Load failed: \(error.localizedDescription)"|' \
  -e 's|        // }|        }|' \
  "$SWIFT_FILE"

# 3. Remove stub loadModel block (3 lines: sleep + isLoaded + loadedModel + statusMessage[STUB])
python3 - <<'PYEOF'
import re
with open("LLMBridgeNexa.swift", "r") as f:
    content = f.read()

# Remove the stub block in loadModel (between the two comment markers)
stub_load = r"        // Stub for development without xcframework:\n        try\? await Task\.sleep\(nanoseconds: 800_000_000\)\n        isLoaded = true\n        loadedModel = model\n        statusMessage = \".*\[STUB\]\"\n"
content = re.sub(stub_load, "", content)

# Uncomment llm declaration
content = content.replace(
    "    // private var llm: Llm?  // NexaSDK object — uncomment after SDK import",
    "    private var llm: Llm?"
)

# Uncomment the NexaSDK streaming block
content = content.replace(
    "        // NexaSDK streaming (uncomment after SDK import):\n        // let stream = try await llm?.generate(\n        //     prompt: prompt,\n        //     options: .init(maxNewTokens: maxTokens, temperature: 0.7)\n        // )\n        // for try await token in stream ?? [] {\n        //     output += token.text\n        //     if let tps = token.tokensPerSecond { tokensPerSecond = tps }\n        //     if token.isFinished { break }\n        // }",
    "        let stream = try await llm?.generate(\n            prompt: prompt,\n            options: .init(maxNewTokens: maxTokens, temperature: 0.7)\n        )\n        for try await token in stream ?? [] {\n            output += token.text\n            if let tps = token.tokensPerSecond { tokensPerSecond = tps }\n            if token.isFinished { break }\n        }"
)

# Remove stub streaming block
stub_stream = r"        // Stub — simulate streaming\n        let demoTokens = .*\.components\(separatedBy: \" \"\)\n        for token in demoTokens \{.*?\}\n\n"
content = re.sub(stub_stream, "", content, flags=re.DOTALL)

# Uncomment llm?.cancel()
content = content.replace("        // llm?.cancel()", "        llm?.cancel()")

with open("LLMBridgeNexa.swift", "w") as f:
    f.write(content)
print("Swift file patched OK")
PYEOF

echo -e "${GREEN}✓ LLMBridgeNexa.swift activated${NC}"

echo ""
echo -e "${CYAN}[3/3] Done! One manual step remaining:${NC}"
echo ""
echo "  1. Open Xcode project (mobile/GodLocal.xcodeproj)"
echo "  2. Drag 'NexaSdk.xcframework' into Project Navigator"
echo "  3. Frameworks, Libraries → set to 'Embed & Sign'"
echo "  4. Cmd+R — done"
echo ""
echo -e "${GREEN}Expected: LFM2 24B → ~40 tok/s | PARO 4B → ~60 tok/s${NC}"
