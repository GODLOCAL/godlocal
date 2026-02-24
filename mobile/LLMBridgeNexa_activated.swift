// mobile/LLMBridgeNexa.swift
// GodLocal v6.8 — NexaSDK on-device LLM bridge
// Supports: LFM2 24B-A2B (Liquid AI MoE), Qwen3, MLX models via NexaML engine

import Foundation

// MARK: - NexaSDK Integration
// SDK: https://docs.nexa.ai/en/nexa-sdk-ios/overview
// xcframework: https://nexa-model-hub-bucket.s3.us-west-1.amazonaws.com/public/ios/latest/NexaSdk.xcframework.zip
// iOS 17.0+ required, Apple Neural Engine acceleration

import NexaSdk

/// Supported on-device model backends
enum NexaBackend: String, CaseIterable {
    case ane   = "ANE"      // Apple Neural Engine — fastest on A17/M-series
    case gpu   = "GPU"      // Metal GPU — broad compatibility
    case cpu   = "CPU"      // CPU fallback
}

/// Model catalog — Nexa Hub compatible IDs
enum NexaModel: String, CaseIterable, Identifiable {
    var id: String { rawValue }

    // Liquid AI — MoE, 2B active / 24B total (via NexaSDK Day-0)
    case lfm2_24b     = "liquid/lfm2-24b-a2b"
    // Qwen3 — GodLocal primary (existing)
    case qwen3_4b     = "Qwen/Qwen3-4B-Instruct-GGUF"
    case qwen3_8b     = "Qwen/Qwen3-8B-Instruct-GGUF"
    // ParoQuant (existing GodLocal preference)
    case paro_4b      = "z-lab/Qwen3-4B-PARO"
    case paro_8b      = "z-lab/Qwen3-8B-PARO"

    var displayName: String {
        switch self {
        case .lfm2_24b:  return "LFM2 24B (Liquid AI)"
        case .qwen3_4b:  return "Qwen3 4B"
        case .qwen3_8b:  return "Qwen3 8B"
        case .paro_4b:   return "PARO 4B (GodLocal)"
        case .paro_8b:   return "PARO 8B (GodLocal)"
        }
    }

    var sizeGB: Float {
        switch self {
        case .lfm2_24b:  return 4.8  // MoE: only 2B active, ~4.8GB disk
        case .qwen3_4b:  return 2.4
        case .qwen3_8b:  return 4.9
        case .paro_4b:   return 1.8
        case .paro_8b:   return 4.0
        }
    }
}

/// Response stream token
struct NexaToken {
    let text: String
    let isFinished: Bool
    let tokensPerSecond: Float?
}

/// LLMBridgeNexa — wraps NexaSDK for GodLocal SwiftUI views
@MainActor
final class LLMBridgeNexa: ObservableObject {

    @Published var isLoaded = false
    @Published var isGenerating = false
    @Published var loadedModel: NexaModel?
    @Published var statusMessage = "Ready"
    @Published var tokensPerSecond: Float = 0
    @Published var output = ""

    private var currentBackend: NexaBackend = .ane
    private var llm: Llm?

    // MARK: - Model Loading

    func loadModel(_ model: NexaModel, backend: NexaBackend = .ane) async {
        statusMessage = "Loading \(model.displayName)..."
        currentBackend = backend
        isLoaded = false

        do {
            llm = try Llm(plugin: backend == .ane ? .ane : backend == .gpu ? .gpu : .cpu)
            let modelURL = try await NexaModelHub.download(model.rawValue)
            try await llm?.load(from: modelURL)
            isLoaded = true
            loadedModel = model
            statusMessage = "\(model.displayName) ready (\(backend.rawValue))"
        } catch {
            statusMessage = "Load failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Inference

    func generate(prompt: String, maxTokens: Int = 512) async {
        guard isLoaded else {
            statusMessage = "No model loaded"
            return
        }
        isGenerating = true
        output = ""

        let stream = try await llm?.generate(
            prompt: prompt,
            options: .init(maxNewTokens: maxTokens, temperature: 0.7)
        )
        for try await token in stream ?? [] {
            output += token.text
            if let tps = token.tokensPerSecond { tokensPerSecond = tps }
            if token.isFinished { break }
        }

        isGenerating = false
        statusMessage = "Done · \(String(format: "%.1f", tokensPerSecond)) tok/s"
    }

    func cancelGeneration() {
        llm?.cancel()
        isGenerating = false
        statusMessage = "Cancelled"
    }
}
