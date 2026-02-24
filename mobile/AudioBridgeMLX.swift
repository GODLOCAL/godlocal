// AudioBridgeMLX.swift
// GodLocal mobile — on-device TTS + STT via MLX-Audio-Swift
// Integrates: github.com/Prince_Canuma/MLX-Audio-Swift (v0.1.0)
//
// SETUP (after installing MLX-Audio-Swift via SPM):
//   File → Add Package → https://github.com/Prince_Canuma/MLX-Audio-Swift
//
// Usage:
//   let audio = AudioBridgeMLX()
//   await audio.speak("Hello from GodLocal", voice: .qwen3)
//   let text = await audio.transcribe(audioURL: fileURL)

import Foundation
import AVFoundation

#if canImport(MLXAudio)
import MLXAudio

// MARK: - TTS Voices

public enum TTSVoice: String, CaseIterable {
    case qwen3   = "Qwen3-TTS"
    case marvis  = "Marvis"
    case soprano = "Soprano"
    case pocket  = "Pocket"

    var modelId: String { rawValue }
}

// MARK: - STT Models

public enum STTModel: String, CaseIterable {
    case lfm25     = "LFM-2.5-Audio"       // Liquid AI — fits our ANE stack
    case voxtral   = "Voxtral-Realtime"    // real-time streaming
    case parakeet  = "Parakeet"
    case qwen3Asr  = "Qwen3-ASR"

    var modelId: String { rawValue }
}

// MARK: - AudioBridgeMLX

@MainActor
public final class AudioBridgeMLX: ObservableObject {
    @Published public var isLoading = false
    @Published public var isSpeaking = false
    @Published public var isTranscribing = false
    @Published public var lastTranscription = ""
    @Published public var selectedVoice: TTSVoice = .qwen3
    @Published public var selectedSTT: STTModel = .lfm25

    private var tts: any TTSEngine?
    private var stt: any STTEngine?

    public init() {}

    // MARK: TTS

    public func loadTTS(voice: TTSVoice? = nil) async throws {
        let v = voice ?? selectedVoice
        isLoading = true
        defer { isLoading = false }
        tts = try await MLXAudio.loadTTS(modelId: v.modelId)
    }

    public func speak(_ text: String, voice: TTSVoice? = nil) async throws {
        if tts == nil { try await loadTTS(voice: voice) }
        isSpeaking = true
        defer { isSpeaking = false }
        // Streaming synthesis — plays directly via AVAudioEngine
        try await tts?.synthesize(text, streaming: true)
    }

    // MARK: STT

    public func loadSTT(model: STTModel? = nil) async throws {
        let m = model ?? selectedSTT
        isLoading = true
        defer { isLoading = false }
        stt = try await MLXAudio.loadSTT(modelId: m.modelId)
    }

    public func transcribe(audioURL: URL, model: STTModel? = nil) async throws -> String {
        if stt == nil { try await loadSTT(model: model) }
        isTranscribing = true
        defer { isTranscribing = false }
        let result = try await stt?.transcribe(audioURL)
        lastTranscription = result ?? ""
        return lastTranscription
    }

    public func transcribeLive(onPartial: @escaping (String) -> Void) async throws {
        if stt == nil { try await loadSTT() }
        isTranscribing = true
        defer { isTranscribing = false }
        try await stt?.transcribeLive(onPartial: onPartial)
    }
}

#else

// MARK: - Stub (MLX-Audio-Swift not yet installed)

@MainActor
public final class AudioBridgeMLX: ObservableObject {
    @Published public var isLoading = false
    @Published public var isSpeaking = false
    @Published public var isTranscribing = false
    @Published public var lastTranscription = ""

    public init() {}

    public func speak(_ text: String, voice: Void? = nil) async throws {
        print("[AudioBridgeMLX STUB] TTS: \(text)")
        print("[AudioBridgeMLX] Add MLX-Audio-Swift via SPM:")
        print("  https://github.com/Prince_Canuma/MLX-Audio-Swift")
    }

    public func transcribe(audioURL: URL, model: Void? = nil) async throws -> String {
        print("[AudioBridgeMLX STUB] STT: \(audioURL.lastPathComponent)")
        return "[STUB — install MLX-Audio-Swift]"
    }

    public func transcribeLive(onPartial: @escaping (String) -> Void) async throws {
        onPartial("[STUB — install MLX-Audio-Swift]")
    }
}

#endif
