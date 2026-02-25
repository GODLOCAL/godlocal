// MobileOBridge.swift
// GodLocal mobile — unified multimodal understanding + generation on-device
//
// Paper : arXiv:2602.20161 — "Mobile-O: Unified Multimodal Understanding
//         and Generation on Mobile Device"
// App   : iOS App Store ID 6759238106
// Model : FastVLM-0.5B + SANA-600M-512 + Mobile Conditioning Projector (2.4M)
// Size  : 1.6B params total, <2GB RAM
// Speed : iPhone 17 Pro — TTFT 248ms · image gen 512×512 ~3s (no cloud)
//
// Usage:
//   let mo = MobileOBridge()
//   // Image understanding (VQA / OCR / captioning)
//   let caption = try await mo.understand(image: uiImage, prompt: "Describe this")
//   // Text-to-image generation
//   let image   = try await mo.generate(prompt: "neon city at night")

import Foundation
import CoreML
import SwiftUI

// MARK: - MobileOTask

public enum MobileOTask {
    case understand(image: UIImage, prompt: String)
    case generate(prompt: String, steps: Int = 20, size: CGSize = CGSize(width: 512, height: 512))
}

// MARK: - MobileOBridge

@MainActor
public final class MobileOBridge: ObservableObject {
    @Published public var isLoading      = false
    @Published public var isRunning      = false
    @Published public var lastCaption    = ""
    @Published public var lastImage: UIImage?
    @Published public var progress: Double = 0   // 0…1 during generation

    // CoreML model handles (loaded lazily)
    private var vlmModel:    MLModel?
    private var ditModel:    MLModel?
    private var vaeDecoder:  MLModel?
    private var mcpModel:    MLModel?

    public init() {}

    // MARK: Public API

    /// Understand: image → text (VQA, OCR, captioning, reasoning)
    /// Runs FastVLM-0.5B + MCP on ANE via CoreML
    public func understand(image: UIImage, prompt: String) async throws -> String {
        isRunning = true; defer { isRunning = false }
        try await ensureLoaded()
        return try await runUnderstanding(image: image, prompt: prompt)
    }

    /// Generate: text → 512×512 image
    /// Runs SANA-600M-512 DiT (flow-matching) + VAE decoder on ANE via CoreML
    public func generate(prompt: String, steps: Int = 20) async throws -> UIImage {
        isRunning = true; defer { isRunning = false; progress = 0 }
        try await ensureLoaded()
        let image = try await runGeneration(prompt: prompt, steps: steps)
        lastImage = image
        return image
    }

    // MARK: Private — Load

    private func ensureLoaded() async throws {
        guard vlmModel == nil else { return }
        isLoading = true
        defer { isLoading = false }

        // Resolve bundled / downloaded CoreML packages
        // Expected in app Documents/MobileO/ after first-launch download
        let base = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MobileO")

        let vlmURL  = base.appendingPathComponent("MobileO_VLM.mlpackage")
        let ditURL  = base.appendingPathComponent("MobileO_DiT.mlpackage")
        let vaeURL  = base.appendingPathComponent("MobileO_VAE.mlpackage")
        let mcpURL  = base.appendingPathComponent("MobileO_MCP.mlpackage")

        guard FileManager.default.fileExists(atPath: vlmURL.path) else {
            throw MobileOError.weightsNotDownloaded
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all   // ANE + GPU + CPU
        vlmModel   = try await MLModel.load(contentsOf: vlmURL, configuration: cfg)
        ditModel   = try await MLModel.load(contentsOf: ditURL, configuration: cfg)
        vaeDecoder = try await MLModel.load(contentsOf: vaeURL, configuration: cfg)
        mcpModel   = try await MLModel.load(contentsOf: mcpURL, configuration: cfg)
    }

    // MARK: Private — Inference stubs
    // Real inference calls the loaded MLModel prediction APIs.
    // These stubs compile without weights — replace with CoreML prediction
    // calls once MobileO_*.mlpackage files are present.

    private func runUnderstanding(image: UIImage, prompt: String) async throws -> String {
        guard vlmModel != nil else { throw MobileOError.notLoaded }
        // TODO: preprocess image → CVPixelBuffer → vlmModel.prediction()
        // → feed MCP conditioning → return decoded text
        return "[MobileO] understanding stub — implement CoreML prediction"
    }

    private func runGeneration(prompt: String, steps: Int) async throws -> UIImage {
        guard ditModel != nil else { throw MobileOError.notLoaded }
        // TODO: encode prompt via VLM text encoder
        //       → DiT denoising loop (steps iterations, update progress)
        //       → VAE decode latents → UIImage
        for i in 0..<steps {
            progress = Double(i + 1) / Double(steps)
            try await Task.sleep(nanoseconds: 1_000_000) // yield
        }
        return UIImage()
    }
}

// MARK: - MobileOError

public enum MobileOError: LocalizedError {
    case weightsNotDownloaded
    case notLoaded
    case inferenceFailure(String)

    public var errorDescription: String? {
        switch self {
        case .weightsNotDownloaded:
            return "MobileO weights not found in Documents/MobileO/. Download via App Store ID 6759238106 or HuggingFace."
        case .notLoaded:
            return "MobileO models not loaded — call understand() or generate() first."
        case .inferenceFailure(let msg):
            return "MobileO inference failed: \(msg)"
        }
    }
}

// MARK: - MobileOView (SwiftUI preview widget)

public struct MobileOView: View {
    @StateObject private var mo = MobileOBridge()
    @State private var prompt = ""
    @State private var mode: Mode = .generate

    public enum Mode: String, CaseIterable { case generate = "Generate", understand = "Understand" }

    private let neonBlue  = Color(hex: "#00f0ff")
    private let neonPink  = Color(hex: "#ff00ff")
    private let bgColor   = Color(red: 0.02, green: 0.02, blue: 0.06)

    public init() {}

    public var body: some View {
        ZStack {
            bgColor.ignoresEdges()
            VStack(spacing: 16) {
                // Header
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("MOBILE-O")
                            .font(.system(size: 16, weight: .black, design: .monospaced))
                            .foregroundStyle(LinearGradient(
                                colors: [neonBlue, neonPink], startPoint: .leading, endPoint: .trailing))
                        Text("arXiv:2602.20161 · 1.6B · <2GB · ~3s/img")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.white.opacity(0.35))
                    }
                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)

                // Mode picker
                Picker("Mode", selection: $mode) {
                    ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 20)

                // Output
                Group {
                    if let img = mo.lastImage, mode == .generate {
                        Image(uiImage: img)
                            .resizable().scaledToFit()
                            .cornerRadius(12)
                            .padding(.horizontal, 20)
                    } else if !mo.lastCaption.isEmpty, mode == .understand {
                        Text(mo.lastCaption)
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundColor(.white.opacity(0.85))
                            .padding()
                            .background(Color.white.opacity(0.04))
                            .cornerRadius(12)
                            .padding(.horizontal, 20)
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.white.opacity(0.04))
                            .frame(height: 200)
                            .overlay(
                                Text(mo.isRunning ? "running…" : "output appears here")
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.2))
                            )
                            .padding(.horizontal, 20)
                    }
                }

                // Progress bar (generation only)
                if mo.isRunning && mode == .generate {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.white.opacity(0.08)).frame(height: 3)
                            Capsule().fill(neonBlue).frame(width: geo.size.width * mo.progress, height: 3)
                        }
                    }
                    .frame(height: 3)
                    .padding(.horizontal, 20)
                }

                // Input + Run
                HStack(spacing: 10) {
                    TextField(mode == .generate ? "Describe the image…" : "Ask about the image…",
                              text: $prompt)
                        .font(.system(size: 13, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(Color.white.opacity(0.06))
                        .cornerRadius(20)
                        .overlay(RoundedRectangle(cornerRadius: 20)
                            .stroke(neonBlue.opacity(0.3), lineWidth: 1))

                    Button {
                        guard !prompt.isEmpty else { return }
                        Task {
                            if mode == .generate {
                                _ = try? await mo.generate(prompt: prompt)
                            }
                        }
                    } label: {
                        Image(systemName: mo.isRunning ? "hourglass" : "bolt.fill")
                            .foregroundColor(prompt.isEmpty ? .gray : .black)
                            .frame(width: 40, height: 40)
                            .background(prompt.isEmpty ? Color.white.opacity(0.08) : neonBlue)
                            .clipShape(Circle())
                    }
                    .disabled(prompt.isEmpty || mo.isRunning)
                }
                .padding(.horizontal, 16).padding(.bottom, 16)
            }
        }
    }
}

#Preview { MobileOView() }
