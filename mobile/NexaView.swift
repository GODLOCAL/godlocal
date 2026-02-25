// mobile/NexaView.swift
// GodLocal v6.8 — On-device AI chat view powered by NexaSDK
// v7.0.5 — mic button via AudioBridgeMLX (Qwen3-ASR-0.6B realtime)

import SwiftUI

struct NexaView: View {
    @StateObject private var bridge = LLMBridgeNexa()
    @StateObject private var audio  = AudioBridgeMLX()   // ← MLX-Audio-Swift
    @State private var inputText = ""
    @State private var selectedModel: NexaModel = .paro_4b
    @State private var selectedBackend: NexaBackend = .ane
    @State private var messages: [(role: String, text: String)] = []
    @State private var isRecording = false

    private let neonBlue   = Color(hex: "#00f0ff")
    private let neonPink   = Color(hex: "#ff00ff")
    private let neonGreen  = Color(hex: "#00FF41")
    private let bgColor    = Color(red: 0.02, green: 0.02, blue: 0.06)

    var body: some View {
        ZStack {
            bgColor.ignoresEdges()

            VStack(spacing: 0) {
                // ── Header ──
                headerBar

                // ── Model picker ──
                modelPicker
                    .padding(.horizontal, 16)
                    .padding(.top, 8)

                // ── Chat messages ──
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(Array(messages.enumerated()), id: \.offset) { i, msg in
                                MessageBubble(role: msg.role, text: msg.text)
                                    .id(i)
                            }
                            if bridge.isGenerating {
                                streamBubble
                            }
                        }
                        .padding(16)
                    }
                    .onChange(of: messages.count) { _ in
                        if let last = messages.indices.last {
                            withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                        }
                    }
                }

                // ── Status bar ──
                statusBar

                // ── Input ──
                inputBar
            }
        }
        .navigationTitle("")
        .navigationBarHidden(true)
        .task {
            await bridge.loadModel(selectedModel, backend: selectedBackend)
        }
    }

    // MARK: - Header

    var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("GODLOCAL")
                    .font(.system(size: 18, weight: .black, design: .monospaced))
                    .foregroundStyle(
                        LinearGradient(colors: [neonBlue, neonPink], startPoint: .leading, endPoint: .trailing)
                    )
                Text("On-Device · No Cloud")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.4))
            }
            Spacer()
            // TPS badge
            if bridge.tokensPerSecond > 0 {
                Text(String(format: "%.0f tok/s", bridge.tokensPerSecond))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(neonGreen)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(neonGreen.opacity(0.12))
                    .clipShape(Capsule())
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
    }

    // MARK: - Model Picker

    var modelPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(NexaModel.allCases) { model in
                    Button {
                        selectedModel = model
                        Task { await bridge.loadModel(model, backend: selectedBackend) }
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.displayName)
                                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            Text(String(format: "%.1fGB", model.sizeGB))
                                .font(.system(size: 9))
                                .opacity(0.6)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(selectedModel == model ? neonBlue.opacity(0.2) : Color.white.opacity(0.04))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(selectedModel == model ? neonBlue : Color.white.opacity(0.1), lineWidth: 1)
                        )
                        .cornerRadius(8)
                    }
                    .foregroundColor(selectedModel == model ? neonBlue : .white.opacity(0.6))
                }
            }
        }
    }

    // MARK: - Stream bubble

    var streamBubble: some View {
        HStack(alignment: .top, spacing: 8) {
            Circle().fill(neonBlue).frame(width: 6, height: 6).padding(.top, 6)
            Text(bridge.output.isEmpty ? "▊" : bridge.output + "▊")
                .font(.system(size: 14, design: .monospaced))
                .foregroundColor(.white.opacity(0.9))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Status Bar

    var statusBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(bridge.isLoaded ? neonGreen : Color.yellow)
                .frame(width: 6, height: 6)
            Text(bridge.statusMessage)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.5))
            Spacer()
            // STT status
            if audio.isTranscribing {
                HStack(spacing: 4) {
                    Circle().fill(neonPink).frame(width: 5, height: 5)
                    Text("listening…")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(neonPink.opacity(0.8))
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Color.white.opacity(0.03))
    }

    // MARK: - Input Bar

    var inputBar: some View {
        HStack(spacing: 10) {
            TextField("Ask GodLocal…", text: $inputText, axis: .vertical)
                .font(.system(size: 14, design: .rounded))
                .foregroundColor(.white)
                .lineLimit(1...5)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.06))
                .cornerRadius(20)
                .overlay(RoundedRectangle(cornerRadius: 20).stroke(neonBlue.opacity(0.3), lineWidth: 1))

            // ── Mic button (Qwen3-ASR-0.6B live transcription) ──
            Button {
                isRecording ? stopRecording() : startRecording()
            } label: {
                Image(systemName: isRecording ? "stop.circle.fill" : "mic.fill")
                    .foregroundColor(isRecording ? neonPink : neonBlue.opacity(0.85))
                    .frame(width: 40, height: 40)
                    .background((isRecording ? neonPink : neonBlue).opacity(0.12))
                    .clipShape(Circle())
                    .overlay(Circle().stroke((isRecording ? neonPink : neonBlue).opacity(0.35), lineWidth: 1))
            }

            // ── Send / Stop ──
            if bridge.isGenerating {
                Button { bridge.cancelGeneration() } label: {
                    Image(systemName: "stop.fill")
                        .foregroundColor(neonPink)
                        .frame(width: 40, height: 40)
                        .background(neonPink.opacity(0.15))
                        .clipShape(Circle())
                }
            } else {
                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "arrow.up")
                        .foregroundColor(inputText.isEmpty ? .gray : .black)
                        .frame(width: 40, height: 40)
                        .background(inputText.isEmpty ? Color.white.opacity(0.08) : neonBlue)
                        .clipShape(Circle())
                }
                .disabled(inputText.isEmpty || !bridge.isLoaded)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    // MARK: - Actions

    private func startRecording() {
        isRecording = true
        Task {
            do {
                try await audio.transcribeLive { partial in
                    Task { @MainActor in
                        inputText = partial
                    }
                }
            } catch {
                print("[NexaView] STT error: \(error)")
            }
            isRecording = false
        }
    }

    private func stopRecording() {
        isRecording = false
        // MLX-Audio-Swift transcribeLive stops when audio session ends
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        messages.append((role: "user", text: text))

        Task {
            await bridge.generate(prompt: buildPrompt(for: text))
            messages.append((role: "assistant", text: bridge.output))
        }
    }

    private func buildPrompt(for query: String) -> String {
        let history = messages.suffix(6).map { m in
            m.role == "user" ? "User: \(m.text)" : "Assistant: \(m.text)"
        }.joined(separator: "\n")
        return "\(history)\nUser: \(query)\nAssistant:"
    }
}

// MARK: - MessageBubble

struct MessageBubble: View {
    let role: String
    let text: String

    private let neonBlue  = Color(hex: "#00f0ff")
    private let neonPink  = Color(hex: "#ff00ff")

    var isUser: Bool { role == "user" }

    var body: some View {
        HStack {
            if isUser { Spacer(minLength: 40) }
            Text(text)
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.9))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    isUser
                        ? LinearGradient(colors: [neonBlue.opacity(0.25), neonPink.opacity(0.15)], startPoint: .topLeading, endPoint: .bottomTrailing)
                        : LinearGradient(colors: [Color.white.opacity(0.05), Color.white.opacity(0.03)], startPoint: .topLeading, endPoint: .bottomTrailing)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(isUser ? neonBlue.opacity(0.4) : Color.white.opacity(0.08), lineWidth: 1)
                )
                .cornerRadius(16)
            if !isUser { Spacer(minLength: 40) }
        }
    }
}

// MARK: - Color hex extension
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let r = Double((int >> 16) & 0xFF) / 255
        let g = Double((int >> 8) & 0xFF) / 255
        let b = Double(int & 0xFF) / 255
        self.init(red: r, green: g, blue: b)
    }
}

#Preview {
    NexaView()
}
