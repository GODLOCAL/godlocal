// MessagesViewController.swift
// GodLocal iMessage Extension — AI agent natively inside Messages.app
// Target: iPhone 17 Pro, iOS 18+
// Architecture: MessagesExtension → GodLocalClient.think() → Picobot VPS
//               (or on-device LLM via MobileOBridge when Picobot unreachable)

import Messages
import SwiftUI

// MARK: - MessagesViewController (entry point)
final class MessagesViewController: MSMessagesAppViewController {

    private var hostingController: UIHostingController<GodLocalChatView>?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = UIColor(red: 0, green: 0, blue: 0, alpha: 1) // #000000
        embedChatView()
    }

    private func embedChatView() {
        let chatView = GodLocalChatView(
            onSend: { [weak self] message in
                self?.insertMessage(message)
            }
        )
        let hc = UIHostingController(rootView: chatView)
        addChild(hc)
        hc.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(hc.view)
        NSLayoutConstraint.activate([
            hc.view.topAnchor.constraint(equalTo: view.topAnchor),
            hc.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hc.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hc.view.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        hc.didMove(toParent: self)
        hostingController = hc
    }

    // Insert AI response as iMessage bubble
    private func insertMessage(_ text: String) {
        guard let conversation = activeConversation else { return }
        let layout = MSMessageTemplateLayout()
        layout.caption = "GodLocal 🤖"
        layout.subcaption = String(text.prefix(100))
        layout.trailingCaption = "on-device"

        let message = MSMessage()
        message.layout = layout
        message.summaryText = text

        conversation.insert(message) { error in
            if let error = error { print("[GodLocal] insert error: \(error)") }
        }
    }
}

// MARK: - GodLocalChatView (SwiftUI)
struct GodLocalChatView: View {
    let onSend: (String) -> Void

    @State private var inputText   = ""
    @State private var messages:    [(role: String, text: String)] = []
    @State private var isThinking  = false
    @State private var backendAlive = true

    private let client = GodLocalConfig.Client()

    // Brand colors
    private let neonGreen  = Color(hex: "#00FF41")
    private let neonCyan   = Color(hex: "#00E5FF")
    private let neonPurple = Color(hex: "#7B2FFF")
    private let bgColor    = Color(hex: "#050505")

    var body: some View {
        ZStack {
            bgColor.ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("GODLOCAL")
                        .font(.system(.caption, design: .monospaced).bold())
                        .foregroundColor(neonGreen)
                    Spacer()
                    Circle()
                        .fill(backendAlive ? neonGreen : Color.red)
                        .frame(width: 8, height: 8)
                    Text(backendAlive ? "PICOBOT" : "OFFLINE")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(backendAlive ? neonGreen : .red)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black)

                Divider().background(neonGreen.opacity(0.3))

                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(Array(messages.enumerated()), id: \.offset) { idx, msg in
                                MessageBubble(
                                    text: msg.text,
                                    isUser: msg.role == "user",
                                    neonGreen: neonGreen,
                                    neonCyan: neonCyan
                                )
                                .id(idx)
                            }

                            if isThinking {
                                ThinkingDots(color: neonCyan)
                                    .id("thinking")
                            }
                        }
                        .padding(12)
                    }
                    .onChange(of: messages.count) { _ in
                        withAnimation { proxy.scrollTo(messages.count - 1) }
                    }
                    .onChange(of: isThinking) { _ in
                        if isThinking { withAnimation { proxy.scrollTo("thinking") } }
                    }
                }

                Divider().background(neonGreen.opacity(0.3))

                // Input bar
                HStack(spacing: 8) {
                    TextField("", text: $inputText,
                              prompt: Text("ask godlocal...")
                                .foregroundColor(neonGreen.opacity(0.4))
                                .font(.system(.body, design: .monospaced)))
                        .font(.system(.body, design: .monospaced))
                        .foregroundColor(neonGreen)
                        .tint(neonCyan)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(neonGreen.opacity(0.4), lineWidth: 1)
                        )
                        .onSubmit { sendMessage() }

                    Button(action: sendMessage) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundColor(inputText.isEmpty || isThinking ? neonGreen.opacity(0.3) : neonGreen)
                    }
                    .disabled(inputText.isEmpty || isThinking)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black)
            }
        }
        .task { backendAlive = await client.isAlive() }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        messages.append((role: "user", text: text))
        inputText  = ""
        isThinking = true

        Task {
            do {
                let response = try await client.think(prompt: text)
                await MainActor.run {
                    messages.append((role: "assistant", text: response))
                    isThinking = false
                    onSend(response) // insert into iMessage thread
                }
            } catch {
                await MainActor.run {
                    messages.append((role: "assistant", text: "Error: \(error.localizedDescription)"))
                    isThinking = false
                }
            }
        }
    }
}

// MARK: - MessageBubble
struct MessageBubble: View {
    let text: String
    let isUser: Bool
    let neonGreen: Color
    let neonCyan: Color

    var body: some View {
        HStack {
            if isUser { Spacer(minLength: 40) }
            Text(text)
                .font(.system(.body, design: .monospaced))
                .foregroundColor(isUser ? .black : neonCyan)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isUser ? neonGreen : Color(white: 0.07))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(isUser ? Color.clear : neonCyan.opacity(0.3), lineWidth: 1)
                        )
                )
            if !isUser { Spacer(minLength: 40) }
        }
    }
}

// MARK: - ThinkingDots
struct ThinkingDots: View {
    let color: Color
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 4) {
            Spacer(minLength: 40)
            HStack(spacing: 4) {
                ForEach(0..<3) { i in
                    Circle()
                        .fill(color.opacity(phase == i ? 1 : 0.3))
                        .frame(width: 6, height: 6)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(white: 0.07))
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(color.opacity(0.3), lineWidth: 1))
            )
        }
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { _ in
                phase = (phase + 1) % 3
            }
        }
    }
}

// MARK: - Color hex init
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let r = Double((int >> 16) & 0xFF) / 255
        let g = Double((int >>  8) & 0xFF) / 255
        let b = Double( int        & 0xFF) / 255
        self.init(red: r, green: g, blue: b)
    }
}
