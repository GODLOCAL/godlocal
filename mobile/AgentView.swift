import SwiftUI

/// AgentView — Hot-swap specialist agents from iPhone (Developer Pro).
struct AgentView: View {
    @EnvironmentObject var config: AppConfig
    @State private var poolStatus: [String: Any] = [:]
    @State private var swapResult: String?
    @State private var loading = false

    let agents = ["default", "coding", "trading", "writing", "medical", "sleep"]
    let icons  = ["🌐", "💻", "📈", "✍️", "🏥", "🌙"]

    var current: String { poolStatus["current"] as? String ?? "unknown" }

    var body: some View {
        NavigationView {
            VStack(spacing: 12) {
                if let swap = swapResult {
                    Text(swap)
                        .font(.caption.monospaced())
                        .foregroundColor(.green)
                        .padding(8)
                        .background(Color(white: 0.07))
                        .cornerRadius(8)
                }

                ForEach(Array(zip(agents, icons)), id: \.0) { agent, icon in
                    Button(action: { Task { await swap(agent) } }) {
                        HStack {
                            Text("\(icon) \(agent)")
                                .foregroundColor(current == agent ? .black : .green)
                                .fontWeight(current == agent ? .bold : .regular)
                            Spacer()
                            if current == agent {
                                Text("ACTIVE").font(.caption.bold()).foregroundColor(.black)
                            }
                        }
                        .padding(12)
                        .background(current == agent ? Color.green : Color(white: 0.08))
                        .cornerRadius(10)
                        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.green.opacity(0.4)))
                    }
                }
                Spacer()
            }
            .padding()
            .background(Color.black.ignoresSafeArea())
            .navigationTitle("Agent Pool")
            .task { await loadStatus() }
        }
    }

    func loadStatus() async {
        do {
            poolStatus = try await API.get("/agent/status", config: config)
        } catch { }
    }

    func swap(_ agent: String) async {
        loading = true; swapResult = "Swapping to \(agent)…"
        do {
            let res = try await API.post("/agent/swap/\(agent)", body: [:], config: config)
            swapResult = "✅ " + (res["status"] as? String ?? "swapped")
            poolStatus = try await API.get("/agent/status", config: config)
        } catch {
            swapResult = "❌ \(error.localizedDescription)"
        }
        loading = false
    }
}
