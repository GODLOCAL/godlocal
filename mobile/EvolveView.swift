import SwiftUI

/// EvolveView — Trigger AutoGenesis evolution tasks from iPhone.
struct EvolveView: View {
    @EnvironmentObject var config: AppConfig
    @State private var task = ""
    @State private var applyMode = false
    @State private var result: String?
    @State private var loading = false

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Describe evolution task:")
                    .foregroundColor(.cyan)
                    .frame(maxWidth: .infinity, alignment: .leading)

                TextEditor(text: $task)
                    .frame(height: 120)
                    .padding(8)
                    .background(Color(white: 0.07))
                    .cornerRadius(8)
                    .foregroundColor(.green)
                    .font(.system(.body, design: .monospaced))

                Toggle(isOn: $applyMode) {
                    Text("Apply changes (⚠️ writes to real files)")
                        .foregroundColor(applyMode ? .red : .gray)
                }
                .tint(.red)

                Button(action: { Task { await evolve() } }) {
                    Label(loading ? "Evolving…" : "⚡ Trigger AutoGenesis", systemImage: "arrow.triangle.2.circlepath")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(task.isEmpty ? Color.gray : Color.green.opacity(0.2))
                        .foregroundColor(task.isEmpty ? .gray : .green)
                        .cornerRadius(10)
                        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.green.opacity(0.5)))
                }
                .disabled(task.isEmpty || loading)

                if let result {
                    ScrollView {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.green)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .background(Color(white: 0.05))
                            .cornerRadius(8)
                    }
                    .frame(maxHeight: 220)
                }
                Spacer()
            }
            .padding()
            .background(Color.black.ignoresSafeArea())
            .navigationTitle("AutoGenesis")
        }
    }

    func evolve() async {
        loading = true; result = nil
        do {
            let res = try await API.post("/mobile/evolve",
                body: ["task": task, "apply": applyMode], config: config)
            let proposed = (res["proposed_files"] as? [String] ?? []).joined(separator: ", ")
            let applied  = (res["applied"] as? [[String: Any]] ?? []).map { $0["filename"] as? String ?? "" }.joined(separator: ", ")
            let elapsed  = res["elapsed_s"] as? Double ?? 0
            result = """
Evolution #\(res["evolution"] ?? "?") — \(String(format: "%.1f", elapsed))s
Proposed: \(proposed.isEmpty ? "none" : proposed)
Applied:  \(applied.isEmpty ? "none (dry-run)" : applied)
"""
        } catch {
            result = "Error: \(error.localizedDescription)"
        }
        loading = false
    }
}
