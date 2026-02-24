import SwiftUI

/// LogView — Tail the autogenesis evolution log in real-time.
struct LogView: View {
    @EnvironmentObject var config: AppConfig
    @State private var log = ""
    @State private var lines = 50
    @State private var loading = false

    var body: some View {
        NavigationView {
            VStack {
                HStack {
                    Stepper("Lines: \(lines)", value: $lines, in: 10...200, step: 10)
                        .foregroundColor(.cyan)
                    Button("Refresh") { Task { await load() } }
                        .foregroundColor(.green)
                }
                .padding(.horizontal)

                ScrollView {
                    ScrollViewReader { proxy in
                        Text(log.isEmpty ? "No log entries yet." : log)
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.green)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .id("bottom")
                        .onChange(of: log) { _ in
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                }
                .background(Color(white: 0.05))
                .cornerRadius(8)
                .padding()
            }
            .background(Color.black.ignoresSafeArea())
            .navigationTitle("NeuroLog")
            .task { await load() }
        }
    }

    func load() async {
        loading = true
        do {
            let res = try await API.get("/log?lines=\(lines)", config: config)
            log = res["log"] as? String ?? ""
        } catch {
            log = "Error: \(error.localizedDescription)"
        }
        loading = false
    }
}
