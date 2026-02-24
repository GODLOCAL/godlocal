import SwiftUI

/// StatusView — Live GodLocal health & soul snapshot.
struct StatusView: View {
    @EnvironmentObject var config: AppConfig
    @State private var status: [String: Any] = [:]
    @State private var loading = false
    @State private var error: String?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if loading {
                        ProgressView("Loading...").foregroundColor(.green)
                    } else if let err = error {
                        Text("⚠️ \(err)").foregroundColor(.red).font(.caption)
                    } else {
                        metricCard("Soul", status["soul_loaded"] as? Bool == true ? "✅ Loaded" : "—")
                        metricCard("Memory", "\(status["memory_vectors"] ?? "—") vectors")
                        metricCard("Session", status["session_id"] as? String ?? "—")
                        metricCard("sleep_cycle", status["sleep_cycle_last"] as? String ?? "never")
                        metricCard("AutoGenesis", status["autogenesis_evolutions"] != nil ?
                                   "\(status["autogenesis_evolutions"]!) evolutions" : "—")
                    }

                    Divider().background(Color.green.opacity(0.3))

                    NavigationLink("⚙️ Settings", destination: SettingsView())
                        .foregroundColor(.cyan)
                        .padding(.top, 8)
                }
                .padding()
            }
            .background(Color.black.ignoresSafeArea())
            .navigationTitle("🌐 GodLocal Status")
            .navigationBarTitleDisplayMode(.large)
            .task { await load() }
            .refreshable { await load() }
        }
    }

    func load() async {
        loading = true; error = nil
        do {
            status = try await API.get("/mobile/status", config: config)
        } catch {
            self.error = error.localizedDescription
        }
        loading = false
    }

    func metricCard(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundColor(.cyan).frame(width: 120, alignment: .leading)
            Text(value).foregroundColor(.white)
        }
        .padding(10)
        .background(Color(white: 0.07))
        .cornerRadius(8)
    }
}

struct SettingsView: View {
    @EnvironmentObject var config: AppConfig
    var body: some View {
        Form {
            Section("Connection") {
                TextField("Base URL", text: $config.baseURL)
                    .autocapitalization(.none)
                    .keyboardType(.URL)
                SecureField("API Key (optional)", text: $config.apiKey)
            }
            Button("Save") { config.save() }.foregroundColor(.green)
        }
        .navigationTitle("Settings")
        .background(Color.black)
    }
}
