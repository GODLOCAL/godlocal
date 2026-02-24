import SwiftUI

/// GodLocal OASIS — Sovereign AI control centre for iPhone.
/// Connects to local GodLocal instance via configurable base URL.
/// Use Tailscale or ngrok to reach your Mac/Steam Deck remotely.

let DEFAULT_BASE_URL = "http://127.0.0.1:8000"

@main
struct OasisApp: App {
    @StateObject private var config = AppConfig()

    var body: some Scene {
        WindowGroup {
            TabView {
                StatusView()
                    .tabItem { Label("БОГ", systemImage: "brain.head.profile") }
                EvolveView()
                    .tabItem { Label("AutoGenesis", systemImage: "arrow.triangle.2.circlepath") }
                LogView()
                    .tabItem { Label("NeuroLog", systemImage: "doc.text") }
                AgentView()
                    .tabItem { Label("Agents", systemImage: "cpu") }
            }
            .environmentObject(config)
            .preferredColorScheme(.dark)
        }
    }
}

// ── Shared config ─────────────────────────────────────────────────────────────
class AppConfig: ObservableObject {
    @Published var baseURL: String = UserDefaults.standard.string(forKey: "base_url") ?? DEFAULT_BASE_URL
    @Published var apiKey: String  = UserDefaults.standard.string(forKey: "api_key")  ?? ""

    func save() {
        UserDefaults.standard.set(baseURL, forKey: "base_url")
        UserDefaults.standard.set(apiKey,  forKey: "api_key")
    }
}

// ── Network helper ────────────────────────────────────────────────────────────
struct API {
    static func get(_ path: String, config: AppConfig) async throws -> [String: Any] {
        guard let url = URL(string: config.baseURL + path) else { throw URLError(.badURL) }
        var req = URLRequest(url: url, timeoutInterval: 10)
        if !config.apiKey.isEmpty { req.setValue(config.apiKey, forHTTPHeaderField: "X-API-Key") }
        let (data, _) = try await URLSession.shared.data(for: req)
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    static func post(_ path: String, body: [String: Any], config: AppConfig) async throws -> [String: Any] {
        guard let url = URL(string: config.baseURL + path) else { throw URLError(.badURL) }
        var req = URLRequest(url: url, timeoutInterval: 30)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !config.apiKey.isEmpty { req.setValue(config.apiKey, forHTTPHeaderField: "X-API-Key") }
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, _) = try await URLSession.shared.data(for: req)
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}
