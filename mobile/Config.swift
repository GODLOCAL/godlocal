// Config.swift — GodLocal base URL configuration
// iPhone connects directly to Picobot VPS — no Mac needed

import Foundation

public enum GodLocalConfig {

    // ── Backend URL ──────────────────────────────────────────────────────────
    // Priority: env override → UserDefaults → Picobot default
    public static var backendURL: URL {
        if let override = UserDefaults.standard.string(forKey: "godlocal_backend_url"),
           let url = URL(string: override) { return url }
        return URL(string: defaultBackendURL)!
    }

    /// Default: Picobot VPS (no Mac needed)
    /// Override via Settings.bundle or `godlocal://set-url?url=...`
    public static let defaultBackendURL = ProcessInfo.processInfo
        .environment["GODLOCAL_BACKEND_URL"] ?? "http://YOUR_PICOBOT_IP:8000"

    public static func setBackend(_ urlString: String) {
        UserDefaults.standard.set(urlString, forKey: "godlocal_backend_url")
    }

    // ── API Endpoints ────────────────────────────────────────────────────────
    public enum API {
        public static func think()         -> URL { GodLocalConfig.backendURL.appendingPathComponent("/think") }
        public static func status()        -> URL { GodLocalConfig.backendURL.appendingPathComponent("/status") }
        public static func health()        -> URL { GodLocalConfig.backendURL.appendingPathComponent("/health") }
        public static func goals()         -> URL { GodLocalConfig.backendURL.appendingPathComponent("/goals") }
        public static func sparks()        -> URL { GodLocalConfig.backendURL.appendingPathComponent("/sparks/evoke") }
        public static func swap()          -> URL { GodLocalConfig.backendURL.appendingPathComponent("/xzero/swap") }
        public static func agentPool()     -> URL { GodLocalConfig.backendURL.appendingPathComponent("/agents/pool") }
    }

    // ── GodLocalClient — async HTTP wrapper ─────────────────────────────────
    public struct Client {
        private let session = URLSession.shared
        private let decoder = JSONDecoder()

        public init() {}

        /// POST /think  → agent response
        public func think(prompt: String) async throws -> String {
            var req = URLRequest(url: API.think())
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["input": prompt])
            req.timeoutInterval = 60

            let (data, resp) = try await session.data(for: req)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                throw GodLocalError.httpError((resp as? HTTPURLResponse)?.statusCode ?? 0)
            }
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            return json?["response"] as? String ?? ""
        }

        /// GET /health → Bool
        public func isAlive() async -> Bool {
            guard let req = try? URLRequest(url: API.health()) else { return false }
            let result = try? await session.data(for: req)
            return (result?.1 as? HTTPURLResponse)?.statusCode == 200
        }
    }
}

// MARK: - GodLocalError
public enum GodLocalError: LocalizedError {
    case httpError(Int)
    case noBackend
    case decodingFailed

    public var errorDescription: String? {
        switch self {
        case .httpError(let code): return "Backend returned HTTP \(code)"
        case .noBackend:           return "GodLocal backend not reachable. Check Picobot is running."
        case .decodingFailed:      return "Failed to decode backend response."
        }
    }
}
