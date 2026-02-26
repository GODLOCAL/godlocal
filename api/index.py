"""
GodLocal Edge API — zero-dependency Vercel Lambda
Groq called via urllib (no groq package needed)
"""
import json
import os
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

_kill_switch: bool = os.environ.get("XZERO_KILL_SWITCH", "false").lower() == "true"
_thoughts: list = []
_sparks: list = []


def _groq_chat(prompt: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are GodLocal, an autonomous AI trading agent. "
                    "Analyze on-chain data, whale movements, and market signals. "
                    "Be concise and direct. Answer in the user's language."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


class handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress access logs

    def _send_json(self, body: dict, status: int = 200):
        enc = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(enc)

    def do_OPTIONS(self):
        self._send_json({}, 200)

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/") or "/"
        if p in ("/", "/health"):
            self._send_json({"status": "ok", "env": "production", "ts": int(time.time())})
        elif p in ("/status", "/mobile/status"):
            self._send_json({
                "kill_switch_active": _kill_switch,
                "circuit_breaker": {
                    "is_tripped": _kill_switch,
                    "consecutive_losses": 0,
                    "daily_loss_sol": 0.0,
                },
                "sparks": _sparks[-10:],
                "thoughts": _thoughts[-5:],
                "ts": int(time.time()),
            })
        elif p == "/debug-env":
            key = os.environ.get("GROQ_API_KEY", "")
            self._send_json({
                "groq_key_present": bool(key),
                "groq_key_prefix": key[:12] if key else "",
                "groq_key_len": len(key),
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        global _kill_switch, _thoughts
        p = self.path.split("?")[0].rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if p == "/mobile/kill-switch":
            _kill_switch = bool(body.get("active", False))
            self._send_json({"kill_switch_active": _kill_switch, "ok": True})

        elif p == "/think":
            prompt = body.get("prompt", "").strip()
            if not prompt:
                self._send_json({"error": "prompt required"}, 400)
                return
            key = os.environ.get("GROQ_API_KEY", "")
            if not key:
                self._send_json({"error": "GROQ_API_KEY not configured"}, 503)
                return
            try:
                text = _groq_chat(prompt)
                thought = {"thought": text, "prompt": prompt, "timestamp": int(time.time() * 1000)}
                _thoughts.append(thought)
                _thoughts = _thoughts[-20:]
                self._send_json({"response": text, "model": GROQ_MODEL})
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                self._send_json({"error": f"Groq HTTP {e.code}", "detail": err_body, "key_prefix": key[:12]}, 500)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)
