"""
GodLocal Edge API — zero-dependency Vercel Lambda
Groq called via urllib; live market data via CoinGecko (no key needed)
"""
import json
import os
import time
import datetime
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum,solana,binancecoin,sui"
    "&vs_currencies=usd"
    "&include_24hr_change=true"
    "&include_market_cap=true"
)
GROQ_MODEL = "llama-3.1-8b-instant"

_kill_switch: bool = os.environ.get("XZERO_KILL_SWITCH", "false").lower() == "true"
_thoughts: list = []
_sparks: list = []

# Simple in-memory cache for market data (5 min TTL)
_market_cache: dict = {"data": None, "fetched_at": 0}
_MARKET_TTL = 300  # seconds


def _fetch_market_data() -> str:
    """Fetch live prices from CoinGecko. Returns formatted string for system prompt."""
    now = time.time()
    if _market_cache["data"] and (now - _market_cache["fetched_at"]) < _MARKET_TTL:
        return _market_cache["data"]
    try:
        req = urllib.request.Request(
            COINGECKO_URL,
            headers={"User-Agent": "GodLocal/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = json.loads(resp.read())
        lines = []
        names = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "solana": "SOL",
            "binancecoin": "BNB",
            "sui": "SUI",
        }
        for coin_id, sym in names.items():
            d = raw.get(coin_id, {})
            price = d.get("usd", "?")
            chg = d.get("usd_24h_change", 0)
            cap = d.get("usd_market_cap", 0)
            sign = "+" if chg and chg >= 0 else ""
            cap_b = f"{cap/1e9:.1f}B" if cap else "?"
            lines.append(f"  {sym}: ${price:,.2f} ({sign}{chg:.2f}% 24h) mcap ${cap_b}")
        result = "Live market data (CoinGecko, UTC " + datetime.datetime.utcnow().strftime("%H:%M") + "):\n" + "\n".join(lines)
        _market_cache["data"] = result
        _market_cache["fetched_at"] = now
        return result
    except Exception as e:
        return f"Market data unavailable ({e})"


def _groq_chat(prompt: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    today = datetime.datetime.utcnow().strftime("%B %d, %Y")
    market = _fetch_market_data()
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are GodLocal, an autonomous AI trading agent.\n"
                    f"Today's date: {today} (UTC).\n"
                    f"{market}\n"
                    "Use this live data when answering. "
                    "Analyze whale movements and market signals. "
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
            "User-Agent": "groq-python/0.21.0",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


class handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, body: dict, status: int = 200):
        enc = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
        elif p == "/market":
            # Expose raw market data endpoint
            self._send_json({"market": _fetch_market_data(), "ts": int(time.time())})
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
                    err_body = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                self._send_json({"error": f"Groq HTTP {e.code}", "detail": err_body}, 500)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)
