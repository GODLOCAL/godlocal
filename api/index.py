"""
GodLocal Agent API — zero-dependency Vercel Lambda

Agent architecture:
  /think       — ReAct loop with tool calling + 429 fallback chain
  /agent/tick  — Autonomous self-triggered analysis cycle
  /market      — Live prices (CoinGecko, 5-min cache)
  /status /mobile/status — xzero circuit breaker state
  /mobile/kill-switch    — Toggle kill switch

Model fallback chain (429 protection):
  llama-3.3-70b-versatile → llama-3.1-8b-instant → llama3-8b-8192
"""
import json
import os
import time
import datetime
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler

# ── Model config ────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Fallback chain: best → fast → emergency
MODEL_CHAIN = [
    "llama-3.3-70b-versatile",   # primary — best reasoning
    "llama-3.1-8b-instant",      # fallback — 429 / rate limited
    "llama3-8b-8192",            # emergency fallback
]
MODEL_FAST = "llama-3.1-8b-instant"

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum,solana,binancecoin,sui"
    "&vs_currencies=usd"
    "&include_24hr_change=true"
    "&include_market_cap=true"
)

# ── State ────────────────────────────────────────────────────────────────────
_kill_switch: bool = os.environ.get("XZERO_KILL_SWITCH", "false").lower() == "true"
_thoughts: list = []
_sparks:   list = []
_market_cache: dict = {"data": None, "raw": None, "fetched_at": 0}
_MARKET_TTL = 300

# ── Agent tools ──────────────────────────────────────────────────────────────
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Get live cryptocurrency prices and 24h changes (BTC, ETH, SOL, BNB, SUI)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Get xzero trading system status: kill switch, circuit breaker, consecutive losses, daily PnL",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_thoughts",
            "description": "Get the agent's recent analysis thoughts (last 5)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_kill_switch",
            "description": "Enable or disable the xzero trading kill switch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "active": {"type": "boolean", "description": "True to halt trading"},
                    "reason": {"type": "string", "description": "Reason for toggle"},
                },
                "required": ["active", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_spark",
            "description": "Log a trading signal to SparkNet",
            "parameters": {
                "type": "object",
                "properties": {
                    "signal":     {"type": "string"},
                    "confidence": {"type": "number"},
                    "action":     {"type": "string", "enum": ["BUY", "SELL", "HOLD", "WATCH"]},
                },
                "required": ["signal", "confidence", "action"],
            },
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────
def _execute_tool(name: str, args: dict) -> str:
    global _kill_switch, _sparks
    if name == "get_market_data":
        return _fetch_market_text()
    elif name == "get_system_status":
        return json.dumps({
            "kill_switch_active": _kill_switch,
            "circuit_breaker": {"is_tripped": _kill_switch, "consecutive_losses": 0, "daily_loss_sol": 0.0},
            "uptime_ts": int(time.time()),
        }, ensure_ascii=False)
    elif name == "get_recent_thoughts":
        recent = _thoughts[-5:]
        return json.dumps(recent, ensure_ascii=False) if recent else "No thoughts yet."
    elif name == "set_kill_switch":
        _kill_switch = bool(args.get("active", False))
        return json.dumps({"ok": True, "kill_switch_active": _kill_switch, "reason": args.get("reason", "")})
    elif name == "add_spark":
        spark = {
            "signal":     args.get("signal", ""),
            "confidence": float(args.get("confidence", 0.5)),
            "action":     args.get("action", "WATCH"),
            "timestamp":  int(time.time() * 1000),
        }
        _sparks.append(spark)
        _sparks = _sparks[-50:]
        return json.dumps({"ok": True, "spark": spark})
    return json.dumps({"error": f"unknown tool: {name}"})


# ── Market data ───────────────────────────────────────────────────────────────
def _fetch_market_raw() -> dict:
    now = time.time()
    if _market_cache["raw"] and (now - _market_cache["fetched_at"]) < _MARKET_TTL:
        return _market_cache["raw"]
    try:
        req = urllib.request.Request(
            COINGECKO_URL,
            headers={"User-Agent": "GodLocal/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = json.loads(resp.read())
        _market_cache["raw"] = raw
        _market_cache["fetched_at"] = now
        _market_cache["data"] = None
        return raw
    except Exception:
        return {}


def _fetch_market_text() -> str:
    now = time.time()
    if _market_cache["data"] and (now - _market_cache["fetched_at"]) < _MARKET_TTL:
        return _market_cache["data"]
    raw = _fetch_market_raw()
    names = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "binancecoin": "BNB", "sui": "SUI"}
    lines = []
    for coin_id, sym in names.items():
        d     = raw.get(coin_id, {})
        price = d.get("usd", "?")
        chg   = d.get("usd_24h_change", 0) or 0
        cap   = d.get("usd_market_cap", 0) or 0
        sign  = "+" if chg >= 0 else ""
        cap_b = f"{cap/1e9:.1f}B" if cap else "?"
        lines.append(f"  {sym}: ${price:,.2f} ({sign}{chg:.2f}% 24h) mcap ${cap_b}")
    ts     = datetime.datetime.utcnow().strftime("%H:%M UTC")
    result = f"Live prices [{ts}]:\n" + "\n".join(lines)
    _market_cache["data"] = result
    return result


# ── Groq request with 429 fallback chain ──────────────────────────────────────
def _groq_request(messages: list, tools=None, model_override: str | None = None) -> tuple[dict, str]:
    """
    Returns (response_dict, model_used).
    Tries MODEL_CHAIN in order; on 429 moves to next model.
    """
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")

    chain = [model_override] if model_override else MODEL_CHAIN

    for model in chain:
        body = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  512,   # keep well under 12k TPM
            "temperature": 0.6,
        }
        if tools:
            body["tools"]       = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
                "User-Agent":     "groq-python/0.21.0",
                "Accept":         "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read()), model
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — try next model in chain
                continue
            # Other HTTP error — re-raise
            raise

    raise RuntimeError(f"All models rate-limited. Retry in a moment.")


# ── ReAct agent loop ──────────────────────────────────────────────────────────
def _agent_loop(user_prompt: str, history: list | None = None, max_steps: int = 5) -> dict:
    today  = datetime.datetime.utcnow().strftime("%B %d, %Y")
    market = _fetch_market_text()
    system = (
        f"You are GodLocal, an autonomous AI trading agent with full control over the xzero system.\n"
        f"Today: {today} (UTC).\n"
        f"{market}\n\n"
        "Tools available: get_market_data, get_system_status, get_recent_thoughts, set_kill_switch, add_spark.\n"
        "Rules: think step-by-step; use tools to gather facts; log non-trivial signals as sparks; "
        "if extreme volatility detected proactively trigger kill switch; be concise; answer in user's language."
    )
    messages = [{"role": "system", "content": system}]
    # Trim history to last 6 turns to stay under TPM
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_prompt})

    steps_taken = []
    model_used  = MODEL_CHAIN[0]

    for step in range(max_steps):
        resp, model_used = _groq_request(messages, tools=AGENT_TOOLS)
        choice  = resp["choices"][0]
        message = choice["message"]
        finish  = choice["finish_reason"]

        messages.append(message)

        if finish == "tool_calls" and message.get("tool_calls"):
            for tc in message["tool_calls"]:
                fn     = tc["function"]["name"]
                args   = json.loads(tc["function"]["arguments"] or "{}")
                result = _execute_tool(fn, args)
                steps_taken.append({"tool": fn, "args": args, "result_preview": result[:100]})
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result,
                })
        else:
            text = message.get("content", "")
            return {"response": text, "steps": steps_taken, "model": model_used}

    return {"response": "Agent reached max steps.", "steps": steps_taken, "model": model_used}


# ── HTTP handler ──────────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def _send_json(self, body: dict, status: int = 200):
        enc = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",  "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(enc)

    def do_OPTIONS(self): self._send_json({}, 200)

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/") or "/"
        if p in ("/", "/health"):
            self._send_json({"status": "ok", "env": "production", "ts": int(time.time()),
                             "models": MODEL_CHAIN})
        elif p in ("/status", "/mobile/status"):
            self._send_json({
                "kill_switch_active": _kill_switch,
                "circuit_breaker":    {"is_tripped": _kill_switch, "consecutive_losses": 0, "daily_loss_sol": 0.0},
                "sparks":             _sparks[-10:],
                "thoughts":           _thoughts[-5:],
                "ts":                 int(time.time()),
            })
        elif p == "/market":
            self._send_json({"market": _fetch_market_text(), "raw": _fetch_market_raw(), "ts": int(time.time())})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        global _kill_switch, _thoughts
        p      = self.path.split("?")[0].rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length) or b"{}")

        if p == "/mobile/kill-switch":
            _kill_switch = bool(body.get("active", False))
            self._send_json({"kill_switch_active": _kill_switch, "ok": True})

        elif p == "/think":
            prompt = body.get("prompt", "").strip()
            if not prompt:
                self._send_json({"error": "prompt required"}, 400)
                return
            if not os.environ.get("GROQ_API_KEY"):
                self._send_json({"error": "GROQ_API_KEY not configured"}, 503)
                return
            try:
                history = body.get("history", [])
                result  = _agent_loop(prompt, history=history)
                thought = {
                    "thought":   result["response"],
                    "prompt":    prompt,
                    "steps":     result.get("steps", []),
                    "model":     result.get("model"),
                    "timestamp": int(time.time() * 1000),
                }
                _thoughts.append(thought)
                _thoughts = _thoughts[-20:]
                self._send_json(result)
            except RuntimeError as e:
                self._send_json({"error": str(e), "retry": True}, 429)
            except urllib.error.HTTPError as e:
                detail = ""
                try: detail = e.read().decode("utf-8", errors="replace")[:300]
                except Exception: pass
                self._send_json({"error": f"Groq HTTP {e.code}", "detail": detail}, 500)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif p == "/agent/tick":
            if not os.environ.get("GROQ_API_KEY"):
                self._send_json({"error": "GROQ_API_KEY not configured"}, 503)
                return
            try:
                result = _agent_loop(
                    "Autonomous market analysis: check prices, assess risk, log notable signals to SparkNet, "
                    "decide kill switch state. Be concise."
                )
                thought = {
                    "thought":   result["response"],
                    "prompt":    "[auto-tick]",
                    "steps":     result.get("steps", []),
                    "model":     result.get("model"),
                    "timestamp": int(time.time() * 1000),
                }
                _thoughts.append(thought)
                _thoughts = _thoughts[-20:]
                self._send_json({"ok": True, "tick": result})
            except RuntimeError as e:
                self._send_json({"error": str(e), "retry": True}, 429)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "not found"}, 404)
