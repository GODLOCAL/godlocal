"""
extensions/connectors/agents/sonic_gateway_connector.py — GodLocal SonicGatewayConnector
Implements the Sonic pattern: OpenAI-style WebSocket Responses API over local LLM.

Sonic (by @iotcoi) core ideas applied to GodLocal:
  • wss://localhost:9000/v1/responses  — OpenAI Responses API-compatible endpoint
  • Stateful threads                   — conversation memory across turns
  • Mid-stream cancellation            — stop generation in-flight
  • Strict JSON tool contracts         — no "Certainly! I'd be happy to…" noise
  • Multi-step agent flow              — ask → think → tool → continue
  • 50% speed gain via WebSocket vs polling

GodLocal integration points:
  1. X-ZERO agent runtime  → replaces REST polling in xzero_delegation.py
  2. AutoGenesis evolve()  → streams patch candidates in real-time
  3. Telegram bot bridge   → streams tokens directly to chat
  4. AgentPool agents      → all share one WebSocket gateway

Route: POST /connectors/agent/sonic_gateway/run
Auto-discovered by ConnectorsModule.

Usage (server mode — run alongside godlocal_v5.py):
  python -m extensions.connectors.agents.sonic_gateway_connector --serve

Usage (client mode — from code):
  from extensions.connectors.agents.sonic_gateway_connector import SonicClient
  async with SonicClient() as client:
      async for token in client.stream("Plan a Jupiter swap for $50"):
          print(token, end="", flush=True)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import AsyncIterator, Callable

logger = logging.getLogger("godlocal.sonic")

# ── Config ────────────────────────────────────────────────────────────────────
SONIC_HOST      = os.getenv("SONIC_HOST",  "localhost")
SONIC_PORT      = int(os.getenv("SONIC_PORT", "9000"))
SONIC_WS_URL    = f"ws://{SONIC_HOST}:{SONIC_PORT}/v1/responses"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("SONIC_MODEL", "qwen3:8b")   # ParoQuant default

# ── Tool contract enforcement ─────────────────────────────────────────────────
TOOL_CONTRACT_SYSTEM = """
TOOL CALLING CONTRACT (MANDATORY):
When you need to call a tool, output ONLY valid JSON — nothing else.
Format: {"tool": "<name>", "args": {<key>: <value>}}

DO NOT output any prose before or after the JSON.
DO NOT say "Certainly!", "I'll help you", or any preamble.
If you do not need a tool, respond normally.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# ── WebSocket Server (Sonic gateway) ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class SonicServer:
    """
    Lightweight WebSocket gateway: OpenAI Responses API → Ollama streaming.

    Message protocol (client → server):
      {"type": "create", "thread_id": "...", "model": "...", "input": "...", "tools": [...]}
      {"type": "cancel", "thread_id": "..."}

    Message protocol (server → client):
      {"type": "token",    "thread_id": "...", "delta": "..."}
      {"type": "done",     "thread_id": "...", "output": "..."}
      {"type": "tool_call","thread_id": "...", "call": {...}}
      {"type": "error",    "thread_id": "...", "message": "..."}
    """

    def __init__(self, host: str = SONIC_HOST, port: int = SONIC_PORT):
        self.host    = host
        self.port    = port
        # thread_id → {"messages": [...], "cancel": asyncio.Event}
        self._threads: dict[str, dict] = {}

    async def start(self):
        if not WS_AVAILABLE:
            raise RuntimeError("websockets not installed: pip install websockets")
        logger.info(f"[Sonic] 🎵 Gateway starting on ws://{self.host}:{self.port}/v1/responses")
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()  # run forever

    async def _handler(self, ws):
        """Handle one WebSocket connection (may send multiple requests on same socket)."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "message": "invalid JSON"}))
                continue

            msg_type  = msg.get("type", "create")
            thread_id = msg.get("thread_id") or str(uuid.uuid4())

            if msg_type == "cancel":
                self._cancel_thread(thread_id)
                continue

            if msg_type == "create":
                asyncio.create_task(self._run_inference(ws, msg, thread_id))

    def _cancel_thread(self, thread_id: str):
        if thread_id in self._threads:
            self._threads[thread_id]["cancel"].set()

    async def _run_inference(self, ws, msg: dict, thread_id: str):
        """Stream Ollama response back over WebSocket."""
        if not HTTPX_AVAILABLE:
            await ws.send(json.dumps({"type": "error", "thread_id": thread_id,
                                      "message": "httpx not installed"}))
            return

        # ── Build / resume thread ─────────────────────────────────────────────
        if thread_id not in self._threads:
            self._threads[thread_id] = {
                "messages": [{"role": "system",
                               "content": TOOL_CONTRACT_SYSTEM}],
                "cancel": asyncio.Event(),
            }
        thread = self._threads[thread_id]
        cancel_event: asyncio.Event = thread["cancel"]
        cancel_event.clear()

        # Append tools to system if provided
        tools = msg.get("tools", [])
        if tools:
            tool_json = json.dumps(tools, indent=2)
            thread["messages"][0]["content"] = (
                TOOL_CONTRACT_SYSTEM + f"\n\nAvailable tools:\n{tool_json}"
            )

        thread["messages"].append({"role": "user", "content": msg.get("input", "")})

        model    = msg.get("model", DEFAULT_MODEL)
        full_out = []

        # ── Stream from Ollama ────────────────────────────────────────────────
        ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model":    model,
            "messages": thread["messages"],
            "stream":   True,
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", ollama_url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if cancel_event.is_set():
                            await ws.send(json.dumps({"type": "cancelled",
                                                       "thread_id": thread_id}))
                            return
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            full_out.append(delta)
                            await ws.send(json.dumps({
                                "type":      "token",
                                "thread_id": thread_id,
                                "delta":     delta,
                            }))
                        if chunk.get("done"):
                            break
        except Exception as e:
            await ws.send(json.dumps({"type": "error", "thread_id": thread_id,
                                      "message": str(e)}))
            return

        output = "".join(full_out)

        # ── Detect tool call (strict JSON contract) ───────────────────────────
        stripped = output.strip()
        tool_call = None
        if stripped.startswith("{") and '"tool"' in stripped:
            try:
                tool_call = json.loads(stripped)
            except json.JSONDecodeError:
                pass

        if tool_call:
            thread["messages"].append({"role": "assistant", "content": output})
            await ws.send(json.dumps({
                "type":      "tool_call",
                "thread_id": thread_id,
                "call":      tool_call,
            }))
        else:
            thread["messages"].append({"role": "assistant", "content": output})
            await ws.send(json.dumps({
                "type":      "done",
                "thread_id": thread_id,
                "output":    output,
            }))


# ═══════════════════════════════════════════════════════════════════════════════
# ── Client ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class SonicClient:
    """
    Async WebSocket client for SonicServer.

    Usage:
        async with SonicClient() as client:
            # simple streaming
            async for token in client.stream("What is Jupiter DEX?"):
                print(token, end="", flush=True)

            # agent loop with tool execution
            result = await client.agent_loop(
                prompt="Swap 0.1 SOL for USDC via Jupiter",
                tools=[{"name": "jupiter_swap", "description": "...", "parameters": {...}}],
                tool_executor=my_tool_fn,
            )
    """

    def __init__(self, url: str = SONIC_WS_URL):
        self.url       = url
        self._ws       = None
        self._thread_id = str(uuid.uuid4())

    async def __aenter__(self):
        if not WS_AVAILABLE:
            raise RuntimeError("websockets not installed: pip install websockets")
        import websockets as _ws
        self._ws = await _ws.connect(self.url)
        return self

    async def __aexit__(self, *_):
        if self._ws:
            await self._ws.close()

    async def stream(
        self,
        prompt: str,
        tools: list | None = None,
        thread_id: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> AsyncIterator[str]:
        """Yield tokens as they stream from the gateway."""
        tid = thread_id or self._thread_id
        msg = {"type": "create", "thread_id": tid, "input": prompt, "model": model}
        if tools:
            msg["tools"] = tools
        await self._ws.send(json.dumps(msg))
        async for raw in self._ws:
            data = json.loads(raw)
            t = data.get("type")
            if t == "token":
                yield data["delta"]
            elif t in ("done", "tool_call", "cancelled", "error"):
                break

    async def ask(self, prompt: str, **kwargs) -> str:
        """Collect full response (no streaming)."""
        return "".join([tok async for tok in self.stream(prompt, **kwargs)])

    async def cancel(self, thread_id: str | None = None):
        tid = thread_id or self._thread_id
        await self._ws.send(json.dumps({"type": "cancel", "thread_id": tid}))

    async def agent_loop(
        self,
        prompt: str,
        tools: list,
        tool_executor: Callable[[dict], str],
        max_steps: int = 6,
        model: str = DEFAULT_MODEL,
    ) -> str:
        """
        Multi-step agent: ask → think → tool → continue.
        Strict JSON tool contract enforced by SonicServer.
        """
        thread_id = str(uuid.uuid4())
        current_prompt = prompt

        for step in range(max_steps):
            tid = thread_id
            msg = {
                "type": "create", "thread_id": tid,
                "input": current_prompt, "model": model,
                "tools": tools,
            }
            await self._ws.send(json.dumps(msg))

            full_output = ""
            tool_call   = None

            async for raw in self._ws:
                data = json.loads(raw)
                t = data.get("type")
                if t == "token":
                    full_output += data["delta"]
                elif t == "tool_call":
                    tool_call = data["call"]
                    break
                elif t in ("done", "cancelled", "error"):
                    full_output = data.get("output", full_output)
                    break

            if tool_call:
                try:
                    tool_result = await asyncio.to_thread(tool_executor, tool_call)
                except Exception as e:
                    tool_result = f"Tool error: {e}"
                # Feed result back as next prompt
                current_prompt = (
                    f"Tool '{tool_call.get('tool')}' returned:\n{tool_result}\n\n"
                    "Continue with the task."
                )
                logger.debug(f"[Sonic] step {step+1}: tool={tool_call.get('tool')} result={tool_result[:80]}")
            else:
                return full_output  # natural end

        return full_output  # max_steps reached


# ═══════════════════════════════════════════════════════════════════════════════
# ── ConnectorsModule Agent Wrapper ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from extensions.connectors.connectors_module import BaseConnectorAgent
except ImportError:
    class BaseConnectorAgent:
        name = ""; description = ""
        def __init__(self, llm=None): self.llm = llm


class SonicGatewayConnector(BaseConnectorAgent):
    """
    Auto-discovered by ConnectorsModule.
    Route: POST /connectors/agent/sonic_gateway/run

    Actions:
      stream   — stream a prompt, return full output
      ask      — alias for stream (blocking)
      status   — check if Sonic server is reachable
    """

    name        = "sonic_gateway"
    description = "OpenAI Responses WebSocket gateway for local LLM (Ollama + ParoQuant)"

    async def run(self, action: str = "ask", **kwargs) -> dict:
        dispatch = {
            "stream": self._ask,
            "ask":    self._ask,
            "status": self._status,
        }
        handler = dispatch.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}"}
        try:
            return await handler(**kwargs)
        except Exception as e:
            return {"error": str(e), "action": action,
                    "tip": f"Is Sonic server running? python -m extensions.connectors.agents.sonic_gateway_connector --serve"}

    async def _ask(self, prompt: str = "", model: str = DEFAULT_MODEL, **kwargs) -> dict:
        async with SonicClient(SONIC_WS_URL) as client:
            output = await client.ask(prompt, model=model)
        return {"status": "ok", "output": output, "model": model}

    async def _status(self, **kwargs) -> dict:
        try:
            import websockets as _ws
            async with _ws.connect(SONIC_WS_URL, open_timeout=3):
                return {"status": "online", "url": SONIC_WS_URL}
        except Exception as e:
            return {"status": "offline", "url": SONIC_WS_URL, "error": str(e),
                    "tip": "Run: python -m extensions.connectors.agents.sonic_gateway_connector --serve"}


# ═══════════════════════════════════════════════════════════════════════════════
# ── __main__ — start the gateway server ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    verbose  = "--verbose" in sys.argv or "-v" in sys.argv
    dry_run  = "--dry-run" in sys.argv
    serve    = "--serve"   in sys.argv
    for arg in sys.argv:
        if arg.startswith("--model="):
            os.environ["SONIC_MODEL"] = arg.split("=", 1)[1]

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s" if verbose
               else "%(asctime)s %(message)s",
    )

    async def _preflight() -> bool:
        ok = True
        # Check packages
        for pkg in ("websockets", "httpx"):
            try:
                __import__(pkg)
                if verbose:
                    print(f"  ✅ {pkg}")
            except ImportError:
                print(f"  ❌ Missing: pip install {pkg}")
                ok = False
        if not ok:
            return False
        # Check Ollama
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=5) as hc:
                r = await hc.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if verbose:
                    print(f"  ✅ Ollama — {len(models)} models")
                if DEFAULT_MODEL not in models:
                    print(f"  ⚠️  Model \'{DEFAULT_MODEL}\' not found.")
                    print(f"     Available: {models[:5]}")
                    print(f"     Fix: ollama pull {DEFAULT_MODEL}")
            else:
                print(f"  ⚠️  Ollama HTTP {r.status_code} — is it running?")
        except Exception as e:
            print(f"  ❌ Ollama unreachable at {OLLAMA_BASE_URL}")
            print(f"     Fix: ollama serve && ollama pull {DEFAULT_MODEL}")
            print(f"     ({e})")
            ok = False
        return ok

    if serve:
        async def _serve():
            print("=== Sonic preflight ===")
            ok = await _preflight()
            if not ok:
                print("\n🔴 Fix issues above, then retry --serve")
                return
            if dry_run:
                print(f"\n🔵 [dry-run] Would bind ws://{SONIC_HOST}:{SONIC_PORT}/v1/responses")
                print(f"   Backend={OLLAMA_BASE_URL}  Model={DEFAULT_MODEL}")
                return
            server = SonicServer()
            print(f"\n🎵 Sonic gateway → ws://{SONIC_HOST}:{SONIC_PORT}/v1/responses")
            print(f"   Backend: {OLLAMA_BASE_URL}  Model: {DEFAULT_MODEL}  Verbose: {verbose}")
            await server.start()
        asyncio.run(_serve())
    else:
        # No --serve: preflight + optional smoke-test
        async def _test():
            print("=== Sonic preflight ===")
            ok = await _preflight()
            if not ok:
                print("\n🔴 Fix above, then: python -m ... --serve")
                return
            if dry_run:
                print("\n🔵 [dry-run] Preflight OK — not connecting (server not running in dry-run).")
                return
            print(f"\n→ Smoke-test: {SONIC_WS_URL}")
            try:
                async with SonicClient() as c:
                    async for tok in c.stream("What is 2+2? One word."):
                        print(tok, end="", flush=True)
                print("\n✅ Smoke-test passed")
            except Exception as e:
                print(f"\n❌ {e}")
                print("   Tip: start server first with --serve in a separate terminal.")
        asyncio.run(_test())
