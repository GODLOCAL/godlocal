"""
GodLocal Edge API — Vercel serverless (standalone, no heavy deps)
Endpoints: /health /status /think /mobile/status /mobile/kill-switch
"""
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GodLocal Edge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state (resets on cold start — VPS handles persistence)
_kill_switch: bool = os.environ.get("XZERO_KILL_SWITCH", "false").lower() == "true"
_thoughts: list = []
_sparks: list = []

_groq_client: Any = None
GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq():
    global _groq_client
    if _groq_client is None:
        key = os.environ.get("GROQ_API_KEY")
        if key:
            try:
                from groq import Groq  # lazy import — avoid top-level crash
                _groq_client = Groq(api_key=key)
            except Exception:
                pass
    return _groq_client


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "env": os.environ.get("GODLOCAL_ENV", "production"),
        "ts": int(time.time()),
    }


@app.get("/status")
async def status():
    return {
        "kill_switch_active": _kill_switch,
        "circuit_breaker": {
            "is_tripped": _kill_switch,
            "consecutive_losses": 0,
            "daily_loss_sol": 0.0,
        },
        "sparks": _sparks[-10:],
        "thoughts": _thoughts[-5:],
        "ts": int(time.time()),
    }


@app.get("/mobile/status")
async def mobile_status():
    return {
        "kill_switch_active": _kill_switch,
        "circuit_breaker": {
            "is_tripped": _kill_switch,
            "consecutive_losses": 0,
            "daily_loss_sol": 0.0,
        },
        "sparks": _sparks[-10:],
        "thoughts": _thoughts[-5:],
        "ts": int(time.time()),
    }


@app.post("/mobile/kill-switch")
async def set_kill_switch(request: Request):
    global _kill_switch
    body = await request.json()
    _kill_switch = bool(body.get("active", False))
    return {"kill_switch_active": _kill_switch, "ok": True}


@app.post("/think")
async def think(request: Request):
    global _thoughts
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)

    client = get_groq()
    if not client:
        return JSONResponse({"error": "GROQ_API_KEY not configured"}, status_code=503)

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are GodLocal, an autonomous AI trading agent running on Vercel edge. "
                        "You analyze on-chain data, whale movements, and market signals. "
                        "Be concise and direct. Answer in the same language as the user."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        response_text = completion.choices[0].message.content
        thought = {
            "thought": response_text,
            "prompt": prompt,
            "timestamp": int(time.time() * 1000),
        }
        _thoughts.append(thought)
        _thoughts = _thoughts[-20:]
        return {"response": response_text, "model": GROQ_MODEL}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Vercel ASGI handler
handler = app
