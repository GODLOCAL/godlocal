"""api/index.py — GodLocal Vercel Edge API

Lightweight FastAPI app for Vercel serverless deployment.
Provides: /health, /status, /think (via Groq)
Full stack (local models, ChromaDB, agents) runs on VPS via godlocal_v6.py
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="GodLocal API",
    description="Your AI. On your machine. Getting smarter while you sleep.",
    version="6.0.0-edge",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START = time.time()
_GROQ_KEY = os.getenv("GROQ_API_KEY", "")


class ThinkRequest(BaseModel):
    prompt: str
    model: str = "llama-3.1-8b-instant"
    max_tokens: int = 512


class ThinkResponse(BaseModel):
    text: str
    model: str
    tok_s: float | None = None
    elapsed_s: float


@app.get("/")
def root():
    return {
        "name": "GodLocal",
        "version": "6.0.0-edge",
        "tagline": "Your AI. On your machine. Getting smarter while you sleep.",
        "docs": "/docs",
        "status": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _START, 1),
        "env": os.getenv("GODLOCAL_ENV", "unknown"),
        "groq_configured": bool(_GROQ_KEY),
    }


@app.get("/status")
def status():
    return {
        "status": "ok",
        "version": "6.0.0-edge",
        "uptime_s": round(time.time() - _START, 1),
        "backends": {
            "groq": {"available": bool(_GROQ_KEY), "tier": "cloud-fast"},
            "local": {"available": False, "note": "Run godlocal_v6.py on VPS for local models"},
        },
        "tiers": ["FAST (Groq)"],
    }


@app.post("/think", response_model=ThinkResponse)
async def think(req: ThinkRequest):
    if not _GROQ_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    try:
        from groq import AsyncGroq
        import asyncio

        client = AsyncGroq(api_key=_GROQ_KEY)
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": req.prompt}],
            max_completion_tokens=req.max_tokens,
        )
        elapsed = time.perf_counter() - t0

        usage = getattr(resp, "usage", None)
        comp_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        tok_s = round(comp_tok / elapsed, 1) if elapsed > 0 else None

        return ThinkResponse(
            text=resp.choices[0].message.content,
            model=resp.model,
            tok_s=tok_s,
            elapsed_s=round(elapsed, 3),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
