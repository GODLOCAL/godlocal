"""models/schemas.py — Pydantic v2 schemas for БОГ || OASIS v6 API"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────
class ThinkRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=8000, description="Prompt / task for the AI")
    max_tokens: int = Field(2048, ge=64, le=8192)
    long_memory: bool = False


class EvolveRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    apply: bool = False
    max_revisions: int = Field(2, ge=1, le=5)


class SwapAgentRequest(BaseModel):
    agent_type: str


class MemoryAddRequest(BaseModel):
    text: str = Field(..., min_length=1)
    long: bool = False


# ── Responses ─────────────────────────────────────────────────────────────
class ThinkResponse(BaseModel):
    response: str
    model: str
    tokens_approx: int


class EvolveResponse(BaseModel):
    status: str
    task: str
    plan: dict[str, Any]
    files: list[dict[str, Any]]
    elapsed_sec: float
    fep: dict[str, Any]


class StatusResponse(BaseModel):
    version: str
    model: str
    soul_loaded: bool
    memory: dict[str, Any]
    fep: dict[str, Any]
    agents: dict[str, Any]
    uptime_sec: float


class AgentSwapResponse(BaseModel):
    agent: str
    model: str
    swaps: int
