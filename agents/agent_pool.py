"""agents/agent_pool.py — RAM-efficient hot-swap AgentPool for БОГ || OASIS v6
One model loaded at a time. Swap evicts previous model from RAM.
POST /agent/swap/{type}  GET /agent/status
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.settings import settings

logger = logging.getLogger(__name__)

AGENT_CONFIGS: dict[str, str] = {
    "coding":    "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
    "trading":   "mlx-community/Qwen2.5-72B-Instruct-4bit",
    "writing":   "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "research":  "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "ocr":       "mlx-community/llava-1.5-7b-4bit",
    "medical":   "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "default":   settings.model,
}


@dataclass
class AgentSlot:
    name: str
    model_id: str
    model: Any = field(default=None, repr=False)
    loaded_at: float = 0.0
    swap_count: int = 0


class AgentPool:
    """Single-active-model pool with graceful MLX fallback."""

    def __init__(self) -> None:
        self._slots: dict[str, AgentSlot] = {
            k: AgentSlot(name=k, model_id=v)
            for k, v in AGENT_CONFIGS.items()
        }
        self._active: str | None = None
        self._lock = asyncio.Lock()
        self._mlx_available = self._check_mlx()

    def _check_mlx(self) -> bool:
        try:
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            logger.warning("mlx_lm not installed — AgentPool running in stub mode")
            return False

    async def swap(self, agent_type: str) -> dict:
        """Hot-swap to agent_type. Evicts current model from RAM first."""
        if agent_type not in self._slots:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self._slots)}")

        async with self._lock:
            # Evict current
            if self._active and self._active != agent_type:
                slot = self._slots[self._active]
                slot.model = None
                logger.info("AgentPool: evicted %s", self._active)

            target = self._slots[agent_type]
            if target.model is None and self._mlx_available:
                logger.info("AgentPool: loading %s (%s)…", agent_type, target.model_id)
                t0 = time.time()
                from mlx_lm import load  # type: ignore
                target.model, _ = load(target.model_id)
                target.loaded_at = time.time()
                target.swap_count += 1
                elapsed = time.time() - t0
                logger.info("AgentPool: %s ready in %.1fs", agent_type, elapsed)
            elif not self._mlx_available:
                # Stub: just mark active (Ollama will serve)
                target.loaded_at = time.time()
                target.swap_count += 1

            self._active = agent_type
            return {"agent": agent_type, "model": target.model_id, "swaps": target.swap_count}

    def status(self) -> dict:
        return {
            "active": self._active,
            "mlx_available": self._mlx_available,
            "slots": [
                {
                    "name": s.name,
                    "model": s.model_id,
                    "loaded": s.model is not None,
                    "swaps": s.swap_count,
                }
                for s in self._slots.values()
            ],
        }


# Module-level singleton
agent_pool = AgentPool()
