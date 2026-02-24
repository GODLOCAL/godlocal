"""
core/agent_pool.py — Multi-agent RAM Hot-Swap for GodLocal v5.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One model in RAM at a time — but hot-swappable on demand.
Each specialist agent config maps to a different MLX model.

API usage (FastAPI endpoint in godlocal_v5.py):
    POST /agent/swap/coding   → loads DeepSeek-Coder
    POST /agent/swap/trading  → loads Qwen2.5-72B (low temp, precise)
    POST /agent/swap/default  → back to baseline Qwen3-4B-PARO

Developer Pro tier: exposes all agent types via API key.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Agent catalogue ───────────────────────────────────────────────────────────
# model: HuggingFace / MLX model identifier
# temp:  sampling temperature (lower = more deterministic)
# desc:  shown in /agent/status
AGENT_CONFIGS: Dict[str, dict] = {
    "default": {
        "model": "z-lab/Qwen3-4B-PARO",
        "temp": 0.8,
        "desc": "Baseline sovereign AI — daily use",
    },
    "coding": {
        "model": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
        "temp": 0.7,
        "desc": "Code generation & debugging — AutoGenesis tasks",
    },
    "trading": {
        "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "temp": 0.3,
        "desc": "Precise, low-hallucination — X-ZERO signal analysis",
    },
    "writing": {
        "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "temp": 0.92,
        "desc": "Creative writing — book, soul templates, marketing",
    },
    "medical": {
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "temp": 0.5,
        "desc": "MRIAnalyzer — HIPAA zero-egress medical queries",
    },
    "sleep": {
        "model": "z-lab/Qwen3-8B-PARO",
        "temp": 0.6,
        "desc": "sleep_cycle() — deeper reasoning for nightly consolidation",
    },
}


class AgentPool:
    """
    Manages a pool of MLX model agents with async hot-swap.

    One model lives in RAM at a time. swap() unloads the current model
    and loads the requested one (lazy — first call triggers load).

    Thread-safe via asyncio.Lock.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple] = {}   # agent_type → (model, tokenizer, temp)
        self.current: str = "default"
        self.lock = asyncio.Lock()
        self._mlx_available: Optional[bool] = None
        self._swap_history: list[dict] = []

    def _check_mlx(self) -> bool:
        if self._mlx_available is None:
            try:
                from mlx_lm import load  # noqa
                self._mlx_available = True
            except ImportError:
                self._mlx_available = False
                logger.warning("[AgentPool] mlx_lm not available — pool disabled")
        return self._mlx_available

    async def get(self, agent_type: str = "default") -> Optional[Tuple]:
        """
        Return (model, tokenizer, temperature) for agent_type.
        Loads from disk on first call; subsequent calls hit cache.
        """
        if not self._check_mlx():
            return None

        cfg = AGENT_CONFIGS.get(agent_type, AGENT_CONFIGS["default"])

        async with self.lock:
            if agent_type not in self._cache:
                t0 = time.time()
                logger.info(f"[AgentPool] Loading {agent_type} → {cfg['model']} ...")
                try:
                    from mlx_lm import load
                    model, tokenizer = load(cfg["model"])
                    self._cache[agent_type] = (model, tokenizer, cfg["temp"])
                    elapsed = round(time.time() - t0, 1)
                    logger.info(f"[AgentPool] {agent_type} loaded in {elapsed}s ✓")
                except Exception as e:
                    logger.error(f"[AgentPool] Failed to load {agent_type}: {e}")
                    return None

            self.current = agent_type
            self._swap_history.append({
                "agent": agent_type, "at": time.time()
            })
            return self._cache[agent_type]

    async def swap(self, agent_type: str) -> dict:
        """
        Hot-swap to agent_type. Returns status dict for API response.
        Evicts all other cached models to free RAM (one model at a time).
        """
        if agent_type not in AGENT_CONFIGS:
            return {"error": f"Unknown agent type: {agent_type}. Available: {list(AGENT_CONFIGS)}"}

        async with self.lock:
            # Evict other cached models to save RAM
            evicted = [k for k in list(self._cache) if k != agent_type]
            for k in evicted:
                del self._cache[k]
                logger.info(f"[AgentPool] Evicted {k} from RAM")

        # Load the requested agent (releases lock during load)
        result = await self.get(agent_type)
        if result is None:
            return {"error": f"Failed to load {agent_type}"}

        cfg = AGENT_CONFIGS[agent_type]
        return {
            "status": f"hot-swapped to {agent_type}",
            "model": cfg["model"],
            "temp": cfg["temp"],
            "desc": cfg["desc"],
            "evicted": evicted,
        }

    def status(self) -> dict:
        """Return current pool state for /agent/status."""
        return {
            "current": self.current,
            "cached": list(self._cache.keys()),
            "available": list(AGENT_CONFIGS.keys()),
            "configs": {k: {"model": v["model"], "temp": v["temp"], "desc": v["desc"]}
                        for k, v in AGENT_CONFIGS.items()},
            "swap_count": len(self._swap_history),
        }


# Module-level singleton (imported by godlocal_v5.py)
agent_pool = AgentPool()
