"""
agents/xzero/heartbeat.py
XZero Heartbeat — background tick integrating ConsciousnessLoop.

Runs:
  - GlintSignalBus.tick()     — 5 parallel OSINT sources
  - ConsciousnessLoop.tick()  — background thoughts -> SparkNet
  - ReasoningBank.flush()     — pending ReasoningBank distillations

Ouroboros pattern: agent is always "alive" — thinking even with zero users.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = float(os.getenv("HEARTBEAT_INTERVAL", "60"))  # seconds


async def heartbeat_loop() -> None:
    """Main heartbeat coroutine. Run via asyncio.ensure_future()."""
    logger.info("[Heartbeat] Starting — interval=%.0fs", _HEARTBEAT_INTERVAL)

    # Start consciousness in background (independent cadence)
    try:
        from agents.xzero.consciousness import start_consciousness
        await start_consciousness()
        logger.info("[Heartbeat] ConsciousnessLoop started")
    except Exception as e:
        logger.warning("[Heartbeat] ConsciousnessLoop start failed: %s", e)

    while True:
        try:
            await _tick()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[Heartbeat] Tick error: %s", e)
        await asyncio.sleep(_HEARTBEAT_INTERVAL)


async def _tick() -> None:
    tasks = []

    # GlintSignalBus OSINT tick
    try:
        from extensions.xzero.glint_signal_bus import get_glint
        tasks.append(get_glint().tick())
    except Exception as e:
        logger.debug("[Heartbeat] GlintSignalBus unavailable: %s", e)

    # ReasoningBank flush
    try:
        from core.brain import get_reasoning_bank
        rb = get_reasoning_bank()
        if hasattr(rb, "flush"):
            tasks.append(rb.flush())
    except Exception as e:
        logger.debug("[Heartbeat] ReasoningBank unavailable: %s", e)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.debug("[Heartbeat] Task %d error: %s", i, r)
