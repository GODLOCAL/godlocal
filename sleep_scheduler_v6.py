"""sleep_scheduler_v6.py — async nightly sleep_cycle for БОГ || OASIS v6
Replaces while True + sleep(30) with asyncio.sleep until next target time.
Phases: 1 Memory consolidation, 2 Self-evolve, 3 Performance, 4 AutoGenesis
Run standalone: python sleep_scheduler_v6.py
Or import: from sleep_scheduler_v6 import start_scheduler (called by lifespan)
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
from pathlib import Path

from core.brain import Brain
from core.settings import settings
from agents.autogenesis_v2 import AutoGenesis

logger = logging.getLogger(__name__)


def _next_run_dt(hour: int, minute: int) -> datetime.datetime:
    """Next UTC datetime for given hour:minute (always ≥ 1 min from now)."""
    now = datetime.datetime.utcnow()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now + datetime.timedelta(minutes=1):
        target += datetime.timedelta(days=1)
    return target


async def sleep_cycle() -> None:
    """Execute 4-phase sleep cycle."""
    logger.info("🌙 sleep_cycle START")
    brain = Brain.get()
    autogenesis = AutoGenesis(root=".")

    # Phase 1 — Memory consolidation
    logger.info("[Phase 1] Memory consolidation")
    brain.memory.consolidate(brain)

    # Phase 2 — Self-evolve (self_evolve.py)
    logger.info("[Phase 2] Self-evolve")
    try:
        from self_evolve import SelfEvolve  # type: ignore
        se = SelfEvolve(brain=brain)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, se.run)
    except ImportError:
        logger.debug("self_evolve.py not found — skipping Phase 2")
    except Exception as e:
        logger.error("[Phase 2] error: %s", e)

    # Phase 3 — Performance analysis
    logger.info("[Phase 3] Performance analysis")
    try:
        from performance_logger import PerformanceLogger  # type: ignore
        pl = PerformanceLogger()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, pl.analyze)
    except ImportError:
        logger.debug("performance_logger.py not found — skipping Phase 3")
    except Exception as e:
        logger.error("[Phase 3] error: %s", e)

    # Phase 4 — AutoGenesis
    logger.info("[Phase 4] AutoGenesis (apply=%s)", settings.autogenesis_apply)
    try:
        tasks_file = Path("tasks/lessons.md")
        task = tasks_file.read_text(encoding="utf-8")[:1000] if tasks_file.exists() else                "Improve code quality: reduce duplication, add type hints, fix TODOs."
        result = await autogenesis.evolve_async(task=task, apply=settings.autogenesis_apply)
        logger.info("[Phase 4] result: %s patches", sum(f.get("patches", 0) for f in result.get("files", [])))
    except Exception as e:
        logger.error("[Phase 4] AutoGenesis error: %s", e)

    logger.info("☀️  sleep_cycle DONE")


async def scheduler_loop() -> None:
    """Infinite async loop — no time drift, precise UTC scheduling."""
    while True:
        next_dt = _next_run_dt(settings.sleep_hour, settings.sleep_minute)
        wait_sec = (next_dt - datetime.datetime.utcnow()).total_seconds()
        logger.info(
            "⏰ Next sleep_cycle at %s UTC (%.0f min)",
            next_dt.strftime("%H:%M"), wait_sec / 60,
        )
        await asyncio.sleep(wait_sec)
        try:
            await sleep_cycle()
        except Exception as e:
            logger.error("sleep_cycle crashed: %s", e)


def start_scheduler() -> asyncio.Task:
    """Start scheduler as background task inside running event loop."""
    return asyncio.create_task(scheduler_loop(), name="sleep_scheduler")


if __name__ == "__main__":
    import sys
    from utils.logger import setup_logging
    setup_logging()

    if "--now" in sys.argv:
        logger.info("Running sleep_cycle immediately (--now)")
        asyncio.run(sleep_cycle())
    else:
        asyncio.run(scheduler_loop())
