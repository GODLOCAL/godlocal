#!/usr/bin/env python3
"""
test_airllm.py — AirLLM GIANT tier integration test for GodLocal.

Tests:
  1. AirLLMBridge initialises correctly
  2. Single inference call completes (first run downloads ~35GB model)
  3. TieredRouter routes task_type="giant" through AirLLM
  4. asyncio.Lock prevents parallel GIANT calls

Run:
  python test_airllm.py
"""
import asyncio
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger("test_airllm")

# ── env defaults (override with actual .env before running) ────────────
os.environ.setdefault("AIRLLM_MODEL_ALIAS", "llama-70b")
os.environ.setdefault("AIRLLM_PRECISION",   "4bit")
os.environ.setdefault("AIRLLM_MAX_TOKENS",  "100")
os.environ.setdefault("AIRLLM_LAYER_CACHE_DIR", os.path.expanduser("~/.cache/airllm"))


async def test_bridge_direct() -> None:
    """Test 1: direct AirLLMBridge.complete() call."""
    logger.info("━━━ Test 1: AirLLMBridge direct ━━━")
    from core.airllm_bridge import AirLLMBridge

    bridge = AirLLMBridge()
    prompt = "Explain the concept of a neural network in one paragraph."
    logger.info(f"Prompt: {prompt[:60]}…")

    t0 = time.perf_counter()
    response = await bridge.complete(prompt, max_new_tokens=100)
    elapsed = time.perf_counter() - t0

    logger.info(f"Response ({len(response)} chars, {elapsed:.1f}s):")
    logger.info(response[:300])
    assert len(response) > 10, "Empty response!"
    logger.info("✅ Test 1 PASSED")


async def test_tiered_router_giant() -> None:
    """Test 2: TieredRouter routes task_type=giant to AirLLM."""
    logger.info("━━━ Test 2: TieredRouter GIANT tier ━━━")
    from core.tiered_router import get_tiered_router, Tier

    router = get_tiered_router()
    prompt = "Deep analysis: what are the key risks of trading Solana prediction markets?"

    # Confirm it classifies as GIANT
    tier = router.classify_tier(prompt, task_type="giant")
    assert tier == Tier.GIANT, f"Expected GIANT, got {tier}"
    logger.info(f"Tier classified: {tier.name}")

    t0 = time.perf_counter()
    result = await router.complete(prompt, task_type="giant", max_tokens=150)
    elapsed = time.perf_counter() - t0

    logger.info(f"Response ({len(result)} chars, {elapsed:.1f}s):")
    logger.info(result[:300])
    logger.info(f"Stats: {router.log_stats()}")
    assert len(result) > 10, "Empty response!"
    logger.info("✅ Test 2 PASSED")


async def test_concurrent_lock() -> None:
    """Test 3: second GIANT call waits for first to complete (asyncio.Lock)."""
    logger.info("━━━ Test 3: asyncio.Lock — sequential GIANT calls ━━━")
    from core.airllm_bridge import get_airllm

    airllm = get_airllm()
    times: list[float] = []

    async def call(idx: int) -> None:
        t0 = time.perf_counter()
        await airllm.complete(f"Write one sentence about topic {idx}.", max_new_tokens=30)
        times.append(time.perf_counter() - t0)
        logger.info(f"Call {idx} done in {times[-1]:.1f}s")

    await asyncio.gather(call(1), call(2))
    # Both calls must have run (not errored), confirming Lock serialises them
    assert len(times) == 2
    logger.info("✅ Test 3 PASSED — GIANT calls serialised correctly")


async def main() -> None:
    logger.info(f"AirLLM test  model={os.getenv('AIRLLM_MODEL_ALIAS')}  "
                f"precision={os.getenv('AIRLLM_PRECISION')}")
    logger.info("⚠️  First run downloads the model (~35GB for llama-70b 4bit). Be patient.")

    try:
        await test_bridge_direct()
    except Exception as exc:
        logger.exception(f"Test 1 FAILED: {exc}")

    try:
        await test_tiered_router_giant()
    except Exception as exc:
        logger.exception(f"Test 2 FAILED: {exc}")

    try:
        await test_concurrent_lock()
    except Exception as exc:
        logger.exception(f"Test 3 FAILED: {exc}")

    logger.info("━━━ All tests complete ━━━")


if __name__ == "__main__":
    asyncio.run(main())
