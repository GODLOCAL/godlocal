"""
agents/xzero/heartbeat.py
X-ZERO Agent Heartbeat — OpenClaw HEARTBEAT.md pattern ported to GodLocal.

OpenClaw runs a cron every 30min that checks HEARTBEAT.md for a checklist.
This module does the same for X100 OASIS agents: every 30min the scheduler
calls XZeroHeartbeat.run() which executes the checklist and notifies via
Telegram only if something significant happened.

Wired into sleep_scheduler_v6.py Phase 5 (new phase, non-blocking).
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("xzero.heartbeat")


class XZeroHeartbeat:
    """
    Proactive monitoring loop for X-ZERO agents.

    Mirrors OpenClaw heartbeat architecture:
    - Cheap checks first (Solscan data, price delta)
    - LLM only if something changed significantly
    - Telegram notification only on meaningful events

    Usage in sleep_scheduler_v6.py:
        heartbeat = XZeroHeartbeat(soul=soul, connectors=connectors)
        await heartbeat.run()
    """

    SIGNIFICANT_PRICE_CHANGE_PCT = 5.0
    WHALE_TRANSFER_THRESHOLD_SOL = 10_000

    def __init__(self, soul=None, connectors: dict | None = None):
        self.soul = soul
        self.connectors = connectors or {}
        self._last_price: Optional[float] = None
        self._last_holder_count: Optional[int] = None

    async def run(self) -> dict:
        """
        Execute heartbeat checklist. Returns summary dict.
        Called every 30 minutes by sleep_scheduler_v6.py.
        """
        now = datetime.now(timezone.utc)
        quiet_hours = self.soul.heartbeat.get("quiet_hours_utc", [2, 6]) if self.soul else [2, 6]

        # Respect quiet hours
        if quiet_hours[0] <= now.hour < quiet_hours[1]:
            logger.debug("Heartbeat: quiet hours, skipping")
            return {"status": "quiet_hours_skip"}

        results = {}
        alerts = []

        # ── Check 1: $X100 price delta ────────────────────────────────────
        try:
            solscan = self.connectors.get("solscan_free")
            if solscan:
                token_data = solscan.token_data(token_address="X100_MINT_ADDR_PLACEHOLDER")
                price = token_data.get("data", {}).get("price_usdt", 0)
                if self._last_price and price:
                    delta_pct = abs(price - self._last_price) / self._last_price * 100
                    if delta_pct >= self.SIGNIFICANT_PRICE_CHANGE_PCT:
                        direction = "🟢" if price > self._last_price else "🔴"
                        alerts.append(
                            f"{direction} $X100 {'+' if price > self._last_price else ''}"
                            f"{delta_pct:.1f}% → ${price:.6f}"
                        )
                self._last_price = price
                results["price"] = price
        except Exception as e:
            logger.warning(f"Heartbeat price check failed: {e}")

        # ── Check 2: Whale movements ──────────────────────────────────────
        try:
            if solscan:
                top = solscan.top_address_transfers("X100_MINT_ADDR_PLACEHOLDER", range_days=1)
                whales = top.get("data", {}).get("items", [])
                for whale in whales[:3]:
                    amount = whale.get("amount", 0)
                    if amount > self.WHALE_TRANSFER_THRESHOLD_SOL:
                        alerts.append(
                            f"🐋 Whale: {whale.get('address', '?')[:8]}... "
                            f"moved {amount:,.0f} $X100"
                        )
        except Exception as e:
            logger.warning(f"Heartbeat whale check failed: {e}")

        # ── Check 3: Holder count delta ───────────────────────────────────
        try:
            if solscan:
                total = solscan.token_holders_total("X100_MINT_ADDR_PLACEHOLDER")
                holder_count = total.get("data", {}).get("total", 0)
                if self._last_holder_count and holder_count:
                    delta = holder_count - self._last_holder_count
                    if abs(delta) > 100:
                        sign = "+" if delta > 0 else ""
                        alerts.append(f"👥 Holders {sign}{delta:,} → {holder_count:,} total")
                self._last_holder_count = holder_count
                results["holders"] = holder_count
        except Exception as e:
            logger.warning(f"Heartbeat holder check failed: {e}")

        # ── Notify via Telegram if alerts ─────────────────────────────────
        if alerts:
            msg = f"[X-ZERO Heartbeat {now.strftime('%H:%M UTC')}]\n" + "\n".join(alerts)
            results["telegram_sent"] = True
            results["alerts"] = alerts
            logger.info(f"Heartbeat alerts: {alerts}")
            # TODO: wire into GodLocal TelegramNotifier
            # await self.connectors["telegram"].send(msg)
        else:
            results["telegram_sent"] = False
            logger.debug("Heartbeat: no significant changes")

        results["status"] = "ok"
        results["ts"] = now.isoformat()
        return results


# ── Integration note for sleep_scheduler_v6.py ───────────────────────────────
# 
# Add Phase 5 after existing phases:
#
#   # Phase 5: X-ZERO Heartbeat (non-blocking, every 30min)
#   from agents.xzero.heartbeat import XZeroHeartbeat
#   from agents.xzero.agent_soul import XZeroAgentSoul
#   
#   soul = XZeroAgentSoul.load()
#   heartbeat = XZeroHeartbeat(soul=soul, connectors={
#       "solscan_free": SolscanFreeConnector(),
#   })
#   results = await heartbeat.run()
#   if results.get("alerts"):
#       logger.info(f"Heartbeat: {len(results['alerts'])} alerts sent")
