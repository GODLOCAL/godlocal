"""
extensions/xzero/hyperliquid_connector.py — Hyperliquid Perps connector (READ-ONLY)
Hyperliquid: on-chain perpetuals DEX on Arbitrum.
API docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
"""
from __future__ import annotations
import logging
from typing import Optional
import requests
from .base import XZeroConnector

logger = logging.getLogger(__name__)

HL_API = "https://api.hyperliquid.xyz/info"


class HyperliquidConnector(XZeroConnector):
    NAME = "hyperliquid"

    def __init__(self, daily_limit: float = 500.0):
        self.daily_limit = daily_limit

    def _post(self, payload: dict) -> dict:
        resp = requests.post(HL_API, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def market_digest(self) -> list[dict]:
        """Fetch top 20 perp markets by open interest."""
        data = self._post({"type": "metaAndAssetCtxs"})
        universe = data[0].get("universe", [])
        ctxs     = data[1] if len(data) > 1 else []

        markets = []
        for asset, ctx in zip(universe[:20], ctxs[:20]):
            markets.append({
                "id": asset.get("name"),
                "name": asset.get("name"),
                "mark_px": ctx.get("markPx"),
                "open_interest": ctx.get("openInterest"),
                "funding": ctx.get("funding"),
                "volume_24h": ctx.get("dayNtlVlm"),
                "platform": "hyperliquid",
            })
        # Sort by open interest descending
        markets.sort(key=lambda x: float(x.get("open_interest") or 0), reverse=True)
        return markets

    def assess_for_trade(self, opp: dict) -> dict:
        """
        Funding-rate signal: positive funding → shorts are paying → short bias.
        Negative funding → longs paying → long bias.
        READ-ONLY signal — no actual trade execution.
        """
        funding = float(opp.get("funding") or 0)
        amount  = min(100.0, self.daily_limit * 0.1)
        threshold = 0.0003  # 0.03% per 8h ~ 33% APR

        if funding > threshold:
            return {"action": "short_signal", "amount": amount,
                    "reason": f"High positive funding={funding:.5f} → short bias",
                    "platform": "hyperliquid"}
        if funding < -threshold:
            return {"action": "long_signal", "amount": amount,
                    "reason": f"Negative funding={funding:.5f} → long bias",
                    "platform": "hyperliquid"}
        return {"action": "skip", "amount": 0,
                "reason": f"Funding neutral: {funding:.5f}", "platform": "hyperliquid"}
