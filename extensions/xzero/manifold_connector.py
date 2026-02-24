"""
extensions/xzero/manifold_connector.py — Manifold Markets READ connector
Manifold: play-money prediction markets, best for signal testing.
API docs: https://docs.manifold.markets/api
"""
from __future__ import annotations
import logging
from typing import Optional
import requests
from .base import XZeroConnector

logger = logging.getLogger(__name__)

MANIFOLD_API = "https://api.manifold.markets/v0"
DEFAULT_LIMIT = 20


class ManifoldConnector(XZeroConnector):
    NAME = "manifold"

    def __init__(self, api_key: Optional[str] = None, daily_limit: float = 500.0):
        self.api_key = api_key
        self.daily_limit = daily_limit
        self._headers = {"Authorization": f"Key {api_key}"} if api_key else {}

    def market_digest(self) -> list[dict]:
        """Fetch top active markets by liquidity."""
        resp = requests.get(
            f"{MANIFOLD_API}/markets",
            params={"limit": DEFAULT_LIMIT, "sort": "liquidity", "order": "desc"},
            timeout=10
        )
        resp.raise_for_status()
        markets = resp.json()
        return [
            {
                "id": m["id"],
                "question": m["question"],
                "probability": m.get("probability"),
                "volume": m.get("volume", 0),
                "close_time": m.get("closeTime"),
                "platform": "manifold",
                "url": m.get("url"),
            }
            for m in markets
            if m.get("isResolved") is False
        ]

    def assess_for_trade(self, opp: dict) -> dict:
        """
        Simple edge detector: bet YES if prob < 0.35 (underpriced),
        bet NO if prob > 0.75 (overpriced). READ-ONLY — returns signal only.
        """
        prob = opp.get("probability", 0.5)
        amount = min(50.0, self.daily_limit * 0.05)

        if prob < 0.35:
            return {"action": "bet_yes", "amount": amount,
                    "reason": f"Underpriced YES: prob={prob:.2f}", "platform": "manifold"}
        if prob > 0.75:
            return {"action": "bet_no", "amount": amount,
                    "reason": f"Overpriced YES: prob={prob:.2f}", "platform": "manifold"}
        return {"action": "skip", "amount": 0, "reason": "No edge detected", "platform": "manifold"}
