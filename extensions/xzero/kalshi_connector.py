"""
extensions/xzero/kalshi_connector.py — Kalshi Prediction Markets connector
Real-money US-regulated prediction markets.
API docs: https://trading-api.readme.io/reference
"""
from __future__ import annotations
import logging
from typing import Optional
import requests
from .base import XZeroConnector

logger = logging.getLogger(__name__)

KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2"


class KalshiConnector(XZeroConnector):
    NAME = "kalshi"

    def __init__(self, api_key: Optional[str] = None, daily_limit: float = 200.0):
        self.api_key = api_key
        self.daily_limit = daily_limit
        self._headers = {"accept": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def market_digest(self) -> list[dict]:
        """Fetch active markets sorted by volume."""
        resp = requests.get(
            f"{KALSHI_API}/events",
            params={"limit": 20, "status": "open"},
            headers=self._headers,
            timeout=10
        )
        if resp.status_code == 401:
            logger.warning("[Kalshi] Auth required — set KALSHI_API_KEY in .env")
            return []
        resp.raise_for_status()
        events = resp.json().get("events", [])
        results = []
        for ev in events:
            for market in ev.get("markets", []):
                results.append({
                    "id": market.get("ticker"),
                    "question": market.get("title") or ev.get("title"),
                    "yes_bid": market.get("yes_bid"),
                    "no_bid": market.get("no_bid"),
                    "volume": market.get("volume"),
                    "platform": "kalshi",
                    "close_time": market.get("expiration_time"),
                })
        return results

    def assess_for_trade(self, opp: dict) -> dict:
        """Signal based on bid spread — tight spread = efficient market, skip."""
        yes_bid = opp.get("yes_bid", 50)
        no_bid  = opp.get("no_bid",  50)
        spread  = abs(100 - yes_bid - no_bid)
        amount  = min(30.0, self.daily_limit * 0.05)

        if spread > 15 and yes_bid < 35:
            return {"action": "bet_yes", "amount": amount,
                    "reason": f"Wide spread + low yes_bid={yes_bid}", "platform": "kalshi"}
        if spread > 15 and no_bid < 35:
            return {"action": "bet_no", "amount": amount,
                    "reason": f"Wide spread + low no_bid={no_bid}", "platform": "kalshi"}
        return {"action": "skip", "amount": 0, "reason": f"Spread too tight: {spread}", "platform": "kalshi"}
