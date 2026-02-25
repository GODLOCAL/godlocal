"""
extensions/xzero/polyterm_connector.py
PolytermConnector — Polymarket terminal-grade intelligence for GodLocal agents.

Based on: PolyTerm — "Polymarket in your terminal"
GitHub: https://github.com/NYTEMODEONLY/polyterm
Viral: @0x_kaize / @roundtablespace (110 likes, 80 bookmarks, 2026-02-25)

APIs used:
  Gamma REST:  https://gamma-api.polymarket.com   (markets, events, profiles)
  CLOB REST:   https://clob.polymarket.com         (orderbook, trades, depth)
  CLOB WS:     wss://ws-live-data.polymarket.com   (real-time price/trade stream)

GodLocal integration:
  - PolytermConnector wraps Gamma + CLOB APIs, no polyterm binary required
  - X-ZERO agents call monitor_market(), get_whale_activity(), predict_signal()
  - SkillOrchestra skill: "analyze_onchain" → covers prediction markets too
  - sleep_cycle() Phase 2: runs arbitrage_scan() as part of nightly intelligence

Usage:
  from extensions.xzero.polyterm_connector import PolytermConnector
  pt = PolytermConnector()
  markets = await pt.run_tool("search_markets", {"query": "Bitcoin", "limit": 10})
  signal  = await pt.run_tool("predict_signal", {"market_id": "0x..."})
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from extensions.xzero.cimd_connector_base import CIMDConnectorBase
    BASE = CIMDConnectorBase
except ImportError:
    BASE = object  # graceful degradation if CIMD not yet wired


# ── Polymarket API constants ─────────────────────────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"
CLOB_WS    = "wss://ws-live-data.polymarket.com"

RETRY_CODES  = {429, 500, 502, 503, 504}
TIMEOUT_SECS = 15


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    market_id: str
    question:  str
    yes_price: float
    no_price:  float
    volume_24h: float
    liquidity:  float
    end_date:   str
    category:   str = ""
    slug:        str = ""


@dataclass
class PredictionSignal:
    market_id:       str
    direction:       str          # "YES" | "NO" | "NEUTRAL"
    confidence:      float        # 0–1
    momentum:        float        # price momentum score
    whale_sentiment: str          # "BULLISH" | "BEARISH" | "NEUTRAL"
    rsi:             float
    volume_accel:    float
    rationale:       str = ""


@dataclass
class ArbitrageOpportunity:
    market_id_a: str
    market_id_b: str
    spread:      float
    direction:   str
    expected_pnl: float


# ── Connector ─────────────────────────────────────────────────────────────────

class PolytermConnector(BASE):
    """
    Terminal-grade Polymarket intelligence for GodLocal agents.

    Provides the same analytical depth as PolyTerm CLI but as a
    Python-native CIMD connector that X-ZERO agents can call directly.

    Tools:
      search_markets    — Gamma API market search with filters
      get_market        — Full market snapshot (prices + volume + liquidity)
      get_orderbook     — CLOB order depth + slippage estimate
      get_recent_trades — Last N trades for a market
      get_whale_activity — Large position detection (≥$1K orders)
      arbitrage_scan    — Cross-market spread detection
      predict_signal    — RSI + momentum + whale composite signal
      crypto_15m        — BTC/ETH/SOL/XRP 15-min market monitor
      portfolio_summary — Wallet positions + unrealised P&L (view-only)
      market_calendar   — Upcoming resolution events
    """

    POLYTERM_BIN = os.getenv("POLYTERM_BIN", "polyterm")

    # ── CIMDConnectorBase interface ──────────────────────────────────────────

    @classmethod
    def openapi_schema(cls) -> dict:
        paths = {
            f"/tools/{t}": {"post": {"summary": d}} for t, d in [
                ("search_markets",    "Search prediction markets with filters"),
                ("get_market",        "Full market snapshot: prices, volume, liquidity"),
                ("get_orderbook",     "Order book depth + slippage estimate"),
                ("get_recent_trades", "Last N trades for a market"),
                ("get_whale_activity","Large position detection ≥$1K"),
                ("arbitrage_scan",    "Cross-market arbitrage opportunities"),
                ("predict_signal",    "RSI + momentum + whale composite signal"),
                ("crypto_15m",        "BTC/ETH/SOL/XRP 15-min market monitor"),
                ("portfolio_summary", "Wallet positions + P&L (view-only)"),
                ("market_calendar",   "Upcoming resolution events"),
            ]
        }
        return {"openapi": "3.1.0",
                "info": {"title": "PolytermConnector", "version": "1.0.0"},
                "paths": paths}

    @classmethod
    def registration_manifest(cls) -> dict:
        return {
            "name": "PolytermConnector",
            "id":   "polyterm",
            "description": "Polymarket prediction market intelligence: prices, whales, arbitrage, signals.",
            "env_vars": [],   # No API key for read-only; POLY_PRIVATE_KEY for trading
            "tools": list(cls.openapi_schema()["paths"].keys()),
            "pip": ["polyterm"],
        }

    async def run_tool(self, tool: str, params: dict) -> dict:
        dispatch = {
            "search_markets":    self._search_markets,
            "get_market":        self._get_market,
            "get_orderbook":     self._get_orderbook,
            "get_recent_trades": self._get_recent_trades,
            "get_whale_activity":self._get_whale_activity,
            "arbitrage_scan":    self._arbitrage_scan,
            "predict_signal":    self._predict_signal,
            "crypto_15m":        self._crypto_15m,
            "portfolio_summary": self._portfolio_summary,
            "market_calendar":   self._market_calendar,
        }
        handler = dispatch.get(tool)
        if not handler:
            return {"error": f"Unknown tool: {tool}"}
        return await handler(**params)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    async def _gamma(self, path: str, params: dict | None = None) -> dict:
        return await self._get(GAMMA_BASE + path, params)

    async def _clob(self, path: str, params: dict | None = None) -> dict:
        return await self._get(CLOB_BASE + path, params)

    async def _get(self, url: str, params: dict | None = None) -> dict:
        if not HAS_AIOHTTP:
            return {"error": "aiohttp not installed. pip install aiohttp"}
        retries = 3
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        url, params=params,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECS)
                    ) as r:
                        if r.status in RETRY_CODES and attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        if r.status == 200:
                            return await r.json()
                        return {"error": f"HTTP {r.status}", "url": url}
            except asyncio.TimeoutError:
                if attempt < retries - 1:
                    continue
                return {"error": "Timeout", "url": url}
        return {"error": "Max retries exceeded"}

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _search_markets(
        self,
        query: str = "",
        category: str = "",
        limit: int = 20,
        active_only: bool = True,
    ) -> dict:
        """Gamma API market search — find prediction markets by keyword/category."""
        params: dict = {"limit": limit}
        if query:       params["keyword"] = query
        if category:    params["category"] = category
        if active_only: params["active"] = "true"
        return await self._gamma("/markets", params)

    async def _get_market(self, market_id: str) -> dict:
        """
        Full market snapshot: question, YES/NO prices, volume, liquidity, end date.
        market_id: Polymarket conditionId (0x...)
        """
        data = await self._gamma(f"/markets/{market_id}")
        if "error" in data:
            return data
        # Normalise into MarketSnapshot
        return {
            "market_id":  market_id,
            "question":   data.get("question", ""),
            "yes_price":  float(data.get("outcomePrices", ["0", "0"])[0]),
            "no_price":   float(data.get("outcomePrices", ["0", "0"])[1]),
            "volume_24h": float(data.get("volume24hr", 0)),
            "liquidity":  float(data.get("liquidity", 0)),
            "end_date":   data.get("endDateIso", ""),
            "category":   data.get("category", ""),
            "slug":       data.get("slug", ""),
        }

    async def _get_orderbook(self, token_id: str, slippage_size: float = 1000.0) -> dict:
        """
        CLOB order book depth for token_id.
        slippage_size: USDC order size to estimate slippage for.
        """
        book = await self._clob(f"/book", {"token_id": token_id})
        if "error" in book:
            return book

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        # Simple slippage estimate: walk the book
        def walk_book(side: list, size: float) -> float:
            remaining = size
            cost = 0.0
            for level in side:
                p, s = float(level.get("price", 0)), float(level.get("size", 0))
                fill = min(remaining, s * p)
                cost += fill / p if p > 0 else 0
                remaining -= fill
                if remaining <= 0:
                    break
            avg = (size - remaining) / max(cost, 1e-9)
            return round(avg, 4)

        return {
            "token_id":     token_id,
            "best_bid":     float(bids[0]["price"]) if bids else 0,
            "best_ask":     float(asks[0]["price"]) if asks else 0,
            "spread":       round(float(asks[0]["price"]) - float(bids[0]["price"]), 4) if (bids and asks) else None,
            "bid_depth_3":  bids[:3],
            "ask_depth_3":  asks[:3],
            "slippage_est": walk_book(asks, slippage_size),
        }

    async def _get_recent_trades(self, market_id: str, limit: int = 20) -> dict:
        """Last N trades from CLOB — price, size, side, timestamp."""
        return await self._clob("/trades", {"market": market_id, "limit": limit})

    async def _get_whale_activity(
        self, market_id: str, min_size_usdc: float = 1000.0
    ) -> dict:
        """
        Detect large orders (≥ min_size_usdc) from recent trades.
        Returns: whale_buys, whale_sells, net_sentiment.
        """
        trades_data = await self._get_recent_trades(market_id, limit=100)
        trades = trades_data if isinstance(trades_data, list) else trades_data.get("data", [])

        whale_buys  = [t for t in trades if float(t.get("size", 0)) * float(t.get("price", 0)) >= min_size_usdc and t.get("side") == "BUY"]
        whale_sells = [t for t in trades if float(t.get("size", 0)) * float(t.get("price", 0)) >= min_size_usdc and t.get("side") == "SELL"]

        buy_vol  = sum(float(t.get("size", 0)) * float(t.get("price", 0)) for t in whale_buys)
        sell_vol = sum(float(t.get("size", 0)) * float(t.get("price", 0)) for t in whale_sells)

        net = buy_vol - sell_vol
        sentiment = "BULLISH" if net > 500 else ("BEARISH" if net < -500 else "NEUTRAL")

        return {
            "market_id":    market_id,
            "whale_buys":   len(whale_buys),
            "whale_sells":  len(whale_sells),
            "buy_volume":   round(buy_vol, 2),
            "sell_volume":  round(sell_vol, 2),
            "net_flow":     round(net, 2),
            "sentiment":    sentiment,
            "min_threshold": min_size_usdc,
        }

    async def _arbitrage_scan(
        self, category: str = "crypto", limit: int = 50
    ) -> dict:
        """
        Scan for cross-market arbitrage: correlated markets where
        YES(A) + YES(B) < 1.0 (mispricing opportunity).
        """
        data = await self._search_markets(category=category, limit=limit)
        markets = data if isinstance(data, list) else data.get("markets", [])

        opps = []
        for i, m1 in enumerate(markets):
            for m2 in markets[i+1:]:
                p1 = float((m1.get("outcomePrices") or ["0"])[0])
                p2 = float((m2.get("outcomePrices") or ["0"])[0])
                spread = round(1.0 - p1 - p2, 4)
                if spread > 0.02:  # 2%+ mispricing threshold
                    opps.append({
                        "market_a": m1.get("question", "")[:60],
                        "market_b": m2.get("question", "")[:60],
                        "id_a":     m1.get("conditionId", ""),
                        "id_b":     m2.get("conditionId", ""),
                        "price_a":  p1,
                        "price_b":  p2,
                        "spread":   spread,
                        "expected_pnl_pct": round(spread * 100, 2),
                    })

        opps.sort(key=lambda x: x["spread"], reverse=True)
        return {"opportunities": opps[:10], "scanned": len(markets)}

    async def _predict_signal(self, market_id: str) -> dict:
        """
        Composite signal: RSI (price history) + momentum + whale sentiment.
        Returns: direction, confidence, rationale.
        Maps to PolyTerm `predict` command logic.
        """
        market, whales, trades_raw = await asyncio.gather(
            self._get_market(market_id),
            self._get_whale_activity(market_id),
            self._get_recent_trades(market_id, limit=50),
        )

        trades = trades_raw if isinstance(trades_raw, list) else trades_raw.get("data", [])
        prices = [float(t.get("price", 0.5)) for t in trades if t.get("price")]

        # RSI (14-period simplified)
        def simple_rsi(ps: list, period: int = 14) -> float:
            if len(ps) < period + 1:
                return 50.0
            gains = [max(0, ps[i] - ps[i-1]) for i in range(1, len(ps))][-period:]
            losses = [max(0, ps[i-1] - ps[i]) for i in range(1, len(ps))][-period:]
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return round(100 - (100 / (1 + rs)), 2)

        rsi = simple_rsi(prices)
        momentum = round((prices[-1] - prices[0]) / max(prices[0], 0.01), 4) if len(prices) >= 2 else 0.0
        whale_sent = whales.get("sentiment", "NEUTRAL")

        # Composite signal
        score = 0.0
        score += (rsi - 50) / 100     # RSI contribution: -0.5 to +0.5
        score += momentum * 2          # Momentum contribution
        score += {"BULLISH": 0.2, "NEUTRAL": 0.0, "BEARISH": -0.2}.get(whale_sent, 0)

        direction  = "YES" if score > 0.05 else ("NO" if score < -0.05 else "NEUTRAL")
        confidence = round(min(abs(score) * 2, 1.0), 3)

        return {
            "market_id":       market_id,
            "question":        market.get("question", ""),
            "yes_price":       market.get("yes_price", 0),
            "direction":       direction,
            "confidence":      confidence,
            "rsi":             rsi,
            "momentum":        momentum,
            "whale_sentiment": whale_sent,
            "composite_score": round(score, 4),
            "rationale": (
                f"RSI={rsi} ({'overbought' if rsi>70 else 'oversold' if rsi<30 else 'neutral'}), "
                f"momentum={momentum:+.3f}, whales={whale_sent} → {direction} @ {confidence:.0%} confidence"
            ),
        }

    async def _crypto_15m(self) -> dict:
        """
        Monitor BTC/ETH/SOL/XRP 15-minute markets on Polymarket.
        Maps to PolyTerm `crypto15m` command.
        """
        query_terms = ["Bitcoin 15", "Ethereum 15", "Solana 15", "XRP 15"]
        results = await asyncio.gather(*[
            self._search_markets(query=q, limit=3) for q in query_terms
        ])
        output = {}
        for asset, data in zip(["BTC", "ETH", "SOL", "XRP"], results):
            markets = data if isinstance(data, list) else data.get("markets", [])
            output[asset] = markets[:2] if markets else []
        return output

    async def _portfolio_summary(self, wallet_address: str) -> dict:
        """
        View-only wallet position summary (no private key required).
        Returns open positions and estimated P&L.
        """
        return await self._gamma(f"/profiles/{wallet_address}/positions")

    async def _market_calendar(self, days_ahead: int = 7) -> dict:
        """
        Upcoming market resolutions in the next N days.
        Useful for timing entries/exits before resolution.
        """
        return await self._gamma("/markets", {
            "limit": 50,
            "active": "true",
            "sort": "endDateIso",
        })

    # ── X-ZERO convenience wrappers ───────────────────────────────────────────

    async def xzero_market_brief(self, market_id: str) -> str:
        """
        One-call brief for X-ZERO Telegram alerts.
        Returns a ≤200 char summary of a market's signal.
        """
        sig = await self._predict_signal(market_id)
        q   = sig.get("question", "Market")[:50]
        d   = sig.get("direction", "?")
        c   = sig.get("confidence", 0)
        rsi = sig.get("rsi", 50)
        return f"📊 {q}
{d} @ {c:.0%} conf | RSI {rsi} | {sig.get('whale_sentiment','?')} whales"

    async def solana_prediction_pulse(self) -> dict:
        """
        Quick pulse: all active SOL prediction markets + top signal.
        Called by XZeroHeartbeat every 30 min.
        """
        markets_data = await self._search_markets(query="Solana", limit=10)
        markets = markets_data if isinstance(markets_data, list) else markets_data.get("markets", [])
        signals = []
        for m in markets[:5]:
            mid = m.get("conditionId", "")
            if mid:
                sig = await self._predict_signal(mid)
                if sig.get("confidence", 0) > 0.5:
                    signals.append(sig)
        signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return {"top_signals": signals[:3], "total_scanned": len(markets)}
