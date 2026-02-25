"""
polymarket_connector.py — Prediction Market Intelligence for X-ZERO
════════════════════════════════════════════════════════════════════
Connects X-ZERO (Eliza + Picobot) to Polymarket prediction markets.

Inspired by: @SuhailKakar polymarket-cli (Rust) + Polymarket/agents (Python, MIT)
Source: github.com/Polymarket/agents

What this adds to X-ZERO:
  - Market signal feed: "Is BTC >100K by March?" → 73% yes → bullish signal
  - Before any DCA/trade, query relevant prediction markets for context
  - XZeroDelegator can use market_probability() as an assessment input
  - GodLocal sleep_cycle() can include market_digest() for nightly briefing

No trading on Polymarket from restricted jurisdictions (see Polymarket ToS).
This module is READ-ONLY by default — market data as AI signal input only.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Polymarket API endpoints (public, no auth for reads)
# ──────────────────────────────────────────────────────────────────────────────
GAMMA_API  = "https://gamma-api.polymarket.com"
CLOB_API   = "https://clob.polymarket.com"
DATA_DIR   = "godlocal_data/polymarket"

CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "crypto", "defi", "token", "blockchain", "altcoin",
    "x100", "pump", "memecoin",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketSignal:
    """A prediction market translated into a trading signal."""
    question:    str
    yes_prob:    float          # 0.0 – 1.0
    no_prob:     float
    volume_usd:  float
    end_date:    Optional[str]
    signal:      str            # "bullish" | "bearish" | "neutral"
    confidence:  str            # "high" | "medium" | "low"
    market_id:   str
    url:         str = ""
    fetched_at:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MarketDigest:
    """Nightly summary of relevant prediction markets — fed into sleep_cycle()."""
    signals:      list[MarketSignal]
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    overall_bias: str = "neutral"   # "bullish" | "bearish" | "neutral"
    summary:      str = ""
    date:         str = field(default_factory=lambda: datetime.utcnow().date().isoformat())


# ──────────────────────────────────────────────────────────────────────────────
# Core connector
# ──────────────────────────────────────────────────────────────────────────────

class PolymarketConnector:
    """
    READ-ONLY Polymarket data connector.
    Fetches market probabilities and translates them into trading signals
    for X-ZERO's DynamicAssessor and GodLocal's sleep_cycle().
    """

    TIMEOUT = 10
    MAX_MARKETS = 20

    def _get(self, url: str) -> dict | list | None:
        try:
            req = Request(url, headers={"User-Agent": "godlocal-xzero/1.0"})
            with urlopen(req, timeout=self.TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except URLError as e:
            logger.warning(f"[Polymarket] GET {url} failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"[Polymarket] parse error: {e}")
            return None

    def get_crypto_markets(self, limit: int = 10) -> list[dict]:
        """Fetch active crypto-related prediction markets sorted by volume."""
        data = self._get(f"{GAMMA_API}/markets?active=true&closed=false&limit={limit}&sort_by=volume24hr")
        if not data:
            return []
        markets = data if isinstance(data, list) else data.get("markets", [])
        # Filter to crypto-relevant markets
        filtered = []
        for m in markets:
            q = (m.get("question") or "").lower()
            if any(kw in q for kw in CRYPTO_KEYWORDS):
                filtered.append(m)
        return filtered[:limit]

    def search_markets(self, query: str, limit: int = 5) -> list[dict]:
        """Search markets by keyword."""
        q = query.replace(" ", "+")
        data = self._get(f"{GAMMA_API}/markets?active=true&closed=false&limit={limit}&_c={q}")
        if not data:
            return []
        return data if isinstance(data, list) else data.get("markets", [])

    def market_probability(self, market_id: str) -> Optional[float]:
        """Return YES probability (0.0–1.0) for a specific market."""
        data = self._get(f"{GAMMA_API}/markets/{market_id}")
        if not data:
            return None
        try:
            tokens = data.get("tokens") or []
            for t in tokens:
                if t.get("outcome", "").upper() == "YES":
                    price = float(t.get("price", 0))
                    return round(price, 4)
        except (TypeError, ValueError):
            pass
        return None

    def to_signal(self, market: dict) -> MarketSignal:
        """Convert raw Polymarket market dict → MarketSignal."""
        question = market.get("question", "")
        tokens   = market.get("tokens") or []
        yes_prob = no_prob = 0.5

        for t in tokens:
            outcome = (t.get("outcome") or "").upper()
            try:
                price = float(t.get("price", 0.5))
            except (TypeError, ValueError):
                price = 0.5
            if outcome == "YES":
                yes_prob = round(price, 4)
            elif outcome == "NO":
                no_prob = round(price, 4)

        volume = 0.0
        try:
            volume = float(market.get("volume24hr") or market.get("volume") or 0)
        except (TypeError, ValueError):
            pass

        # Signal classification
        if yes_prob >= 0.65:
            signal = "bullish"
        elif yes_prob <= 0.35:
            signal = "bearish"
        else:
            signal = "neutral"

        # Confidence based on volume
        if volume >= 100_000:
            confidence = "high"
        elif volume >= 10_000:
            confidence = "medium"
        else:
            confidence = "low"

        end_date = market.get("endDate") or market.get("end_date_iso")
        market_id = str(market.get("id") or market.get("condition_id") or "")
        url = f"https://polymarket.com/event/{market.get('slug', market_id)}"

        return MarketSignal(
            question=question,
            yes_prob=yes_prob,
            no_prob=no_prob,
            volume_usd=volume,
            end_date=end_date,
            signal=signal,
            confidence=confidence,
            market_id=market_id,
            url=url,
        )

    def get_crypto_signals(self, limit: int = 10) -> list[MarketSignal]:
        """
        Main method for X-ZERO DynamicAssessor integration.
        Returns prediction market signals for crypto markets.
        """
        markets = self.get_crypto_markets(limit=limit)
        signals = [self.to_signal(m) for m in markets]
        logger.info(f"[Polymarket] {len(signals)} crypto signals fetched")
        return signals

    def market_digest(self, limit: int = 10) -> MarketDigest:
        """
        Nightly digest for GodLocal sleep_cycle() integration.
        Summarizes prediction market sentiment for crypto.
        """
        signals = self.get_crypto_signals(limit=limit)
        digest  = MarketDigest(signals=signals)

        for s in signals:
            if s.signal == "bullish":
                digest.bullish_count += 1
            elif s.signal == "bearish":
                digest.bearish_count += 1
            else:
                digest.neutral_count += 1

        if digest.bullish_count > digest.bearish_count + digest.neutral_count:
            digest.overall_bias = "bullish"
        elif digest.bearish_count > digest.bullish_count + digest.neutral_count:
            digest.overall_bias = "bearish"
        else:
            digest.overall_bias = "neutral"

        # Build text summary (for sleep_cycle() insight extraction)
        lines = [f"Polymarket Digest — {digest.date}",
                 f"Overall bias: {digest.overall_bias.upper()} "
                 f"(🟢{digest.bullish_count} bullish / 🔴{digest.bearish_count} bearish / ⚪{digest.neutral_count} neutral)"]
        for s in signals[:5]:
            bar = "🟢" if s.signal == "bullish" else ("🔴" if s.signal == "bearish" else "⚪")
            lines.append(f"{bar} {s.question[:60]} — YES {s.yes_prob:.0%} [{s.confidence}]")
        digest.summary = "\n".join(lines)

        return digest

    def assess_for_trade(self, token: str) -> dict:
        """
        X-ZERO DynamicAssessor hook.
        Call before a DCA/swap to get prediction market context.

        Returns dict with:
          market_bias: "bullish" | "bearish" | "neutral"
          confidence:  "high" | "medium" | "low" | "no_data"
          top_signal:  MarketSignal | None
          recommendation: str
        """
        query = token.lower()
        markets = self.search_markets(query, limit=3)
        if not markets:
            return {
                "market_bias": "neutral",
                "confidence": "no_data",
                "top_signal": None,
                "recommendation": f"No prediction markets found for {token} — proceed with standard risk rules",
            }

        signals = [self.to_signal(m) for m in markets]
        # Pick highest volume signal
        top = max(signals, key=lambda s: s.volume_usd)

        rec_map = {
            ("bullish", "high"):   "Strong market consensus supports this trade",
            ("bullish", "medium"): "Moderate bullish signal — proceed with normal position sizing",
            ("bullish", "low"):    "Weak bullish signal — low volume, treat as noise",
            ("bearish", "high"):   "⚠️ Strong bearish consensus — consider reducing position or skipping",
            ("bearish", "medium"): "Moderate bearish signal — consider halving position size",
            ("bearish", "low"):    "Weak bearish signal — low volume, proceed cautiously",
            ("neutral", "high"):   "Market undecided — neutral signal, use standard sizing",
            ("neutral", "medium"): "Neutral signal — proceed normally",
            ("neutral", "low"):    "Insufficient market data",
        }
        rec = rec_map.get((top.signal, top.confidence), "Neutral signal — proceed normally")

        return {
            "market_bias": top.signal,
            "confidence":  top.confidence,
            "top_signal":  top,
            "recommendation": rec,
        }



# ── Closed-candle gating (GlintIntel pattern) ─────────────────────────────────

from datetime import datetime, timezone
import math


class ClosedCandleGate:
    """
    GlintIntel insight: don't trade on an open candle — wait for confirmation.
    "30 seconds from event to screen" but only trade after candle closes.

    Gating policy options:
      "closed_candle"  — wait until current 1-min candle closes before allowing trade
      "open"           — no gating, trade immediately (old behaviour)

    Usage:
        gate = ClosedCandleGate(policy="closed_candle")
        if gate.is_open():
            result = connector.execute_trade(...)
        else:
            secs = gate.seconds_until_next()
            await asyncio.sleep(secs)
            result = connector.execute_trade(...)
    """

    CANDLE_SECONDS = 60   # 1-minute candle

    def __init__(self, policy: str = "closed_candle"):
        self.policy = policy
        self._last_candle: int = 0   # last candle start epoch

    def _current_candle(self) -> int:
        """Returns the start epoch of the current 1-min candle."""
        now = int(time.time())
        return (now // self.CANDLE_SECONDS) * self.CANDLE_SECONDS

    def is_open(self) -> bool:
        """
        Returns True if a trade is allowed NOW.
        closed_candle policy: True only in the first 5s of a new candle
        (i.e., previous candle just closed).
        """
        if self.policy != "closed_candle":
            return True
        candle_start = self._current_candle()
        elapsed_in_candle = int(time.time()) - candle_start
        # Allow in the first 5 seconds of each new candle
        is_fresh = elapsed_in_candle <= 5
        if is_fresh and candle_start != self._last_candle:
            self._last_candle = candle_start
            return True
        return False

    def seconds_until_next(self) -> float:
        """Seconds until the next candle opens (i.e., next trade window)."""
        candle_start = self._current_candle()
        candle_end = candle_start + self.CANDLE_SECONDS
        return max(0.0, candle_end - time.time())

    async def wait_for_candle(self) -> None:
        """Async wait until the next candle is open."""
        import asyncio
        wait = self.seconds_until_next()
        if wait > 0:
            await asyncio.sleep(wait)


# Module-level gate — shared across all PolymarketConnector instances
_default_gate = ClosedCandleGate(policy="closed_candle")

def get_candle_gate() -> ClosedCandleGate:
    return _default_gate


# ──────────────────────────────────────────────────────────────────────────────
# sleep_cycle() integration hook
# Called from SleepCycle.run() in godlocal_v5.py Phase 3+
# ──────────────────────────────────────────────────────────────────────────────

def get_polymarket_digest_for_sleep_cycle() -> dict:
    """
    Drop-in for sleep_cycle() integration.
    Returns dict compatible with SleepCycle.run() report format.

    Usage in godlocal_v5.py SleepCycle.run():
        try:
            from polymarket_connector import get_polymarket_digest_for_sleep_cycle
            report["polymarket"] = get_polymarket_digest_for_sleep_cycle()
        except Exception as e:
            report["polymarket"] = {"error": str(e)}
    """
    connector = PolymarketConnector()
    digest = connector.market_digest(limit=10)
    return {
        "overall_bias":   digest.overall_bias,
        "bullish":        digest.bullish_count,
        "bearish":        digest.bearish_count,
        "neutral":        digest.neutral_count,
        "summary":        digest.summary,
        "markets_count":  len(digest.signals),
        "date":           digest.date,
    }
