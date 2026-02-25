"""
extensions/xzero/glint_signal_bus.py
GlintSignalBus — GlintIntel-pattern multi-source signal aggregator for X-ZERO.

Inspired by: @glintintel (Polymarket-backed terminal, Feb 2026)
Pattern: 5 parallel sources → classify → match to market → 30s to screen

Sources:
  1. OSINT: Twitter/X keyword monitor (crypto + geopolitical signals)
  2. Telegram: channel scanner (public crypto alpha channels)
  3. ADS-B: military/unusual flight activity via ADS-B Exchange (free API)
  4. Whale: on-chain large move detector via Whale Alert (or public Solscan)
  5. Fresh Wallet: Polymarket new wallet + big bet detector (GlintIntel key insight)

All signals go to SparkNetConnector with semantic tags.
XZeroHeartbeat calls glint_signal_bus_tick() every 30 min (alongside solana_prediction_pulse).

Usage:
    from extensions.xzero.glint_signal_bus import GlintSignalBus
    bus = GlintSignalBus()
    signals = await bus.tick()   # Run all 5 sources in parallel, returns list[GlintSignal]

Env vars (all optional — bus degrades gracefully without them):
  WHALE_ALERT_API_KEY  — whale alert HTTP API key
  ADSB_EXCHANGE_KEY    — ADS-B Exchange RapidAPI key
  TWITTER_BEARER       — Twitter Bearer token for keyword search
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class GlintSignal:
    """
    A single market-moving signal from any source.
    30s pipeline: source → classify → match → sparknet.
    """
    source:       str                   # "osint_twitter" | "whale" | "flight" | "fresh_wallet" | "telegram"
    content:      str                   # Human-readable description ≤200 chars
    tags:         list[str]             # Semantic tags for SparkNet routing
    urgency:      float                 # 0.0–1.0 (1.0 = trade now)
    raw:          dict = field(default_factory=dict)  # Original API payload
    ts:           float = field(default_factory=time.time)

    @property
    def spark_content(self) -> str:
        """Formatted for SparkNet capture() — ≤200 chars."""
        return self.content[:200]


# ── ADS-B connector (military flight tracking) ────────────────────────────────

MILITARY_CALLSIGN_PREFIXES = {
    "RCH", "REACH", "SPAR", "IRON", "DARK",  # USAF AMC
    "JAKE", "TOPAZ", "NOBLE", "VIPER",         # combat + recon
    "FORTE", "CHAOS", "SKULL",                 # special ops
    "RRR",                                      # Royal Air Force
    "CTM", "GATOR",                             # NATO
}

SQUAWK_EMERGENCY = {"7500", "7600", "7700"}

async def fetch_adsb_signals(session: aiohttp.ClientSession, api_key: str | None) -> list[GlintSignal]:
    """
    Fetch unusual military/government flights from ADS-B Exchange.
    Free tier: 100 req/day via adsbexchange.com/api — no key needed for v2/lat/lon/dist/.
    """
    signals: list[GlintSignal] = []
    try:
        # Public endpoint: military squawks
        url = "https://opendata.adsbexchange.com/flight-data-samples/2022-01-01/2022-01-01-1400Z.json.gz"
        # Use lightweight public v2 API (no key needed for lat/lon/dist)
        # Hotspot: Washington DC area (38.9 lat, -77.0 lon, 250nm)
        api_url = "https://api.adsbexchange.com/v2/lat/38.9/lon/-77.0/dist/250/"
        headers = {}
        if api_key:
            headers["api-auth"] = api_key

        async with session.get(api_url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return signals
            data = await resp.json(content_type=None)
            aircraft = data.get("ac", []) or []

            for ac in aircraft[:50]:   # cap at 50 to avoid noise
                callsign = (ac.get("flight") or ac.get("r") or "").strip().upper()
                squawk = str(ac.get("squawk") or "")
                mil = ac.get("mil") == 1 or ac.get("military") == 1
                prefix_match = any(callsign.startswith(p) for p in MILITARY_CALLSIGN_PREFIXES)

                if mil or prefix_match or squawk in SQUAWK_EMERGENCY:
                    alt = ac.get("alt_baro") or ac.get("alt_geom") or 0
                    lat = ac.get("lat", 0)
                    lon = ac.get("lon", 0)
                    desc = f"[FLIGHT] {callsign or 'UNKNOWN'} sqk={squawk} alt={alt}ft @{lat:.1f},{lon:.1f}"
                    urgency = 0.7 if squawk in SQUAWK_EMERGENCY else 0.4
                    tags = ["flight", "military", "osint"]
                    if squawk in SQUAWK_EMERGENCY:
                        tags.append("emergency")
                    signals.append(GlintSignal(
                        source="flight",
                        content=desc[:200],
                        tags=tags,
                        urgency=urgency,
                        raw=ac,
                    ))
    except Exception as e:
        pass   # Degrade gracefully
    return signals[:5]   # Max 5 flight signals per tick


# ── Whale Alert connector ─────────────────────────────────────────────────────

WHALE_USD_THRESHOLD = 500_000   # $500k minimum move

async def fetch_whale_signals(session: aiohttp.ClientSession, api_key: str | None) -> list[GlintSignal]:
    """
    Fetch large on-chain transfers from Whale Alert API.
    Without API key: falls back to public Solscan large-tx endpoint.
    """
    signals: list[GlintSignal] = []
    try:
        if api_key:
            url = f"https://api.whale-alert.io/v1/transactions?api_key={api_key}&min_value={WHALE_USD_THRESHOLD}&start={int(time.time()) - 1800}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return signals
                data = await resp.json()
                for tx in (data.get("transactions") or [])[:10]:
                    symbol = tx.get("symbol", "?").upper()
                    amount_usd = tx.get("amount_usd", 0)
                    from_owner = tx.get("from", {}).get("owner_type", "unknown")
                    to_owner = tx.get("to", {}).get("owner_type", "unknown")
                    desc = f"[WHALE] {symbol} ${amount_usd:,.0f} {from_owner}→{to_owner}"
                    urgency = min(1.0, amount_usd / 5_000_000)
                    signals.append(GlintSignal(
                        source="whale",
                        content=desc[:200],
                        tags=["whale", symbol.lower(), "onchain"],
                        urgency=urgency,
                        raw=tx,
                    ))
        else:
            # Fallback: Solscan public large transfers (SOL + tokens)
            url = "https://public-api.solscan.io/transaction/last?limit=20&filter=largeAmount"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    txs = await resp.json()
                    for tx in (txs if isinstance(txs, list) else [])[:5]:
                        sig = tx.get("txHash", "")[:16]
                        desc = f"[WHALE/SOL] large tx {sig}..."
                        signals.append(GlintSignal(
                            source="whale",
                            content=desc[:200],
                            tags=["whale", "sol", "onchain"],
                            urgency=0.3,
                            raw=tx,
                        ))
    except Exception:
        pass
    return signals[:5]


# ── Fresh wallet + big bet detector (GlintIntel key insight) ─────────────────

FRESH_WALLET_AGE_DAYS = 7
FRESH_WALLET_BET_USD  = 10_000   # $10k minimum bet

async def fetch_fresh_wallet_signals(session: aiohttp.ClientSession) -> list[GlintSignal]:
    """
    GlintIntel pattern: fresh wallet + big bet = insider.
    "you know the new wallet + huge bet pattern that always wins? Glint tracks it live."

    Queries Polymarket CLOB API for recent large trades from wallets with
    first transaction < FRESH_WALLET_AGE_DAYS days ago.
    Falls back to public CLOB trades endpoint.
    """
    signals: list[GlintSignal] = []
    try:
        # Polymarket CLOB: recent large trades
        url = "https://clob.polymarket.com/trades?limit=50&filter=large"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return signals
            data = await resp.json()
            trades = data if isinstance(data, list) else data.get("data", [])

            now = time.time()
            for trade in trades[:50]:
                maker = trade.get("maker_address") or trade.get("taker_address") or ""
                size = float(trade.get("size") or trade.get("amount") or 0)
                price = float(trade.get("price") or 0.5)
                usd_value = size * price

                if usd_value < FRESH_WALLET_BET_USD:
                    continue

                # Check wallet freshness via Polymarket profile API
                try:
                    profile_url = f"https://api.polymarket.com/user?address={maker}"
                    async with session.get(profile_url, timeout=aiohttp.ClientTimeout(total=4)) as pr:
                        if pr.status == 200:
                            profile = await pr.json()
                            created = profile.get("created_at") or profile.get("timestamp")
                            if created:
                                age_days = (now - float(created)) / 86400
                                if age_days < FRESH_WALLET_AGE_DAYS:
                                    market_id = trade.get("market") or trade.get("condition_id", "")[:16]
                                    outcome = trade.get("outcome") or trade.get("side", "?")
                                    desc = (
                                        f"[FRESH WALLET] {maker[:8]}…{maker[-4:]} "
                                        f"age={age_days:.1f}d bet=${usd_value:,.0f} "
                                        f"on {market_id} [{outcome}]"
                                    )
                                    signals.append(GlintSignal(
                                        source="fresh_wallet",
                                        content=desc[:200],
                                        tags=["whale", "insider", "fresh_wallet", "polymarket", "signal"],
                                        urgency=0.9,   # High urgency — classic insider pattern
                                        raw=trade,
                                    ))
                except Exception:
                    pass
    except Exception:
        pass
    return signals[:3]   # Max 3 fresh-wallet signals (they're high urgency)


# ── Twitter OSINT keyword scanner ─────────────────────────────────────────────

OSINT_KEYWORDS = [
    "breaking:", "just in:", "confirmed:", "military action",
    "emergency declaration", "exchange hack", "rug pull",
    "SEC action", "coinbase acquisition", "bitcoin reserve",
    "nuclear", "taiwan strait", "missile",
]

async def fetch_osint_twitter_signals(session: aiohttp.ClientSession, bearer: str | None) -> list[GlintSignal]:
    """
    Twitter keyword monitor for market-moving OSINT.
    Requires Twitter Bearer token (Basic tier or above).
    Without token: returns empty (degrade gracefully).
    """
    if not bearer:
        return []
    signals: list[GlintSignal] = []
    try:
        query = " OR ".join(f'"{kw}"' for kw in OSINT_KEYWORDS[:5]) + " -is:retweet lang:en"
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {"query": query, "max_results": "10", "tweet.fields": "created_at,public_metrics"}
        headers = {"Authorization": f"Bearer {bearer}"}
        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return signals
            data = await resp.json()
            for tweet in (data.get("data") or []):
                text = tweet.get("text", "")[:180]
                metrics = tweet.get("public_metrics", {})
                likes = metrics.get("like_count", 0)
                desc = f"[OSINT/X] {text}"
                urgency = min(0.8, 0.3 + likes / 1000)
                signals.append(GlintSignal(
                    source="osint_twitter",
                    content=desc[:200],
                    tags=["osint", "twitter", "breaking"],
                    urgency=urgency,
                    raw=tweet,
                ))
    except Exception:
        pass
    return signals[:3]


# ── GlintSignalBus ─────────────────────────────────────────────────────────────

class GlintSignalBus:
    """
    GlintIntel-pattern signal aggregator for GodLocal xzero.

    Runs 4 sources in parallel, pushes high-urgency signals to SparkNet.
    Called by XZeroHeartbeat.solana_prediction_pulse() every 30 min.

    Urgency levels:
      0.9+ → fresh_wallet insider      → immediate spark + alert
      0.7+ → whale move / emergency    → spark with tags [whale, signal]
      0.4+ → military flight / OSINT   → spark with tags [flight, osint]
      <0.4 → background noise          → skip (don't pollute SparkNet)

    Usage:
        bus = GlintSignalBus()
        signals = await bus.tick()
        # High-urgency signals already pushed to SparkNet automatically
    """

    URGENCY_THRESHOLD = 0.4   # Below this → don't push to SparkNet

    def __init__(self) -> None:
        self._whale_key  = os.getenv("WHALE_ALERT_API_KEY")
        self._adsb_key   = os.getenv("ADSB_EXCHANGE_KEY")
        self._twitter_bearer = os.getenv("TWITTER_BEARER")
        self._sparknet   = None   # Lazy — set on first tick()

    def _get_sparknet(self):
        if self._sparknet is None:
            from extensions.xzero.sparknet_connector import get_sparknet
            self._sparknet = get_sparknet()
        return self._sparknet

    async def tick(self) -> list[GlintSignal]:
        """
        Run all signal sources in parallel.
        Push signals with urgency >= URGENCY_THRESHOLD to SparkNet.
        Returns all signals collected.

        Completes in ~8s (bounded by slowest source timeout).
        """
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                fetch_adsb_signals(session, self._adsb_key),
                fetch_whale_signals(session, self._whale_key),
                fetch_fresh_wallet_signals(session),
                fetch_osint_twitter_signals(session, self._twitter_bearer),
                fetch_apify_intelligence(),
                return_exceptions=True,
            )

        all_signals: list[GlintSignal] = []
        for r in results:
            if isinstance(r, list):
                all_signals.extend(r)

        # Sort by urgency descending
        all_signals.sort(key=lambda s: s.urgency, reverse=True)

        # Push to SparkNet
        sparknet = self._get_sparknet()
        pushes = []
        for sig in all_signals:
            if sig.urgency >= self.URGENCY_THRESHOLD:
                pushes.append(sparknet.capture(
                    agent="glint_signal_bus",
                    content=sig.spark_content,
                    tags=sig.tags,
                    context=sig.source,
                ))
        if pushes:
            await asyncio.gather(*pushes, return_exceptions=True)

        return all_signals

    async def high_urgency_signals(self, threshold: float = 0.7) -> list[GlintSignal]:
        """Run tick() and return only high-urgency signals."""
        all_sigs = await self.tick()
        return [s for s in all_sigs if s.urgency >= threshold]

    def log_summary(self, signals: list[GlintSignal]) -> str:
        counts = {}
        for s in signals:
            counts[s.source] = counts.get(s.source, 0) + 1
        parts = [f"{src}={n}" for src, n in sorted(counts.items())]
        urgent = sum(1 for s in signals if s.urgency >= 0.7)
        return f"GlintSignalBus: {len(signals)} signals ({", ".join(parts)}) | urgent={urgent}"


# ── Apify 5th intelligence source ─────────────────────────────────────────────

async def fetch_apify_intelligence() -> list[GlintSignal]:
    """
    5th GlintSignalBus source: Apify RAG-web-browser for open-ended OSINT.
    Searches crypto whale movements + Polymarket intelligence on every tick.
    No-op (returns []) if APIFY_TOKEN not set.
    """
    from extensions.xzero.apify_mcp_connector import APIFY_AVAILABLE, get_apify
    if not APIFY_AVAILABLE:
        return []
    try:
        apify = get_apify()
        queries = [
            "crypto market whale movements urgent news today",
            "polymarket prediction market high-value bets today",
        ]
        all_results = await asyncio.gather(
            *[apify.rag_web_browser(q, max_results=3) for q in queries],
            return_exceptions=True,
        )
        signals: list[GlintSignal] = []
        for results in all_results:
            if not isinstance(results, list):
                continue
            for item in results[:2]:
                text = str(item.get("markdown", item.get("text", item.get("url", ""))))[:300]
                if not text.strip():
                    continue
                signals.append(GlintSignal(
                    source="apify_osint",
                    signal_type="web_intelligence",
                    content=text,
                    urgency=0.5,
                    tags=["apify", "web", "osint"],
                ))
        return signals
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Apify intelligence fetch failed: {exc}")
        return []


# ── Module-level singleton ────────────────────────────────────────────────────

_bus: GlintSignalBus | None = None

def get_signal_bus() -> GlintSignalBus:
    global _bus
    if _bus is None:
        _bus = GlintSignalBus()
    return _bus
