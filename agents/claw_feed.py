"""
agents/claw_feed.py
ClawFeed — AI news curator for GodLocal v6
─────────────────────────────────────────
Inspired by @wildmindai ClawFeed concept.
Eats X/RSS/HackerNews → structured highlights.
Works as standalone agent OR GoalExecutor skill.

Sources supported:
  - HackerNews   (Top/Best/New stories — no API key)
  - RSS feeds    (any URL via feedparser)
  - Twitter/X    (via GoalExecutor signal or search)

Output:
  - Ranked highlights with relevance score
  - Source pack grouping
  - One-click deep-dive summaries
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import httpx

logger = logging.getLogger("godlocal.claw_feed")

# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class FeedItem:
    id:          str
    title:       str
    url:         str
    source:      str          # "hackernews" | "rss:<name>" | "twitter"
    score:       int          = 0      # upstream score/votes
    relevance:   float        = 0.0    # 0–1 AI-assigned relevance
    summary:     str          = ""
    tags:        list[str]    = field(default_factory=list)
    fetched_at:  str          = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedDigest:
    generated_at: str
    query:        str
    total_items:  int
    highlights:   list[FeedItem]
    source_packs: dict[str, list[FeedItem]]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "query":        self.query,
            "total_items":  self.total_items,
            "highlights":   [h.to_dict() for h in self.highlights],
            "source_packs": {
                k: [i.to_dict() for i in v]
                for k, v in self.source_packs.items()
            },
        }


# ── AIRanker protocol ─────────────────────────────────────────────────────────
AIRanker = Callable[[str, str], Coroutine[Any, Any, str]]


# ── Fetchers ──────────────────────────────────────────────────────────────────

async def fetch_hackernews(
    client: httpx.AsyncClient,
    feed_type: str = "topstories",
    limit: int = 30,
) -> list[FeedItem]:
    """
    feed_type: topstories | beststories | newstories | askstories | showstories
    """
    base = "https://hacker-news.firebaseio.com/v0"
    try:
        resp = await client.get(f"{base}/{feed_type}.json", timeout=10)
        ids  = resp.json()[:limit]
    except Exception as e:
        logger.warning("HN fetch failed: %s", e)
        return []

    async def _item(story_id: int) -> Optional[FeedItem]:
        try:
            r = await client.get(f"{base}/item/{story_id}.json", timeout=8)
            d = r.json()
            if not d or d.get("type") != "story":
                return None
            url = d.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
            return FeedItem(
                id      = f"hn:{story_id}",
                title   = d.get("title", ""),
                url     = url,
                source  = "hackernews",
                score   = d.get("score", 0),
            )
        except Exception as e:
            logger.debug("HN item %s failed: %s", story_id, e)
            return None

    tasks  = [_item(i) for i in ids]
    items  = await asyncio.gather(*tasks)
    result = [i for i in items if i]
    logger.info("HN: fetched %d/%d stories", len(result), limit)
    return result


async def fetch_rss(
    client: httpx.AsyncClient,
    feed_url: str,
    feed_name: str,
    limit: int = 20,
) -> list[FeedItem]:
    """Fetch and parse an RSS/Atom feed."""
    try:
        import feedparser  # type: ignore
    except ImportError:
        logger.warning("feedparser not installed — run: pip install feedparser")
        return []

    try:
        resp    = await client.get(feed_url, timeout=12, follow_redirects=True)
        parsed  = feedparser.parse(resp.text)
    except Exception as e:
        logger.warning("RSS %s failed: %s", feed_url, e)
        return []

    items = []
    for entry in parsed.entries[:limit]:
        uid = hashlib.md5(entry.get("link", entry.get("id", "")).encode()).hexdigest()[:12]
        items.append(FeedItem(
            id     = f"rss:{uid}",
            title  = entry.get("title", ""),
            url    = entry.get("link", ""),
            source = f"rss:{feed_name}",
            score  = 0,
        ))
    logger.info("RSS %s: fetched %d items", feed_name, len(items))
    return items


# ── Default source packs ──────────────────────────────────────────────────────

DEFAULT_RSS_PACKS: dict[str, str] = {
    "techcrunch":   "https://techcrunch.com/feed/",
    "verge":        "https://www.theverge.com/rss/index.xml",
    "arxiv_cs_ai":  "https://rss.arxiv.org/rss/cs.AI",
    "solana_blog":  "https://solana.com/news/rss.xml",
    "producthunt":  "https://www.producthunt.com/feed",
}


# ── ClawFeedAgent ─────────────────────────────────────────────────────────────

class ClawFeedAgent:
    """
    AI news curator. Usage:

        agent = ClawFeedAgent(ai_runner=brain.think)

        # Curate with default sources
        digest = await agent.curate("AI agents on Solana")

        # Deep dive into a single item
        summary = await agent.deep_dive(digest.highlights[0])

        # Add custom RSS source
        agent.add_rss("defillama", "https://defillama.com/blog/rss.xml")
    """

    def __init__(
        self,
        ai_runner:   AIRanker,
        cache_dir:   Path = Path("data/claw_feed"),
        cache_ttl:   int  = 3600,          # seconds
        top_n:       int  = 10,            # highlights to return
        rss_packs:   Optional[dict[str, str]] = None,
        hn_feeds:    Optional[list[str]]       = None,
    ):
        self.ai         = ai_runner
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl  = cache_ttl
        self.top_n      = top_n
        self.rss_packs  = rss_packs or dict(DEFAULT_RSS_PACKS)
        self.hn_feeds   = hn_feeds  or ["topstories", "beststories"]
        self._cache: dict[str, tuple[float, Any]] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def add_rss(self, name: str, url: str) -> None:
        self.rss_packs[name] = url
        logger.info("ClawFeed: added RSS source '%s'", name)

    def remove_rss(self, name: str) -> None:
        self.rss_packs.pop(name, None)

    async def curate(
        self,
        query:   str = "",
        sources: Optional[list[str]] = None,    # subset of rss_packs keys + "hackernews"
        limit:   int = 20,
    ) -> FeedDigest:
        """
        Main entry point — fetch, filter, rank, summarise.
        Returns a FeedDigest with highlights + source packs.
        """
        logger.info("ClawFeed curate: query='%s'", query)
        raw = await self._fetch_all(sources=sources, limit=limit)
        ranked = await self._rank(raw, query)
        highlights = ranked[:self.top_n]

        # Parallel summarise top 5
        top5 = highlights[:5]
        summaries = await asyncio.gather(*[self._summarise(item) for item in top5])
        for item, s in zip(top5, summaries):
            item.summary = s

        source_packs: dict[str, list[FeedItem]] = {}
        for item in ranked:
            source_packs.setdefault(item.source, []).append(item)

        digest = FeedDigest(
            generated_at = datetime.now(timezone.utc).isoformat(),
            query        = query,
            total_items  = len(raw),
            highlights   = highlights,
            source_packs = source_packs,
        )
        self._persist(digest)
        return digest

    async def deep_dive(self, item: FeedItem) -> str:
        """Fetch full page text and return AI summary."""
        cache_key = f"dive:{item.id}"
        if cache_key in self._cache:
            ts, val = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return val

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(item.url, follow_redirects=True)
                text = resp.text[:8000]   # trim for LLM
        except Exception as e:
            return f"Fetch failed: {e}"

        prompt = (
            f"Summarise this article in 3–5 bullet points. Focus on key insights.\n\n"
            f"Title: {item.title}\nURL: {item.url}\n\n{text}"
        )
        summary = await self.ai(prompt, f"claw_dive:{item.id}")
        self._cache[cache_key] = (time.time(), summary)
        return summary

    def last_digest(self) -> Optional[dict]:
        """Load the most recently persisted digest."""
        files = sorted(self.cache_dir.glob("digest_*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text(encoding="utf-8"))

    # ── Internals ─────────────────────────────────────────────────────────

    async def _fetch_all(
        self,
        sources: Optional[list[str]] = None,
        limit:   int = 20,
    ) -> list[FeedItem]:
        enabled_rss = {
            k: v for k, v in self.rss_packs.items()
            if sources is None or k in sources
        }
        use_hn = sources is None or "hackernews" in sources

        async with httpx.AsyncClient(
            headers={"User-Agent": "GodLocal-ClawFeed/1.0"},
        ) as client:
            tasks = []
            if use_hn:
                for feed_type in self.hn_feeds:
                    tasks.append(fetch_hackernews(client, feed_type, limit))
            for name, url in enabled_rss.items():
                tasks.append(fetch_rss(client, url, name, limit))

            batches = await asyncio.gather(*tasks, return_exceptions=True)

        all_items: list[FeedItem] = []
        for batch in batches:
            if isinstance(batch, list):
                all_items.extend(batch)
            elif isinstance(batch, Exception):
                logger.warning("Fetch error: %s", batch)

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[FeedItem] = []
        for item in all_items:
            if item.url not in seen:
                seen.add(item.url)
                unique.append(item)

        logger.info("ClawFeed: %d unique items from %d sources", len(unique), len(batches))
        return unique

    async def _rank(self, items: list[FeedItem], query: str) -> list[FeedItem]:
        """Ask LLM to score items by relevance to query, fall back to HN score."""
        if not items:
            return []
        if not query:
            # Rank by upstream score
            return sorted(items, key=lambda x: x.score, reverse=True)

        batch = [{"idx": i, "title": it.title, "source": it.source}
                 for i, it in enumerate(items)]
        prompt = (
            f"Rate these news items by relevance to: '{query}'\n"
            f"Return JSON: [{{'idx': N, 'score': 0.0-1.0}}]\n"
            f"Score 1.0 = extremely relevant, 0.0 = not relevant.\n"
            f"Items:\n{json.dumps(batch, ensure_ascii=False)[:6000]}"
        )
        try:
            raw = await self.ai(prompt, "claw_rank")
            start = raw.index("["); end = raw.rindex("]") + 1
            scores = {entry["idx"]: entry["score"] for entry in json.loads(raw[start:end])}
            for i, item in enumerate(items):
                item.relevance = scores.get(i, 0.0)
        except Exception as e:
            logger.warning("Ranking LLM failed (%s) — using score fallback", e)
            max_score = max((it.score for it in items), default=1) or 1
            for item in items:
                item.relevance = item.score / max_score

        return sorted(items, key=lambda x: (x.relevance, x.score), reverse=True)

    async def _summarise(self, item: FeedItem) -> str:
        prompt = (
            f"One sentence (max 25 words): what is this article about?\n"
            f"Title: {item.title}\nURL: {item.url}"
        )
        try:
            return await self.ai(prompt, f"claw_summary:{item.id}")
        except Exception as e:
            return item.title

    def _persist(self, digest: FeedDigest) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.cache_dir / f"digest_{ts}.json"
        path.write_text(
            json.dumps(digest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Keep only last 10 digests
        old = sorted(self.cache_dir.glob("digest_*.json"))[:-10]
        for f in old:
            f.unlink(missing_ok=True)
        logger.info("ClawFeed: digest saved → %s", path.name)


# ── FastAPI routes ─────────────────────────────────────────────────────────────

def register_routes(app, agent: "ClawFeedAgent") -> None:
    """
    Mount ClawFeed routes onto existing FastAPI app.

    Usage in godlocal_v6.py:
        from agents.claw_feed import ClawFeedAgent, register_routes as register_feed
        feed_agent = ClawFeedAgent(ai_runner=brain.think)
        register_feed(app, feed_agent)
    """
    from fastapi import Body, Query
    from fastapi.responses import JSONResponse

    @app.get("/feed")
    async def get_feed(
        q:       str            = Query("", description="Topic / query to rank by"),
        sources: Optional[str]  = Query(None, description="Comma-separated source names"),
        limit:   int            = Query(20,  description="Items per source"),
    ):
        """Curate and return ranked feed digest."""
        src_list = [s.strip() for s in sources.split(",")] if sources else None
        digest = await agent.curate(query=q, sources=src_list, limit=limit)
        return JSONResponse(digest.to_dict())

    @app.get("/feed/last")
    async def last_digest():
        """Return the last cached digest."""
        d = agent.last_digest()
        if not d:
            return JSONResponse({"error": "no digest yet"}, status_code=404)
        return JSONResponse(d)

    @app.post("/feed/dive")
    async def deep_dive(url: str = Body(..., embed=True),
                         title: str = Body("", embed=True)):
        """Deep dive into a single URL."""
        item = FeedItem(
            id=hashlib.md5(url.encode()).hexdigest()[:12],
            title=title or url, url=url, source="manual"
        )
        summary = await agent.deep_dive(item)
        return JSONResponse({"url": url, "summary": summary})

    @app.post("/feed/sources")
    async def add_source(name: str = Body(..., embed=True),
                          rss_url: str = Body(..., embed=True)):
        """Add a custom RSS source."""
        agent.add_rss(name, rss_url)
        return JSONResponse({"added": name, "url": rss_url})

    @app.get("/feed/sources")
    async def list_sources():
        """List all RSS sources."""
        return JSONResponse({"sources": agent.rss_packs, "hn_feeds": agent.hn_feeds})


# ── Example wiring ─────────────────────────────────────────────────────────────
#
# from agents.claw_feed import ClawFeedAgent, register_routes as register_feed
#
# feed_agent = ClawFeedAgent(
#     ai_runner = brain.think,
#     top_n     = 10,
# )
# register_feed(app, feed_agent)
#
# # Optional — add custom RSS sources
# feed_agent.add_rss("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/")
# feed_agent.add_rss("x100news", "https://x100-app.vercel.app/rss.xml")
#
# GET  /feed?q=Solana+AI+agents         → ranked digest
# GET  /feed/last                        → last cached
# POST /feed/dive  {"url":"..."}         → deep summary
# POST /feed/sources {"name":"...","rss_url":"..."}
# GET  /feed/sources
