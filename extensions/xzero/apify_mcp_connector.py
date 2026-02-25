"""
extensions/xzero/apify_mcp_connector.py
ApifyMCPConnector — 15,000+ web scraping & automation tools via Apify MCP.

Apify MCP server at https://mcp.apify.com exposes:
  - search_actors()     — find scraping tools in Apify Store
  - call_actor()        — run any Actor, get results
  - rag_web_browser()   — AI-powered general web browsing
  - get_dataset_items() — retrieve scraped data

Used by Sovereign/AutoGenesis for:
  - GlintSignalBus enrichment (scrape Twitter/Reddit/news on demand)
  - Polymarket intelligence (scrape market data, influencer positions)
  - Competitor/token research on demand

Registration: get_connector("apify")

Docs: https://docs.apify.com/platform/integrations/mcp
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

APIFY_TOKEN:   str | None = os.getenv("APIFY_TOKEN")
APIFY_MCP_URL: str        = os.getenv("APIFY_MCP_URL", "https://mcp.apify.com")
APIFY_API_URL: str        = "https://api.apify.com/v2"

APIFY_AVAILABLE: bool = bool(APIFY_TOKEN)


class ApifyMCPConnector:
    """
    Connector to Apify platform — 15,000+ web scraping Actors as tools.
    Primarily uses direct REST API (no local MCP daemon needed).
    Falls back gracefully if APIFY_TOKEN not set.
    """

    def __init__(self) -> None:
        if not APIFY_TOKEN:
            raise RuntimeError(
                "APIFY_TOKEN not set.\n"
                "Get free token at: https://apify.com → Settings → API & Integrations"
            )
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {APIFY_TOKEN}"},
            timeout=60.0,
        )

    # ─── Core Methods ────────────────────────────────────────────────────────

    async def search_actors(self, query: str, limit: int = 10) -> list[dict]:
        """Search Apify Store for actors matching query."""
        r = await self._client.get(
            f"{APIFY_API_URL}/store",
            params={"search": query, "limit": limit},
        )
        r.raise_for_status()
        data = r.json()
        items = data.get("data", {}).get("items", [])
        return [
            {
                "id":          a.get("id"),
                "name":        a.get("name"),
                "username":    a.get("username"),
                "description": a.get("description", "")[:200],
                "actor_id":    f"{a.get('username')}/{a.get('name')}",
            }
            for a in items
        ]

    async def call_actor(
        self,
        actor_id: str,
        input_data: dict[str, Any],
        timeout_secs: int = 60,
        memory_mb: int = 256,
    ) -> list[dict]:
        """
        Run an Apify Actor and return dataset results.
        actor_id: e.g. "apify/web-scraper" or "apify/rag-web-browser"
        """
        # Start run
        run_resp = await self._client.post(
            f"{APIFY_API_URL}/acts/{actor_id}/runs",
            params={"waitForFinish": timeout_secs, "memory": memory_mb},
            json=input_data,
        )
        run_resp.raise_for_status()
        run = run_resp.json().get("data", {})

        dataset_id = run.get("defaultDatasetId")
        status     = run.get("status")

        logger.info(f"Apify [{actor_id}] run={run.get('id')} status={status} dataset={dataset_id}")

        if not dataset_id:
            return []

        return await self.get_dataset_items(dataset_id)

    async def get_dataset_items(
        self,
        dataset_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve scraped items from a dataset."""
        r = await self._client.get(
            f"{APIFY_API_URL}/datasets/{dataset_id}/items",
            params={"limit": limit, "clean": "true"},
        )
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []

    # ─── High-level helpers (used by GlintSignalBus + Sovereign) ─────────────

    async def rag_web_browser(self, query: str, max_results: int = 5) -> list[dict]:
        """
        AI-powered general web search + scrape using apify/rag-web-browser.
        Best for open-ended OSINT queries (news, social, prices).
        """
        logger.info(f"Apify RAG-web-browser: {query[:80]}")
        results = await self.call_actor(
            "apify/rag-web-browser",
            {
                "query": query,
                "maxResults": max_results,
                "outputFormats": ["markdown"],
            },
            timeout_secs=45,
        )
        # Log to SparkNet
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            asyncio.ensure_future(
                get_sparknet().capture(
                    "apify_scrape",
                    f"Apify RAG-browser: {query[:100]} → {len(results)} results",
                    tags=["apify", "web_scraping", "osint"],
                )
            )
        except Exception:
            pass
        return results

    async def scrape_twitter_profile(self, username: str) -> list[dict]:
        """Scrape tweets from a Twitter/X profile (no API key needed)."""
        return await self.call_actor(
            "apify/twitter-scraper",
            {
                "handles": [username.lstrip("@")],
                "maxTweets": 20,
            },
        )

    async def scrape_polymarket(self, query: str) -> list[dict]:
        """Extract Polymarket market data — used by GlintSignalBus fresh-wallet intelligence."""
        return await self.rag_web_browser(f"polymarket {query} prediction market odds")

    async def google_search(self, query: str, max_results: int = 10) -> list[dict]:
        """Google search results via Apify Google Search Scraper."""
        return await self.call_actor(
            "apify/google-search-scraper",
            {
                "queries": [query],
                "maxPagesPerQuery": 1,
                "resultsPerPage": max_results,
            },
        )

    async def close(self) -> None:
        await self._client.aclose()


_apify_instance: ApifyMCPConnector | None = None


def get_apify() -> ApifyMCPConnector:
    global _apify_instance
    if _apify_instance is None:
        _apify_instance = ApifyMCPConnector()
    return _apify_instance
