"""extensions/web/web_agent.py — WebAgent: search + fetch for БОГ || OASIS v6

Capabilities:
  search(query, n)  — DuckDuckGo (no API key needed) with Brave fallback
  fetch(url)        — httpx GET → trafilatura clean text extraction
  think_with_web()  — Brain.think() + auto web context injection
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

log = logging.getLogger("oasis.web")


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class FetchResult:
    url: str
    title: str
    text: str
    status: int
    error: Optional[str] = None


# ── Cache (in-memory, TTL 10 min) ─────────────────────────────────────────────
class _Cache:
    def __init__(self, ttl: int = 600):
        self._store: dict[str, tuple[float, object]] = {}
        self._ttl = ttl

    def get(self, key: str):
        if key in self._store:
            ts, val = self._store[key]
            if time.time() - ts < self._ttl:
                return val
        return None

    def set(self, key: str, val):
        self._store[key] = (time.time(), val)

    def key(self, *parts) -> str:
        return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


_cache = _Cache()


# ── DuckDuckGo search (HTML scrape, no API key) ───────────────────────────────
async def _ddg_search(query: str, n: int = 5) -> list[SearchResult]:
    """DuckDuckGo HTML search — free, no API key, no rate limit enforcement."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))
        return results
    except ImportError:
        log.warning("duckduckgo-search not installed — falling back to DDG lite HTML")
        return await _ddg_html_fallback(query, n)
    except Exception as e:
        log.warning(f"DDG search error: {e}")
        return await _ddg_html_fallback(query, n)


async def _ddg_html_fallback(query: str, n: int = 5) -> list[SearchResult]:
    """DuckDuckGo Lite HTML fallback (no JS, no tracking)."""
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://duckduckgo.com/lite/",
                params={"q": query, "kl": "wt-wt"},
                headers={"User-Agent": "Mozilla/5.0 (compatible; GodLocal/6.0)"},
            )
            text = resp.text
            # Extract result links and snippets
            links = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>', text)
            snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>([^<]+)</td>', text)
            results = []
            for i, (url, title) in enumerate(links[:n]):
                snippet = snippets[i] if i < len(snippets) else ""
                results.append(SearchResult(title=title.strip(), url=url.strip(), snippet=snippet.strip()))
            return results
    except Exception as e:
        log.error(f"DDG lite fallback failed: {e}")
        return []


# ── URL fetch + clean text ─────────────────────────────────────────────────────
async def _fetch_url(url: str, max_chars: int = 8000) -> FetchResult:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; GodLocal/6.0)"},
            )
            raw_html = resp.text

        # Try trafilatura (best quality)
        try:
            import trafilatura
            text = trafilatura.extract(
                raw_html,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            ) or ""
        except ImportError:
            # Fallback: strip tags
            text = re.sub(r"<[^>]+>", " ", raw_html)
            text = re.sub(r"\s+", " ", text).strip()

        # Extract title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", raw_html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else url

        return FetchResult(
            url=url,
            title=title,
            text=text[:max_chars],
            status=resp.status_code,
        )
    except Exception as e:
        return FetchResult(url=url, title="", text="", status=0, error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# WebAgent
# ══════════════════════════════════════════════════════════════════════════════
class WebAgent:
    """
    Async web agent for БОГ || OASIS.

    Usage:
        web = WebAgent()
        results = await web.search("latest AI news", n=5)
        page    = await web.fetch("https://example.com")
        answer  = await web.think_with_web(brain, "What is FEP in neuroscience?")
    """

    def __init__(self):
        pass

    async def search(self, query: str, n: int = 5) -> list[SearchResult]:
        key = _cache.key("search", query, n)
        cached = _cache.get(key)
        if cached:
            log.debug(f"Cache hit: search '{query}'")
            return cached
        results = await _ddg_search(query, n)
        _cache.set(key, results)
        return results

    async def fetch(self, url: str, max_chars: int = 8000) -> FetchResult:
        key = _cache.key("fetch", url)
        cached = _cache.get(key)
        if cached:
            log.debug(f"Cache hit: fetch '{url}'")
            return cached
        result = await _fetch_url(url, max_chars)
        if not result.error:
            _cache.set(key, result)
        return result

    async def search_and_fetch(
        self, query: str, n: int = 3, top_pages: int = 2, max_chars: int = 4000
    ) -> dict:
        """Search + fetch top N pages. Returns structured context."""
        results = await self.search(query, n)
        pages = []
        if top_pages > 0:
            tasks = [self.fetch(r.url, max_chars) for r in results[:top_pages]]
            pages = await asyncio.gather(*tasks)
        return {"query": query, "results": results, "pages": pages}

    async def think_with_web(
        self,
        brain,
        task: str,
        search: bool = True,
        n: int = 5,
        fetch_top: int = 1,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Augmented thinking: inject web context into Brain.think().

        Returns:
            {"response": str, "sources": list[dict], "web_used": bool}
        """
        sources = []
        context_block = ""

        if search:
            try:
                data = await self.search_and_fetch(task, n=n, top_pages=fetch_top)
                snippets = []
                for i, r in enumerate(data["results"], 1):
                    snippets.append(f"[{i}] {r.title}
{r.url}
{r.snippet}")
                    sources.append({"index": i, "title": r.title, "url": r.url})

                pages_text = ""
                for p in data.get("pages", []):
                    if p.text:
                        pages_text += f"\n\n--- {p.title} ({p.url}) ---\n{p.text[:3000]}"

                context_block = (
                    "=== WEB CONTEXT ===\n"
                    + "\n\n".join(snippets)
                    + pages_text
                    + "\n=== END WEB CONTEXT ===\n\n"
                )
            except Exception as e:
                log.warning(f"Web context build failed: {e}")

        augmented_task = context_block + task if context_block else task
        response = await brain.think(augmented_task, max_tokens=max_tokens)

        return {
            "response": response,
            "sources": sources,
            "web_used": bool(context_block),
        }

    def status(self) -> dict:
        return {
            "provider": "DuckDuckGo (no-key)",
            "fetch": "trafilatura",
            "cache_entries": len(_cache._store),
            "cache_ttl_sec": _cache._ttl,
        }


# Singleton
_web_agent: Optional[WebAgent] = None


def get_web_agent() -> WebAgent:
    global _web_agent
    if _web_agent is None:
        _web_agent = WebAgent()
    return _web_agent
