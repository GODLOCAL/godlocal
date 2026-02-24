"""extensions/web/router.py — FastAPI router для web search/fetch endpoints"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, HttpUrl

from .web_agent import get_web_agent

router = APIRouter(prefix="/web", tags=["web"])


class SearchRequest(BaseModel):
    query: str
    n: int = 5
    fetch_top: int = 1


class FetchRequest(BaseModel):
    url: str
    max_chars: int = 8000


class WebThinkRequest(BaseModel):
    task: str
    search: bool = True
    n: int = 5
    fetch_top: int = 1
    max_tokens: int = 2048


@router.post("/search")
async def web_search(req: SearchRequest):
    """Search the web via DuckDuckGo."""
    agent = get_web_agent()
    results = await agent.search(req.query, req.n)
    return {
        "query": req.query,
        "results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ],
    }


@router.post("/fetch")
async def web_fetch(req: FetchRequest):
    """Fetch and extract clean text from a URL."""
    agent = get_web_agent()
    page = await agent.fetch(req.url, req.max_chars)
    return {
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "status": page.status,
        "error": page.error,
    }


@router.post("/think")
async def web_think(req: WebThinkRequest):
    """Brain.think() augmented with live web context."""
    from core.brain import Brain
    brain = Brain.get()
    agent = get_web_agent()
    result = await agent.think_with_web(
        brain,
        task=req.task,
        search=req.search,
        n=req.n,
        fetch_top=req.fetch_top,
        max_tokens=req.max_tokens,
    )
    return result


@router.get("/status")
async def web_status():
    """WebAgent status."""
    agent = get_web_agent()
    return agent.status()
