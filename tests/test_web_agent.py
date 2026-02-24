"""tests/test_web_agent.py — WebAgent coverage"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from extensions.web.web_agent import (
    WebAgent,
    SearchResult,
    FetchResult,
    _Cache,
    get_web_agent,
)


class TestCache:
    def test_miss_returns_none(self):
        c = _Cache(ttl=60)
        assert c.get("nonexistent") is None

    def test_set_and_get(self):
        c = _Cache(ttl=60)
        c.set("k", [1, 2, 3])
        assert c.get("k") == [1, 2, 3]

    def test_expired_returns_none(self):
        c = _Cache(ttl=0)
        c.set("k", "value")
        import time; time.sleep(0.01)
        assert c.get("k") is None

    def test_key_is_deterministic(self):
        c = _Cache()
        k1 = c.key("search", "hello", 5)
        k2 = c.key("search", "hello", 5)
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        c = _Cache()
        assert c.key("search", "a") != c.key("search", "b")


class TestWebAgent:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_results = [
            SearchResult(title="T1", url="https://a.com", snippet="s1"),
            SearchResult(title="T2", url="https://b.com", snippet="s2"),
        ]
        agent = WebAgent()
        with patch("extensions.web.web_agent._ddg_search", AsyncMock(return_value=mock_results)):
            results = await agent.search("test query", n=2)
        assert len(results) == 2
        assert results[0].title == "T1"
        assert results[0].url == "https://a.com"

    @pytest.mark.asyncio
    async def test_search_uses_cache(self):
        agent = WebAgent()
        mock_results = [SearchResult(title="C", url="https://c.com", snippet="c")]
        with patch("extensions.web.web_agent._ddg_search", AsyncMock(return_value=mock_results)) as mock:
            await agent.search("cached query", n=1)
            await agent.search("cached query", n=1)  # should hit cache
            assert mock.call_count == 1  # called only once

    @pytest.mark.asyncio
    async def test_fetch_returns_result(self):
        mock_page = FetchResult(url="https://x.com", title="X", text="content", status=200)
        agent = WebAgent()
        with patch("extensions.web.web_agent._fetch_url", AsyncMock(return_value=mock_page)):
            result = await agent.fetch("https://x.com")
        assert result.status == 200
        assert result.text == "content"

    @pytest.mark.asyncio
    async def test_fetch_error_not_cached(self):
        mock_page = FetchResult(url="https://fail.com", title="", text="", status=0, error="timeout")
        agent = WebAgent()
        with patch("extensions.web.web_agent._fetch_url", AsyncMock(return_value=mock_page)) as mock:
            await agent.fetch("https://fail.com")
            await agent.fetch("https://fail.com")  # errors not cached
            assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_think_with_web_returns_response(self):
        mock_brain = MagicMock()
        mock_brain.think = AsyncMock(return_value="augmented LLM answer")
        mock_results = [SearchResult(title="T", url="https://t.com", snippet="info")]
        mock_page = FetchResult(url="https://t.com", title="T", text="full text", status=200)

        agent = WebAgent()
        with patch("extensions.web.web_agent._ddg_search", AsyncMock(return_value=mock_results)),              patch("extensions.web.web_agent._fetch_url", AsyncMock(return_value=mock_page)):
            result = await agent.think_with_web(mock_brain, "test task")

        assert result["response"] == "augmented LLM answer"
        assert result["web_used"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["url"] == "https://t.com"

    @pytest.mark.asyncio
    async def test_think_with_web_search_false_skips_web(self):
        mock_brain = MagicMock()
        mock_brain.think = AsyncMock(return_value="plain answer")
        agent = WebAgent()
        result = await agent.think_with_web(mock_brain, "task", search=False)
        assert result["web_used"] is False
        assert result["response"] == "plain answer"
        assert result["sources"] == []

    def test_status_returns_dict(self):
        agent = WebAgent()
        s = agent.status()
        assert "provider" in s
        assert "DuckDuckGo" in s["provider"]
        assert "cache_entries" in s

    def test_singleton(self):
        a1 = get_web_agent()
        a2 = get_web_agent()
        assert a1 is a2


class TestWebAPIRouter:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from extensions.web.router import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_web_status(self, client):
        resp = client.get("/web/status")
        assert resp.status_code == 200
        assert "provider" in resp.json()

    def test_web_search(self, client):
        mock_results = [SearchResult(title="T", url="https://r.com", snippet="s")]
        with patch("extensions.web.web_agent._ddg_search", AsyncMock(return_value=mock_results)):
            resp = client.post("/web/search", json={"query": "hello", "n": 1, "fetch_top": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "hello"
        assert len(data["results"]) == 1

    def test_web_fetch(self, client):
        mock_page = FetchResult(url="https://f.com", title="F", text="text", status=200)
        with patch("extensions.web.web_agent._fetch_url", AsyncMock(return_value=mock_page)):
            resp = client.post("/web/fetch", json={"url": "https://f.com"})
        assert resp.status_code == 200
        assert resp.json()["text"] == "text"
