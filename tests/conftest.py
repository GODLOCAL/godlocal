"""tests/conftest.py — shared fixtures for БОГ || OASIS v6 test suite"""
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Shared event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_root(tmp_path):
    """Temporary project root with soul file + memory dir."""
    soul = tmp_path / "BOH_OASIS.md"
    soul.write_text("You are БОГ || OASIS. Test soul.", encoding="utf-8")
    (tmp_path / "godlocal_data" / "memory").mkdir(parents=True)
    (tmp_path / "tests").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def mock_ollama_response():
    return {"message": {"content": "Test LLM response"}}


@pytest.fixture
def mock_ollama(mock_ollama_response):
    with patch("ollama.chat", return_value=mock_ollama_response):
        yield


@pytest.fixture
def mock_chromadb(tmp_path):
    """Real ChromaDB in temp directory."""
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path / "test_memory"))
    return client


@pytest.fixture(autouse=True)
def reset_brain_singleton():
    """Reset Brain singleton between tests to prevent state bleed."""
    yield
    try:
        from core.brain import Brain
        Brain.reset()
    except ImportError:
        pass
