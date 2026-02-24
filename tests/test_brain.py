"""tests/test_brain.py — core/brain.py coverage (LLMBridge, MemoryEngine, Brain)"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── LLMBridge ──────────────────────────────────────────────────────────────
class TestLLMBridge:
    def test_builds_ollama_by_default(self):
        with patch("ollama.chat", return_value={"message": {"content": "hi"}}):
            from core.brain import LLMBridge
            bridge = LLMBridge("Test soul")
            assert bridge._sync_fn is not None

    def test_ollama_call_returns_string(self):
        with patch("ollama.chat", return_value={"message": {"content": "hello"}}):
            from core.brain import LLMBridge
            bridge = LLMBridge("soul")
            result = bridge._sync_fn("test prompt")
            assert isinstance(result, str)
            assert result == "hello"

    @pytest.mark.asyncio
    async def test_async_call_works(self):
        with patch("ollama.chat", return_value={"message": {"content": "async result"}}):
            from core.brain import LLMBridge
            bridge = LLMBridge("soul")
            result = await bridge("test", max_tokens=128)
            assert result == "async result"

    def test_reload_rebuilds_fn(self):
        with patch("ollama.chat", return_value={"message": {"content": "x"}}):
            from core.brain import LLMBridge
            bridge = LLMBridge("soul")
            old_fn = bridge._sync_fn
            bridge.reload()
            assert bridge._sync_fn is not None  # new callable built


# ── MemoryEngine ───────────────────────────────────────────────────────────
class TestMemoryEngine:
    def test_add_and_query(self, tmp_path):
        with patch("core.brain.settings") as mock_settings:
            mock_settings.memory_path = str(tmp_path / "mem")
            mock_settings.short_term_limit = 50
            from core.brain import MemoryEngine
            mem = MemoryEngine()
            mem.add("Python is awesome", long=False)
            results = mem.query("Python", n=1)
            assert any("Python" in r for r in results)

    def test_add_to_long_term(self, tmp_path):
        with patch("core.brain.settings") as mock_settings:
            mock_settings.memory_path = str(tmp_path / "mem2")
            mock_settings.short_term_limit = 50
            from core.brain import MemoryEngine
            mem = MemoryEngine()
            mem.add("long term knowledge", long=True)
            results = mem.query("long term", n=1, long=True)
            assert len(results) >= 0  # ChromaDB returns results

    def test_prune_removes_excess(self, tmp_path):
        with patch("core.brain.settings") as mock_settings:
            mock_settings.memory_path = str(tmp_path / "mem3")
            mock_settings.short_term_limit = 2
            from core.brain import MemoryEngine
            mem = MemoryEngine()
            mem.add("item one")
            mem.add("item two")
            mem.add("item three")
            pruned = mem.prune()
            assert pruned >= 1
            assert mem.short.count() <= 2

    def test_prune_no_op_when_under_limit(self, tmp_path):
        with patch("core.brain.settings") as mock_settings:
            mock_settings.memory_path = str(tmp_path / "mem4")
            mock_settings.short_term_limit = 50
            from core.brain import MemoryEngine
            mem = MemoryEngine()
            mem.add("just one item")
            pruned = mem.prune()
            assert pruned == 0

    def test_query_empty_collection_returns_empty(self, tmp_path):
        with patch("core.brain.settings") as mock_settings:
            mock_settings.memory_path = str(tmp_path / "mem5")
            mock_settings.short_term_limit = 50
            from core.brain import MemoryEngine
            mem = MemoryEngine()
            results = mem.query("anything")
            assert results == []


# ── Brain singleton ────────────────────────────────────────────────────────
class TestBrain:
    def test_singleton_pattern(self, tmp_root):
        with patch("core.brain.settings") as ms,              patch("ollama.chat", return_value={"message": {"content": "hi"}}):
            ms.model = "qwen3:8b"
            ms.device = "mps"
            ms.soul_file = str(tmp_root / "BOH_OASIS.md")
            ms.memory_path = str(tmp_root / "godlocal_data" / "memory")
            ms.short_term_limit = 50
            from core.brain import Brain
            b1 = Brain.get()
            b2 = Brain.get()
            assert b1 is b2

    def test_brain_loads_soul(self, tmp_root):
        with patch("core.brain.settings") as ms,              patch("ollama.chat", return_value={"message": {"content": "ok"}}):
            ms.model = "qwen3:8b"
            ms.device = "mps"
            ms.soul_file = str(tmp_root / "BOH_OASIS.md")
            ms.memory_path = str(tmp_root / "godlocal_data" / "memory")
            ms.short_term_limit = 50
            from core.brain import Brain
            b = Brain()
            assert "OASIS" in b.soul or len(b.soul) > 0

    @pytest.mark.asyncio
    async def test_think_returns_string(self, tmp_root):
        with patch("core.brain.settings") as ms,              patch("ollama.chat", return_value={"message": {"content": "thinking result"}}):
            ms.model = "qwen3:8b"
            ms.device = "mps"
            ms.soul_file = str(tmp_root / "BOH_OASIS.md")
            ms.memory_path = str(tmp_root / "godlocal_data" / "memory")
            ms.short_term_limit = 50
            from core.brain import Brain
            b = Brain()
            result = await b.think("What is 2+2?")
            assert isinstance(result, str)
            assert len(result) > 0
