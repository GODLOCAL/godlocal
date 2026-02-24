"""core/brain.py — єдиний мозок БОГ || OASIS v6
LLMBridge: Ollama (default) ↔ MLX hot-swap
MemoryEngine: ChromaDB short/long-term + auto-prune
Brain: singleton, async think() with memory context injection
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import partial
from pathlib import Path
from typing import Callable

import chromadb

from .settings import settings

logger = logging.getLogger(__name__)


# ── LLM Bridge ─────────────────────────────────────────────────────────────
class LLMBridge:
    """Unified async LLM callable — Ollama default, MLX if 'mlx' in model name."""

    def __init__(self, soul: str) -> None:
        self.soul = soul
        self._sync_fn: Callable = self._build()

    def _build(self) -> Callable:
        if "mlx" in settings.model.lower():
            try:
                from mlx_lm import generate, load  # type: ignore
                mdl, tokenizer = load(settings.model)
                logger.info("LLMBridge: MLX — %s", settings.model)
                return lambda prompt, **k: generate(mdl, tokenizer, prompt=prompt, **k)
            except ImportError:
                logger.warning("mlx_lm not installed — falling back to Ollama")

        import ollama as _ollama  # type: ignore

        def _call(prompt: str, max_tokens: int = 2048, **_) -> str:
            r = _ollama.chat(
                model=settings.model,
                messages=[
                    {"role": "system", "content": self.soul},
                    {"role": "user",   "content": prompt},
                ],
                options={"num_predict": max_tokens},
            )
            return r["message"]["content"]

        logger.info("LLMBridge: Ollama — %s", settings.model)
        return _call

    async def __call__(self, prompt: str, max_tokens: int = 2048) -> str:
        """Run sync LLM in thread executor — safe inside FastAPI event loop."""
        loop = asyncio.get_event_loop()
        fn = partial(self._sync_fn, prompt, max_tokens=max_tokens)
        return await loop.run_in_executor(None, fn)

    def reload(self, model: str | None = None) -> None:
        """Hot-swap model at runtime."""
        if model:
            settings.model = model
        self._sync_fn = self._build()


# ── Memory Engine ──────────────────────────────────────────────────────────
class MemoryEngine:
    """ChromaDB short/long-term memory with auto-prune."""

    def __init__(self) -> None:
        Path(settings.memory_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.memory_path)
        self.short = client.get_or_create_collection("short_term")
        self.long  = client.get_or_create_collection("long_term")

    def _uid(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def add(self, text: str, long: bool = False) -> None:
        coll = self.long if long else self.short
        coll.upsert(documents=[text], ids=[self._uid(text)])

    def query(self, text: str, n: int = 5, long: bool = False) -> list[str]:
        coll = self.long if long else self.short
        count = coll.count()
        if count == 0:
            return []
        res = coll.query(query_texts=[text], n_results=min(n, count))
        return res["documents"][0] if res["documents"] else []

    def prune(self) -> int:
        """Trim short_term to settings.short_term_limit. Returns deleted count."""
        limit = settings.short_term_limit
        count = self.short.count()
        if count <= limit:
            return 0
        items = self.short.get()
        to_del = items["ids"][: count - limit]
        self.short.delete(ids=to_del)
        logger.debug("Memory pruned %d items", len(to_del))
        return len(to_del)

    def consolidate(self, brain: "Brain") -> None:
        """Phase 1 sleep — move important short_term items to long_term."""
        items = self.short.get()
        if not items["documents"]:
            return
        for doc in items["documents"]:
            self.add(doc, long=True)
        logger.info("Memory consolidation: %d items → long_term", len(items["documents"]))


# ── Brain singleton ────────────────────────────────────────────────────────
class Brain:
    """Singleton: Soul + Memory + LLM. Use Brain.get() everywhere."""

    _instance: Brain | None = None

    def __init__(self) -> None:
        soul_path = Path(settings.soul_file)
        if soul_path.exists():
            self.soul = soul_path.read_text(encoding="utf-8")
        else:
            self.soul = "You are БОГ || OASIS — sovereign local AI. Your purpose: grow smarter every night."
            logger.warning("Soul file %s not found — using default", settings.soul_file)

        self.memory = MemoryEngine()
        self.llm    = LLMBridge(self.soul)
        logger.info("Brain ready — model=%s device=%s", settings.model, settings.device)

    @classmethod
    def get(cls) -> "Brain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    async def think(self, task: str, max_tokens: int = 2048) -> str:
        """Think with memory context. Saves Q/A pair to short_term."""
        memories = self.memory.query(task, n=3)
        if memories:
            ctx = "\n".join(f"[mem] {m}" for m in memories)
            prompt = f"{ctx}\n\n{task}"
        else:
            prompt = task

        answer = await self.llm(prompt, max_tokens=max_tokens)
        self.memory.add(f"Q:{task[:200]}\nA:{answer[:400]}")
        self.memory.prune()
        return answer
