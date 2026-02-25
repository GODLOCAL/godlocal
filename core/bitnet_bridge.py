"""
core/bitnet_bridge.py
BitNet b1.58 2B bridge — CPU-native 1.58-bit inference.

Model: microsoft/bitnet-b1.58-2B-4T (HuggingFace)
Backend: llama.cpp GGUF (llama-cpp-python)

Key stats (from @0x0SojalSec / Microsoft):
  - 0.4 GB RAM  (vs 2 GB for LLaMA 1.5B)
  - 40% faster token generation
  - W1.58A8: 1.58-bit weights, 8-bit activations
  - Outperforms LLaMA, ~= Qwen 2.5 1.5B quality
  - Runs on any CPU: Apple M2, VPS x86, iPhone-class ARM

Usage:
    from core.bitnet_bridge import get_bitnet, BITNET_AVAILABLE
    if BITNET_AVAILABLE:
        result = await get_bitnet().complete(prompt, task_type="classify")

Env:
    BITNET_MODEL_PATH  — path to .gguf file (default: models/bitnet-b1.58-2B.gguf)
    BITNET_N_THREADS   — CPU threads (default: 4)
    BITNET_ENABLED     — set to "false" to disable (default: enabled if model found)
"""
from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Model path & availability
# --------------------------------------------------------------------------- #
_MODEL_PATH = os.getenv(
    "BITNET_MODEL_PATH",
    "models/bitnet-b1.58-2B.gguf",
)
_N_THREADS = int(os.getenv("BITNET_N_THREADS", "4"))
_ENABLED = os.getenv("BITNET_ENABLED", "true").lower() != "false"

# Model is available if the gguf file exists AND llama-cpp-python is installed
def _check_available() -> bool:
    if not _ENABLED:
        return False
    if not os.path.exists(_MODEL_PATH):
        return False
    try:
        from llama_cpp import Llama  # noqa: F401
        return True
    except ImportError:
        return False

BITNET_AVAILABLE: bool = _check_available()


# --------------------------------------------------------------------------- #
# Task-type → prompt template (keep prompts short — MICRO tier only)
# --------------------------------------------------------------------------- #
_TASK_SYSTEM_PROMPTS: dict[str, str] = {
    "classify":      "Classify the input. Reply with a single label only.",
    "summarize_short": "Summarize in one sentence.",
    "sentiment":     "Reply with: positive, negative, or neutral.",
    "tag_infer":     "List relevant tags, comma-separated.",
    "yes_no":        "Answer yes or no only.",
    "single_label":  "Reply with a single word label.",
    "translate_short": "Translate the text accurately.",
    "signal":        "Classify market signal: bullish, bearish, or neutral.",
    "micro":         "Process the input and respond concisely.",
}

_DEFAULT_SYSTEM = "You are a fast, precise assistant. Be concise."


# --------------------------------------------------------------------------- #
# Bridge
# --------------------------------------------------------------------------- #
class AsyncBitNetBridge:
    """
    Async wrapper around llama-cpp-python for BitNet b1.58 GGUF models.

    Runs inference in a thread pool to avoid blocking the event loop.
    Single asyncio.Lock prevents concurrent loads (model is ~0.4 GB, fits
    comfortably in memory — no layer offloading needed).
    """

    def __init__(self) -> None:
        self._llm: Optional[object] = None
        self._lock = asyncio.Lock()

    def _load(self) -> object:
        """Load model synchronously (called once, inside thread pool)."""
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama
        logger.info("[BitNet] Loading %s on %d threads …", _MODEL_PATH, _N_THREADS)
        self._llm = Llama(
            model_path=_MODEL_PATH,
            n_ctx=2048,
            n_threads=_N_THREADS,
            n_gpu_layers=0,  # CPU only — BitNet W1.58A8 kernel path
            verbose=False,
        )
        logger.info("[BitNet] Model loaded — 0.4 GB, CPU-native W1.58A8")
        return self._llm

    def _infer(self, prompt: str, max_tokens: int, system: str) -> str:
        """Blocking inference (runs in thread executor)."""
        llm = self._load()
        full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        out = llm(
            full_prompt,
            max_tokens=max_tokens,
            stop=["<|user|>", "<|system|>", "\n\n"],
            echo=False,
        )
        text: str = out["choices"][0]["text"]
        return text.strip()

    async def complete(
        self,
        prompt: str,
        task_type: str = "micro",
        max_tokens: int = 128,
    ) -> str:
        """
        Async completion via BitNet b1.58 2B.

        Designed for MICRO tier: short prompts, fast classification,
        signal tagging, yes/no, sentiment. Max 128 tokens by default.
        """
        system = _TASK_SYSTEM_PROMPTS.get(task_type, _DEFAULT_SYSTEM)
        async with self._lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._infer, prompt, max_tokens, system
            )
        return result

    async def classify(self, text: str) -> str:
        """Convenience: one-shot classification."""
        return await self.complete(text, task_type="classify", max_tokens=16)

    async def sentiment(self, text: str) -> str:
        """Convenience: positive / negative / neutral."""
        return await self.complete(text, task_type="sentiment", max_tokens=8)

    async def signal(self, text: str) -> str:
        """Convenience: bullish / bearish / neutral market signal."""
        return await self.complete(text, task_type="signal", max_tokens=8)


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #
_instance: Optional[AsyncBitNetBridge] = None


def get_bitnet() -> AsyncBitNetBridge:
    """Get or create the shared BitNet bridge instance."""
    global _instance
    if _instance is None:
        _instance = AsyncBitNetBridge()
    return _instance


# --------------------------------------------------------------------------- #
# Install helper
# --------------------------------------------------------------------------- #
INSTALL_HINT = """
BitNet setup (one-time):

  # 1. Install llama-cpp-python (CPU build)
  pip install llama-cpp-python

  # 2. Download BitNet b1.58 2B GGUF (0.4 GB)
  mkdir -p models
  huggingface-cli download \\
      microsoft/bitnet-b1.58-2B-4T \\
      --include "*.gguf" \\
      --local-dir models/ \\
      --local-dir-use-symlinks False

  # Rename to expected path:
  mv models/*.gguf models/bitnet-b1.58-2B.gguf

  # Or set custom path:
  export BITNET_MODEL_PATH=/path/to/model.gguf
"""
