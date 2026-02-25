"""
core/airllm_bridge.py
AirLLMBridge — Layer-by-layer 70B+ inference for GodLocal VPS.

Pattern from: AirLLM (lyogavin/airllm), Apache 2.0
"70B parameter models on a single 4GB GPU — no quantization, no distillation"
Shared by: @thisguyknowsai, Feb 25 2026

Problem it solves for GodLocal:
  TieredRouter FAST tier = qwen3:1.7b (VPS)
  TieredRouter FULL tier = qwen3:8b (VPS)
  ↓
  GIANT tier is MISSING — 70B+ tasks sent to expensive cloud APIs or fail.

What AirLLM enables:
  → Llama-3.1-70B on Picobot VPS (8GB VRAM) with layer-by-layer offload
  → Llama-3.1-405B on 8GB VRAM (slower but possible)
  → 4-bit/8-bit compression for 3x speed with minor quality loss
  → No cloud API. No A100. Just pip install airllm.

Architecture:
  AirLLMBridge wraps AirLLM's AirLLMLlama2 / AirLLMQWen / AirLLMGemma
  with same interface as TieredRouter.complete().
  TieredRouter auto-selects GIANT tier when task_type="giant" or
  prompt tokens > GIANT_THRESHOLD.

Usage:
    from core.airllm_bridge import AirLLMBridge, get_airllm
    bridge = get_airllm()
    response = await bridge.complete("Explain quantum entanglement in 500 words")

    # Via TieredRouter (auto-routed):
    from core.tiered_router import get_tiered_router
    router = get_tiered_router()
    result = await router.complete(prompt, task_type="giant")   # → AirLLM tier

Requirements (VPS):
    pip install airllm
    # GPU: any with 4GB+ VRAM (RTX 3060, 4060, etc.)
    # CPU: 32GB+ RAM recommended for layer cache
    # Model cached in ~/.cache/airllm/ on first run

Env vars:
    AIRLLM_MODEL      — HuggingFace model ID (default: meta-llama/Meta-Llama-3.1-70B-Instruct)
    AIRLLM_PRECISION  — "4bit" | "8bit" | "float16" (default: 4bit, 3x faster)
    AIRLLM_MAX_TOKENS — max new tokens (default: 512)
    AIRLLM_LAYER_CACHE_DIR — layer cache dir (default: ~/.cache/airllm)
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional


# ── Model registry — supported AirLLM backends ────────────────────────────────

AIRLLM_MODEL_REGISTRY: dict[str, str] = {
    # Llama
    "llama-70b":    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama-8b":     "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-405b":   "meta-llama/Meta-Llama-3.1-405B-Instruct",   # 8GB VRAM, slow
    # Qwen
    "qwen-72b":     "Qwen/Qwen2.5-72B-Instruct",
    "qwen-32b":     "Qwen/Qwen2.5-32B-Instruct",
    # Mistral
    "mistral-7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    # Default
    "default":      "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

# Map HuggingFace model IDs → AirLLM class name
AIRLLM_CLASS_MAP: dict[str, str] = {
    "llama":    "AirLLMLlama2",
    "qwen":     "AirLLMQWen",
    "mistral":  "AirLLMMistral",
    "gemma":    "AirLLMGemma",
    "baichuan": "AirLLMBaichuan",
    "chatglm":  "AirLLMChatGLM",
    "internlm": "AirLLMInternLM",
}


def _detect_airllm_class(model_id: str) -> str:
    """Auto-detect AirLLM class from model ID (mirrors AirLLM's auto-detection)."""
    model_lower = model_id.lower()
    for key, cls in AIRLLM_CLASS_MAP.items():
        if key in model_lower:
            return cls
    return "AirLLMLlama2"   # Default fallback


# ── AirLLMBridge ──────────────────────────────────────────────────────────────

class AirLLMBridge:
    """
    GodLocal wrapper for AirLLM layer-by-layer inference.

    Supports 70B+ models on 4-8GB VRAM via CPU/GPU layer offloading.
    Same interface as TieredRouter.complete() for drop-in integration.

    Key behaviour:
      - Lazy-loads model on first call (HuggingFace download ~30-60 min first time)
      - Layer cache persists across calls (fast subsequent runs)
      - Thread-safe: one model instance, asyncio.Lock guards inference
      - Degrades gracefully: if AirLLM not installed → raises ImportError with pip command
      - 4-bit default: 3x speed vs float16, negligible quality loss on instruction tasks
    """

    GIANT_TOKEN_THRESHOLD = 2000   # Route to GIANT tier if prompt > 2000 tokens

    def __init__(
        self,
        model_alias: str | None = None,
        precision: str | None = None,
        max_new_tokens: int | None = None,
        layer_cache_dir: str | None = None,
    ) -> None:
        alias = model_alias or os.getenv("AIRLLM_MODEL_ALIAS", "llama-70b")
        self._model_id = AIRLLM_MODEL_REGISTRY.get(alias, alias)   # Allows raw HF ID too
        self._precision = precision or os.getenv("AIRLLM_PRECISION", "4bit")
        self._max_new_tokens = max_new_tokens or int(os.getenv("AIRLLM_MAX_TOKENS", "512"))
        self._layer_cache_dir = layer_cache_dir or os.getenv(
            "AIRLLM_LAYER_CACHE_DIR",
            os.path.expanduser("~/.cache/airllm")
        )
        self._model = None          # Lazy-loaded
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self._airllm_class_name = _detect_airllm_class(self._model_id)
        self._loaded_at: float = 0.0

    def _ensure_airllm_installed(self) -> None:
        """Check AirLLM is installed, raise informative error if not."""
        try:
            import airllm   # noqa: F401
        except ImportError:
            raise ImportError(
                "AirLLM not installed. Run: pip install airllm
"
                "Then restart the GodLocal backend (godlocal_v5.py)."
            )

    def _load_model(self) -> None:
        """Load model with layer-by-layer offloading. Called once, cached."""
        self._ensure_airllm_installed()
        import airllm
        cls = getattr(airllm, self._airllm_class_name, None)
        if cls is None:
            # Fallback to generic auto
            from airllm import AutoModel
            cls = AutoModel
        compression = None
        if self._precision == "4bit":
            compression = "4bit"
        elif self._precision == "8bit":
            compression = "8bit"

        self._model = cls.from_pretrained(
            self._model_id,
            offload_folder=self._layer_cache_dir,
            compression=compression,
        )
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._loaded_at = time.time()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_model()

    async def complete(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> str:
        """
        Generate completion using layer-by-layer inference.
        Thread-safe via asyncio.Lock.
        Equivalent API to TieredRouter.complete().
        """
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._sync_complete, prompt, system, max_new_tokens or self._max_new_tokens
            )

    def _sync_complete(self, prompt: str, system: str, max_new_tokens: int) -> str:
        """Synchronous inference — runs in thread executor to avoid blocking event loop."""
        self._ensure_loaded()
        # Format as chat message (instruction tuning format)
        formatted = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        input_ids = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).input_ids

        output = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=False,
        )
        # Decode only newly generated tokens (skip input)
        new_tokens = output[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def stats(self) -> dict:
        return {
            "model_id":    self._model_id,
            "class":       self._airllm_class_name,
            "precision":   self._precision,
            "max_tokens":  self._max_new_tokens,
            "loaded":      self._model is not None,
            "loaded_at":   self._loaded_at,
            "cache_dir":   self._layer_cache_dir,
        }


# ── TieredRouter integration ──────────────────────────────────────────────────

def patch_tiered_router_with_giant_tier(router_instance) -> None:
    """
    Monkey-patch an existing TieredRouter instance to add GIANT tier.

    After patching:
        router.complete(prompt, task_type="giant")   → AirLLM 70B
        router.complete(long_prompt)                 → auto-routed to GIANT if > 2000 tokens

    Call this once at startup in godlocal_v5.py:
        from core.airllm_bridge import patch_tiered_router_with_giant_tier
        patch_tiered_router_with_giant_tier(get_tiered_router())
    """
    bridge = get_airllm()
    original_complete = router_instance.complete

    async def patched_complete(prompt: str, task_type: str = "auto", **kwargs):
        # Route to GIANT if explicit or prompt is very long
        token_estimate = len(prompt.split()) * 1.3   # rough token estimate
        if task_type == "giant" or (task_type == "auto" and token_estimate > AirLLMBridge.GIANT_TOKEN_THRESHOLD):
            return await bridge.complete(prompt, **kwargs)
        return await original_complete(prompt, task_type=task_type, **kwargs)

    router_instance.complete = patched_complete
    router_instance._airllm_bridge = bridge   # expose for introspection


# ── Singleton ─────────────────────────────────────────────────────────────────

_bridge: AirLLMBridge | None = None

def get_airllm(model_alias: str | None = None) -> AirLLMBridge:
    """Get or create the global AirLLMBridge singleton."""
    global _bridge
    if _bridge is None:
        _bridge = AirLLMBridge(model_alias=model_alias)
    return _bridge
