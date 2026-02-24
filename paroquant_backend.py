"""
paroquant_backend.py — ParoQuant LLM Backend for GodLocal
══════════════════════════════════════════════════════════
Drop-in replacement for the Ollama backend, using ParoQuant 4-bit quantized
models from Hugging Face z-lab collection (ICLR 2026, MIT license).

Why ParoQuant over AWQ for GodLocal?
  • Reasoning accuracy: Qwen3-4B AWQ = 68.2 MMLU-Pro, PARO = 71.0 (+4%)
  • sleep_cycle() chains long reasoning → AWQ accumulates errors → PARO fixes this
  • Same 4-bit size (~1.8GB) but better reasoning chains in self_evolve.py
  • Fused single kernel: <10% runtime overhead vs AWQ

Recommended model for Steam Deck (16GB unified RAM, AMD RDNA2):
  → z-lab/Qwen3-4B-PARO  (4-bit, ~1.8GB, best quality/speed balance)
  → z-lab/Qwen3-8B-PARO  (4-bit, ~4GB, best reasoning quality)
  → z-lab/DeepSeek-R1-Distill-Llama-8B-PARO  (4-bit, reasoning specialist)

Integration points in GodLocal:
  1. godlocal_v5.py  → replace Ollama call in handle_message() with ParoQuantBackend.chat()
  2. self_evolve.py  → replace Ollama call with ParoQuantBackend.chat()
  3. sleep_cycle()   → Phase 4 (meta-reflection): use reasoning=True for deep analysis

Device support:
  • CUDA  (NVIDIA GPU)
  • ROCm  (AMD GPU — Steam Deck with ROCm 5.7+)
  • MPS   (Apple Silicon)
  • CPU   (fallback, slower but works — Steam Deck without ROCm)

Paper: https://arxiv.org/abs/2511.10645
Models: https://huggingface.co/collections/z-lab/paroquant
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParoQuantConfig:
    """Runtime config for the ParoQuant backend."""

    # HuggingFace model ID (all pre-quantized 4-bit ParoQuant models)
    model_id: str = "z-lab/Qwen3-4B-PARO"

    # Where to cache the downloaded model (default: ~/.cache/paroquant/)
    cache_dir: Optional[str] = None

    # Device: "auto" lets transformers pick (cuda > mps > cpu)
    device: str = "auto"

    # Generation defaults
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Reasoning mode (for sleep_cycle Phase 4 + self_evolve.py deep analysis)
    # When True: lower temperature, more tokens for chain-of-thought
    reasoning_max_tokens: int = 2048
    reasoning_temperature: float = 0.3

    @classmethod
    def for_steam_deck(cls) -> "ParoQuantConfig":
        """Optimized config for Steam Deck (AMD RDNA2, 16GB shared RAM)."""
        return cls(
            model_id="z-lab/Qwen3-4B-PARO",  # 1.8GB — fits in Steam Deck VRAM budget
            device="auto",                    # ROCm if available, else CPU
            max_new_tokens=512,
            temperature=0.7,
        )

    @classmethod
    def for_reasoning(cls) -> "ParoQuantConfig":
        """Config optimised for sleep_cycle() + self_evolve.py reasoning chains."""
        return cls(
            model_id="z-lab/Qwen3-8B-PARO",  # Better reasoning, ~4GB
            device="auto",
            max_new_tokens=2048,
            temperature=0.3,
            top_p=0.95,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────────────────────

class ParoQuantBackend:
    """
    GodLocal LLM backend using ParoQuant quantized models.

    Drop-in replacement for the Ollama API calls in:
      godlocal_v5.py  — handle_message()
      self_evolve.py  — generate_hypothesis(), evaluate_performance()
      sleep_cycle()   — Phase 4 meta-reflection

    Usage:
        backend = ParoQuantBackend()  # lazy-loads on first call
        response = backend.chat("What should I remember from today?")
        # streaming
        for chunk in backend.stream("Summarize this conversation: ..."):
            print(chunk, end="", flush=True)
    """

    def __init__(self, config: Optional[ParoQuantConfig] = None):
        self.config = config or ParoQuantConfig.for_steam_deck()
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False

    def _ensure_deps(self) -> None:
        """Check required packages are installed."""
        missing = []
        try:
            import transformers  # noqa
        except ImportError:
            missing.append("transformers>=4.45.0")
        try:
            import torch  # noqa
        except ImportError:
            missing.append("torch>=2.1.0")
        try:
            import autoawq  # noqa
        except ImportError:
            missing.append("autoawq>=0.2.7")
        if missing:
            raise ImportError(
                f"ParoQuant backend requires: {', '.join(missing)}\n"
                f"Install: pip install {' '.join(missing)}"
            )

    def _resolve_device(self) -> str:
        """Auto-detect best available device."""
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("[ParoQuant] Device: CUDA")
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("[ParoQuant] Device: MPS (Apple Silicon)")
                return "mps"
            # Check for ROCm (AMD GPU — Steam Deck)
            if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name(0):
                logger.info("[ParoQuant] Device: ROCm (AMD GPU)")
                return "cuda"  # ROCm uses CUDA API
        except Exception:
            pass
        logger.info("[ParoQuant] Device: CPU (fallback — slower on Steam Deck without ROCm)")
        return "cpu"

    def load(self) -> None:
        """Download (first time) and load the ParoQuant model."""
        if self._loaded:
            return

        self._ensure_deps()
        from transformers import AutoTokenizer

        device = self._resolve_device()
        self._device = device
        cache = self.config.cache_dir or str(Path.home() / ".cache" / "paroquant")
        os.makedirs(cache, exist_ok=True)

        logger.info(f"[ParoQuant] Loading {self.config.model_id} → {device}")
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            cache_dir=cache,
            trust_remote_code=True,
        )

        # Try AWQ loader first (ParoQuant models are AWQ-compatible with fused rotations)
        try:
            from awq import AutoAWQForCausalLM  # autoawq
            self._model = AutoAWQForCausalLM.from_quantized(
                self.config.model_id,
                cache_dir=cache,
                fuse_layers=True,   # fuse ParoQuant rotation kernel
                trust_remote_code=True,
            )
            logger.info("[ParoQuant] Loaded via AutoAWQ (fused kernel)")
        except (ImportError, Exception) as e:
            logger.warning(f"[ParoQuant] AutoAWQ failed ({e}), falling back to transformers")
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                cache_dir=cache,
                quantization_config=bnb_config,
                device_map="auto" if device != "cpu" else None,
                trust_remote_code=True,
            )
            logger.info("[ParoQuant] Loaded via transformers + bitsandbytes 4-bit fallback")

        elapsed = time.time() - t0
        logger.info(f"[ParoQuant] Model ready in {elapsed:.1f}s")
        self._loaded = True

    def _build_prompt(self, messages: list[dict]) -> str:
        """Apply chat template (Qwen3 / LLaMA instruct format)."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple system/user/assistant format
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}<|end|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}<|end|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[list[dict]] = None,
        reasoning: bool = False,
    ) -> str:
        """
        Single-turn or multi-turn chat.

        Args:
            prompt:    User message
            system:    System prompt (optional — GodLocal soul injected here)
            history:   Previous [(role, content), ...] turns
            reasoning: If True, use reasoning config (more tokens, lower temp)

        Returns:
            Model response text (stripped)

        Raises:
            RuntimeError if model not loaded
        """
        self.load()

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        for turn in (history or []):
            messages.append(turn)
        messages.append({"role": "user", "content": prompt})

        full_prompt = self._build_prompt(messages)

        import torch
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        max_tokens = self.config.reasoning_max_tokens if reasoning else self.config.max_new_tokens
        temperature = self.config.reasoning_temperature if reasoning else self.config.temperature

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        reasoning: bool = False,
    ) -> Iterator[str]:
        """
        Streaming generator — yields tokens one by one.
        Use for Telegram bot streaming responses.

        Usage:
            for token in backend.stream("Tell me about Solana"):
                # send_message chunk to Telegram
                yield token
        """
        self.load()

        try:
            from transformers import TextIteratorStreamer
            import threading
        except ImportError:
            # Fallback: return full response as single chunk
            yield self.chat(prompt, system=system, reasoning=reasoning)
            return

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        full_prompt = self._build_prompt(messages)

        import torch
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        max_tokens = self.config.reasoning_max_tokens if reasoning else self.config.max_new_tokens
        temperature = self.config.reasoning_temperature if reasoning else self.config.temperature

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        def _generate():
            with torch.no_grad():
                self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    do_sample=temperature > 0,
                    streamer=streamer,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        for token in streamer:
            yield token

    def unload(self) -> None:
        """Free model memory (useful when switching backends)."""
        if self._model is not None:
            import gc, torch  # noqa
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False
            if self._device and "cuda" in self._device:
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("[ParoQuant] Model unloaded, memory freed")


# ──────────────────────────────────────────────────────────────────────────────
# Ollama-compatible adapter
# ──────────────────────────────────────────────────────────────────────────────

class OllamaCompatAdapter:
    """
    Wraps ParoQuantBackend with Ollama-compatible API surface.
    Drop this in wherever godlocal_v5.py calls Ollama.

    Instead of:
        response = ollama.chat(model="qwen3:4b", messages=[...])
        text = response["message"]["content"]

    Use:
        response = adapter.chat(model="z-lab/Qwen3-4B-PARO", messages=[...])
        text = response["message"]["content"]
    """

    def __init__(self, config: Optional[ParoQuantConfig] = None):
        self._backend = ParoQuantBackend(config)

    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Ollama-compatible .chat() interface."""
        system = None
        history = []
        user_msg = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                user_msg = content
                history.append(m)
            elif role == "assistant":
                history.append(m)

        response_text = self._backend.chat(
            prompt=user_msg,
            system=system,
            history=history[:-1] if history else [],
        )
        return {
            "message": {"role": "assistant", "content": response_text},
            "model": model,
            "done": True,
        }


# ──────────────────────────────────────────────────────────────────────────────
# sleep_cycle() integration hook
# ──────────────────────────────────────────────────────────────────────────────

_default_backend: Optional[ParoQuantBackend] = None


def get_paroquant_backend(reasoning: bool = False) -> ParoQuantBackend:
    """
    Module-level singleton for sleep_cycle() integration.

    Usage in godlocal_v5.py SleepCycle.run() Phase 4:
        from paroquant_backend import get_paroquant_backend
        backend = get_paroquant_backend(reasoning=True)
        insight = backend.chat(
            prompt=f"Reflect on these patterns: {lessons_text}",
            system=GODLOCAL_SOUL,
            reasoning=True
        )
    """
    global _default_backend
    if _default_backend is None:
        cfg = ParoQuantConfig.for_reasoning() if reasoning else ParoQuantConfig.for_steam_deck()
        _default_backend = ParoQuantBackend(cfg)
    return _default_backend


# ──────────────────────────────────────────────────────────────────────────────
# CLI: quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "z-lab/Qwen3-4B-PARO"
    print(f"Testing ParoQuant backend: {model_id}")
    cfg = ParoQuantConfig(model_id=model_id, max_new_tokens=200)
    backend = ParoQuantBackend(cfg)
    response = backend.chat(
        prompt="What is ParoQuant and why does it improve reasoning models?",
        system="You are GodLocal, a sovereign AI assistant."
    )
    print("\n--- Response ---")
    print(response)
