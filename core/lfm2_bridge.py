"""
core/lfm2_bridge.py
LFM2.5-1.2B-Thinking ONNX bridge - 200+ tok/s CPU/CUDA inference.

Model: liquid-ai/lfm2.5-1.2b-thinking (LiquidAI)
Backend: ONNX Runtime (onnxruntime / onnxruntime-gpu)

Key stats (via @xenovacom / LiquidAI):
  - 1.2B parameters, Liquid Foundation Model architecture
  - 200+ tok/s on WebGPU; comparable on ONNX Runtime CUDA EP
  - Thinking variant: chain-of-thought reasoning in 1.2B
  - Zero install on browser (WebGPU); minimal footprint on VPS
  - Transformers.js + ONNX Runtime Web validated

Env:
    LFM2_MODEL_PATH    path to exported ONNX model dir (default: models/lfm2)
    LFM2_N_THREADS     CPU intra-op threads (default: 4)
    LFM2_USE_GPU       true to enable CUDA EP (default: false)
    LFM2_ENABLED       false to disable

Export ONNX model (one-time):
    pip install transformers optimum onnxruntime
    optimum-cli export onnx --model liquid-ai/lfm2.5-1.2b-thinking models/lfm2/
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = os.getenv("LFM2_MODEL_PATH", "models/lfm2")
_N_THREADS  = int(os.getenv("LFM2_N_THREADS", "4"))
_USE_GPU    = os.getenv("LFM2_USE_GPU", "false").lower() == "true"
_ENABLED    = os.getenv("LFM2_ENABLED", "true").lower() != "false"


def _check_available() -> bool:
    if not _ENABLED:
        return False
    if not os.path.isdir(_MODEL_PATH):
        return False
    try:
        import onnxruntime  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
        return True
    except ImportError:
        return False


LFM2_AVAILABLE: bool = _check_available()

_THINKING_TASKS = {"reason", "analyze", "plan", "distill", "multi_step"}

_SYSTEM_PROMPTS: dict[str, str] = {
    "classify":        "Classify the input. Reply with a single label only.",
    "sentiment":       "Reply with: positive, negative, or neutral.",
    "yes_no":          "Answer yes or no only.",
    "single_label":    "Reply with a single word label.",
    "tag_infer":       "List relevant tags, comma-separated.",
    "signal":          "Classify market signal: bullish, bearish, or neutral.",
    "summarize_short": "Summarize in one sentence.",
    "translate_short": "Translate accurately and concisely.",
    "reason":          "Think step by step, then give a concise answer.",
    "analyze":         "Analyze carefully, then summarize key points.",
    "plan":            "Create a concise step-by-step plan.",
    "micro":           "Process the input and respond concisely.",
}

_DEFAULT_SYSTEM = "You are a fast, precise assistant. Be concise."


class AsyncLFM2Bridge:
    """Async ONNX Runtime bridge for LFM2.5-1.2B-Thinking."""

    def __init__(self) -> None:
        self._session: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._lock = asyncio.Lock()

    def _load(self) -> tuple:
        if self._session is not None:
            return self._session, self._tokenizer
        import onnxruntime as ort
        from transformers import AutoTokenizer
        model_file = os.path.join(_MODEL_PATH, "model.onnx")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"LFM2 ONNX model not found: {model_file}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _USE_GPU else ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = _N_THREADS
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        logger.info("[LFM2] Loading %s (GPU=%s)", _MODEL_PATH, _USE_GPU)
        self._session = ort.InferenceSession(model_file, sess_options=opts, providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
        logger.info("[LFM2] LFM2.5-1.2B ready - providers: %s", self._session.get_providers())
        return self._session, self._tokenizer

    def _infer(self, prompt: str, max_tokens: int, system: str, thinking: bool) -> str:
        import numpy as np
        session, tokenizer = self._load()
        tag = "<|think|>" if thinking else "<|assistant|>"
        full = f"<|system|>\n{system}\n<|user|>\n{prompt}\n{tag}\n"
        input_ids = tokenizer.encode(full, return_tensors="np")
        generated = input_ids.tolist()[0]
        eos_id = tokenizer.eos_token_id
        for _ in range(max_tokens):
            feed = {"input_ids": np.array([generated], dtype="int64")}
            logits = session.run(["logits"], feed)[0]
            next_id = int(logits[0, -1].argmax())
            if next_id == eos_id:
                break
            generated.append(next_id)
        new_tokens = generated[len(input_ids[0]):]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if not thinking and "<think>" in text:
            text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        return text.strip()

    async def complete(self, prompt: str, task_type: str = "micro", max_tokens: int = 256) -> str:
        thinking = task_type in _THINKING_TASKS
        system   = _SYSTEM_PROMPTS.get(task_type, _DEFAULT_SYSTEM)
        if not thinking:
            max_tokens = min(max_tokens, 32)
        async with self._lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._infer, prompt, max_tokens, system, thinking)
        return result

    async def classify(self, text: str) -> str:
        return await self.complete(text, task_type="classify", max_tokens=16)

    async def sentiment(self, text: str) -> str:
        return await self.complete(text, task_type="sentiment", max_tokens=8)

    async def signal(self, text: str) -> str:
        return await self.complete(text, task_type="signal", max_tokens=8)

    async def reason(self, text: str, max_tokens: int = 512) -> str:
        return await self.complete(text, task_type="reason", max_tokens=max_tokens)


_instance: Optional[AsyncLFM2Bridge] = None


def get_lfm2() -> AsyncLFM2Bridge:
    global _instance
    if _instance is None:
        _instance = AsyncLFM2Bridge()
    return _instance


INSTALL_HINT = """
LFM2.5-1.2B ONNX setup (one-time):
  pip install onnxruntime optimum transformers
  # GPU: pip install onnxruntime-gpu
  mkdir -p models
  optimum-cli export onnx --model liquid-ai/lfm2.5-1.2b-thinking models/lfm2/
  # Size: ~2.4GB FP32 or ~1.2GB FP16; Speed: 200+ tok/s CUDA, 30-50 CPU
"""
