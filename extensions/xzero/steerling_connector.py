"""
extensions/xzero/steerling_connector.py — Steerling-8B Interpretable LLM Connector
Guide Labs Steerling-8B: first large-scale inherently interpretable language model.
Every token traceable to: Input Context | Training Data | Human-understandable concepts.

Key capability: self-monitor for memorized content, suppress at inference time (no retraining).
This connector surfaces interpretability data alongside completions — reason traces exposed.

Source: https://guidelabs.ai/post/steerling-8b-base-model-release/
        https://github.com/guidelabs/steerling
        https://huggingface.co/guidelabs/steerling-8b
Inspired by: @guidelabsai tweet 2026-02-25

Usage:
    conn = get_connector("steerling")()
    result = await conn.run_tool("steerling_complete", {
        "prompt": "What is the capital of France?",
        "return_trace": True
    })
    # result["completion"]    — text output
    # result["trace"]         — interpretability trace (context / training / concept attributions)
    # result["memorized"]     — True if output was suppressed as memorized content
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .cimd_connector_base import CIMDConnectorBase

logger = logging.getLogger(__name__)

HF_MODEL_ID = "guidelabs/steerling-8b"
HF_API_BASE = "https://api-inference.huggingface.co/models"
GUIDELABS_API_BASE = "https://api.guidelabs.ai/v1"


class SteerlingConnector(CIMDConnectorBase):
    """Steerling-8B: inherently interpretable LLM connector.

    Every completion optionally returns an interpretability trace showing:
    - context_attribution: which input tokens influenced the output
    - training_attribution: training data provenance signals
    - concept_attribution: human-understandable concept activations
    - memorized: whether memorized content was detected and suppressed
    """

    name = "steerling"
    version = "1.0.0"
    description = "Steerling-8B interpretable LLM — trace every token to context/training/concepts"

    def __init__(self) -> None:
        super().__init__()
        # Prefer GuideLabsAI API key; fall back to HuggingFace Inference API
        self._guidelabs_key = os.getenv("GUIDELABS_API_KEY", "")
        self._hf_key = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_API_KEY", ""))
        if not self._guidelabs_key and not self._hf_key:
            logger.warning(
                "SteerlingConnector: no API key found. "
                "Set GUIDELABS_API_KEY or HF_TOKEN. "
                "Running in offline/Ollama mode."
            )

    # ─── CIMD interface ────────────────────────────────────────────────────────

    def openapi_schema(self) -> dict:
        return {
            "openapi": "3.0.0",
            "info": {"title": "Steerling-8B Connector", "version": self.version},
            "paths": {
                "/steerling/complete": {
                    "post": {
                        "operationId": "steerling_complete",
                        "summary": "Generate completion with full interpretability trace",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["prompt"],
                                        "properties": {
                                            "prompt": {"type": "string"},
                                            "max_tokens": {"type": "integer", "default": 512},
                                            "temperature": {"type": "number", "default": 0.7},
                                            "return_trace": {"type": "boolean", "default": True},
                                            "suppress_memorized": {"type": "boolean", "default": True},
                                        },
                                    }
                                }
                            },
                        },
                    }
                },
                "/steerling/trace": {
                    "post": {
                        "operationId": "steerling_trace",
                        "summary": "Trace attribution for a given prompt+completion pair",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["prompt", "completion"],
                                        "properties": {
                                            "prompt": {"type": "string"},
                                            "completion": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    }
                },
                "/steerling/concepts": {
                    "get": {
                        "operationId": "steerling_list_concepts",
                        "summary": "List human-understandable concept dimensions Steerling tracks",
                    }
                },
                "/steerling/memorized_check": {
                    "post": {
                        "operationId": "steerling_memorized_check",
                        "summary": "Check if a completion contains suppressed memorized content",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["text"],
                                        "properties": {"text": {"type": "string"}},
                                    }
                                }
                            },
                        },
                    }
                },
            },
        }

    def registration_manifest(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tools": [
                "steerling_complete",
                "steerling_trace",
                "steerling_list_concepts",
                "steerling_memorized_check",
            ],
            "requires_env": ["GUIDELABS_API_KEY or HF_TOKEN (optional — falls back to Ollama)"],
            "tags": ["llm", "interpretable", "xai", "steerling", "guidelabs"],
        }

    async def run_tool(self, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        """Route tool call to the appropriate method."""
        dispatch = {
            "steerling_complete": self._complete,
            "steerling_trace": self._trace,
            "steerling_list_concepts": self._list_concepts,
            "steerling_memorized_check": self._memorized_check,
        }
        if tool not in dispatch:
            raise ValueError(f"Unknown tool: {tool!r}. Available: {list(dispatch)}")
        return await dispatch[tool](params)

    # ─── Tools ────────────────────────────────────────────────────────────────

    async def _complete(self, params: dict) -> dict:
        """Generate completion with optional interpretability trace.

        Priority:
        1. GuideLabsAI API (native interpretability JSON)
        2. HuggingFace Inference API (standard generation, mock trace)
        3. Ollama local (steerling:8b tag if available, else qwen3:8b fallback)
        """
        prompt = params["prompt"]
        max_tokens = int(params.get("max_tokens", 512))
        temperature = float(params.get("temperature", 0.7))
        return_trace = bool(params.get("return_trace", True))
        suppress_memorized = bool(params.get("suppress_memorized", True))

        # 1) GuideLabsAI native API
        if self._guidelabs_key:
            return await self._complete_guidelabs(
                prompt, max_tokens, temperature, return_trace, suppress_memorized
            )

        # 2) HuggingFace Inference API
        if self._hf_key:
            return await self._complete_hf(prompt, max_tokens, temperature)

        # 3) Local Ollama fallback
        return await self._complete_ollama(prompt, max_tokens, temperature)

    async def _complete_guidelabs(
        self, prompt: str, max_tokens: int, temperature: float,
        return_trace: bool, suppress_memorized: bool
    ) -> dict:
        """Call GuideLabsAI native API — returns full interpretability trace."""
        import aiohttp
        headers = {
            "Authorization": f"Bearer {self._guidelabs_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "steerling-8b",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "return_trace": return_trace,
            "suppress_memorized": suppress_memorized,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{GUIDELABS_API_BASE}/completions", headers=headers, json=body, timeout=60
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return {
            "completion": data.get("text", ""),
            "trace": data.get("trace", {}),
            "memorized": data.get("memorized_suppressed", False),
            "model": "steerling-8b (guidelabs)",
            "backend": "guidelabs_api",
        }

    async def _complete_hf(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Call HuggingFace Inference API for Steerling-8B."""
        import aiohttp
        headers = {"Authorization": f"Bearer {self._hf_key}"}
        body = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
        }
        url = f"{HF_API_BASE}/{HF_MODEL_ID}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body, timeout=120) as resp:
                resp.raise_for_status()
                data = await resp.json()
        text = data[0]["generated_text"] if isinstance(data, list) else str(data)
        # Strip echoed prompt if HF returns full sequence
        if text.startswith(prompt):
            text = text[len(prompt):]
        return {
            "completion": text.strip(),
            "trace": {"note": "Full trace requires GUIDELABS_API_KEY"},
            "memorized": False,
            "model": HF_MODEL_ID,
            "backend": "huggingface_inference",
        }

    async def _complete_ollama(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Fallback: local Ollama. Tries steerling:8b, falls back to qwen3:8b."""
        import aiohttp
        for model_tag in ["steerling:8b", "qwen3:8b"]:
            try:
                body = {
                    "model": model_tag,
                    "prompt": prompt,
                    "options": {"num_predict": max_tokens, "temperature": temperature},
                    "stream": False,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:11434/api/generate", json=body, timeout=120
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return {
                                "completion": data.get("response", ""),
                                "trace": {"note": "Interpretability trace requires GUIDELABS_API_KEY or HF_TOKEN"},
                                "memorized": False,
                                "model": model_tag,
                                "backend": "ollama_local",
                            }
            except Exception as e:
                logger.debug("Ollama %s failed: %s", model_tag, e)
        raise RuntimeError(
            "SteerlingConnector: all backends failed. "
            "Set GUIDELABS_API_KEY, HF_TOKEN, or run Ollama locally."
        )

    async def _trace(self, params: dict) -> dict:
        """Attribute an existing prompt+completion pair."""
        if not self._guidelabs_key:
            return {
                "error": "Full trace requires GUIDELABS_API_KEY",
                "hint": "Sign up at https://guidelabs.ai/post/steerling-8b-base-model-release/",
            }
        import aiohttp
        headers = {"Authorization": f"Bearer {self._guidelabs_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{GUIDELABS_API_BASE}/trace",
                headers=headers,
                json={"prompt": params["prompt"], "completion": params["completion"]},
                timeout=60,
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _list_concepts(self, _params: dict) -> dict:
        """Return known human-understandable concept dimensions Steerling tracks."""
        return {
            "concepts": [
                "factual_recall",
                "logical_reasoning",
                "emotional_tone",
                "uncertainty_expression",
                "instruction_following",
                "context_grounding",
                "training_data_recitation",
                "creative_generation",
                "safety_alignment",
                "domain_expertise",
            ],
            "source": "https://guidelabs.ai",
            "note": "Full concept vocabulary available via GUIDELABS_API_KEY",
        }

    async def _memorized_check(self, params: dict) -> dict:
        """Check if text is flagged as memorized training content."""
        if not self._guidelabs_key:
            return {
                "memorized": False,
                "confidence": 0.0,
                "note": "Memorization detection requires GUIDELABS_API_KEY",
            }
        import aiohttp
        headers = {"Authorization": f"Bearer {self._guidelabs_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{GUIDELABS_API_BASE}/memorized",
                headers=headers,
                json={"text": params["text"]},
                timeout=30,
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
