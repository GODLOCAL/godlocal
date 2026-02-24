"""
extensions/xzero/vinext_connector.py
X-ZERO Connector for Replicate API via vinext/Cloudflare Workers pattern.
Runs any Replicate model (FLUX, LLaMA, Whisper, SDXL...) from GodLocal agents.
"""
from __future__ import annotations
import os, time, httpx
from typing import Any
from .cimd_connector_base import CIMDConnectorBase


class VinextReplicateConnector(CIMDConnectorBase):
    """Run Replicate models from GodLocal via REST.

    Env:
        REPLICATE_API_TOKEN  — get from replicate.com/account/api-tokens
    """

    name = "replicate"
    base_url = "https://api.replicate.com/v1"

    # ── Featured models ─────────────────────────────────────────────────────
    MODELS = {
        "flux-2-klein":   "black-forest-labs/flux-2-klein-9b",
        "sdxl":           "stability-ai/sdxl:39ed52f2319f9260000000000000000000",
        "whisper-large":  "openai/whisper:cdd97b257f93cb89dede1c7584e3f3dfc969571b",
        "llama-3.1-8b":   "meta/meta-llama-3.1-8b-instruct",
        "qwen2.5-coder":  "deepseek-ai/deepseek-coder-v2-lite-instruct",
    }

    def __init__(self):
        self.token = os.getenv("REPLICATE_API_TOKEN", "")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    # ── CIMDConnectorBase interface ──────────────────────────────────────────
    def openapi_schema(self) -> dict:
        return {
            "openapi": "3.0.0",
            "info": {"title": "Replicate via X-ZERO", "version": "1.0"},
            "paths": {
                "/predict": {
                    "post": {
                        "summary": "Run a Replicate model",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["model"],
                                        "properties": {
                                            "model": {
                                                "type": "string",
                                                "description": f"Model alias. One of: {list(self.MODELS.keys())}",
                                            },
                                            "input": {
                                                "type": "object",
                                                "description": "Model-specific input dict (e.g. {prompt: '...'} for image models)",
                                            },
                                            "wait": {
                                                "type": "boolean",
                                                "default": True,
                                                "description": "Poll until completed (default True). False = return prediction_id.",
                                            },
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "Prediction output"}},
                    }
                },
                "/models": {
                    "get": {
                        "summary": "List available model aliases",
                        "responses": {"200": {"description": "Model list"}},
                    }
                },
            },
        }

    def registration_manifest(self) -> dict:
        return {
            "id": "replicate",
            "name": "Replicate AI Models",
            "description": "Run 100K+ AI models (image, LLM, audio, video) via Replicate API",
            "tools": ["predict", "list_models"],
            "auth": {"type": "env", "key": "REPLICATE_API_TOKEN"},
        }

    def run_tool(self, tool: str, params: dict) -> Any:
        if tool == "list_models":
            return self.MODELS
        if tool == "predict":
            return self.predict(**params)
        raise ValueError(f"Unknown tool: {tool}")

    # ── Core predict ────────────────────────────────────────────────────────
    def predict(
        self,
        model: str,
        input: dict | None = None,
        wait: bool = True,
        timeout: int = 120,
    ) -> dict:
        """Create prediction and optionally poll until done."""
        version = self.MODELS.get(model, model)  # allow full version strings too
        payload: dict = {"input": input or {}}

        # Models with explicit version string
        if ":" in version:
            model_name, ver = version.rsplit(":", 1)
            payload["version"] = ver
            r = self._client.post("/predictions", json=payload)
        else:
            # New Deployments API (no version needed for latest)
            r = self._client.post(f"/models/{version}/predictions", json=payload)

        r.raise_for_status()
        pred = r.json()

        if not wait:
            return {"prediction_id": pred["id"], "status": pred["status"]}

        return self._poll(pred["id"], timeout=timeout)

    def _poll(self, prediction_id: str, timeout: int = 120) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self._client.get(f"/predictions/{prediction_id}")
            r.raise_for_status()
            pred = r.json()
            status = pred.get("status")
            if status == "succeeded":
                return {"status": "succeeded", "output": pred.get("output"), "metrics": pred.get("metrics")}
            if status in ("failed", "canceled"):
                return {"status": status, "error": pred.get("error")}
            time.sleep(2)
        return {"status": "timeout", "prediction_id": prediction_id}
