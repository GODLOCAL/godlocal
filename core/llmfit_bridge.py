"""
llmfit_bridge.py — Hardware-aware model selector for GodLocal
Integrates AlexsJones/llmfit: 157 models, 30 providers, 1 command.
https://github.com/AlexsJones/llmfit

Usage:
    from core.llmfit_bridge import detect_best_backend, get_model_recommendations
    backend = detect_best_backend()          # "mlx" | "ollama" | "cpu"
    models  = get_model_recommendations()   # sorted list of runnable models
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Backend = Literal["mlx", "ollama", "cpu"]


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    has_apple_silicon: bool = False
    has_cuda: bool = False
    has_rocm: bool = False
    recommended_backend: Backend = "cpu"
    recommended_models: list[str] = field(default_factory=list)


def _run_llmfit() -> dict:
    """Run `llmfit detect` and parse JSON output. Returns empty dict on failure."""
    try:
        result = subprocess.run(
            ["llmfit", "detect", "--json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug("llmfit not available: %s", e)
    return {}


def _detect_hardware_fallback() -> HardwareProfile:
    """Manual detection when llmfit is not installed."""
    import platform, psutil
    profile = HardwareProfile()
    profile.ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    # Apple Silicon
    if platform.system() == "Darwin":
        try:
            cpu_info = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            ).stdout
            if "Apple" in cpu_info:
                profile.has_apple_silicon = True
        except Exception:
            pass

    # CUDA
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        profile.has_cuda = True
    except Exception:
        pass

    # ROCm
    try:
        import torch
        if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name(0):
            profile.has_rocm = True
    except Exception:
        pass

    # Select backend
    if profile.has_apple_silicon:
        profile.recommended_backend = "mlx"
    elif profile.has_cuda or profile.has_rocm:
        profile.recommended_backend = "ollama"
    else:
        profile.recommended_backend = "ollama" if profile.ram_gb >= 8 else "cpu"

    # Model recommendations by RAM
    if profile.ram_gb >= 12:
        profile.recommended_models = ["LFM2-24B-A2B", "qwen3:8b", "llama3.1:8b"]
    elif profile.ram_gb >= 8:
        profile.recommended_models = ["paro4b", "qwen3:4b", "phi3:mini"]
    else:
        profile.recommended_models = ["phi3:mini", "tinyllama"]

    return profile


def detect_best_backend() -> Backend:
    """
    Detect the optimal LLM backend for current hardware.
    Uses llmfit if available, falls back to manual detection.

    Returns: "mlx" | "ollama" | "cpu"
    """
    llmfit_data = _run_llmfit()
    if llmfit_data:
        # Parse llmfit JSON output
        hw = llmfit_data.get("hardware", {})
        has_apple = hw.get("apple_silicon", False)
        has_cuda  = hw.get("cuda", False)
        ram_gb    = hw.get("ram_gb", 0)
        logger.info("llmfit detected — RAM: %.1fGB, Apple Silicon: %s, CUDA: %s",
                    ram_gb, has_apple, has_cuda)
        if has_apple:
            return "mlx"
        if has_cuda:
            return "ollama"
        return "ollama" if ram_gb >= 8 else "cpu"

    profile = _detect_hardware_fallback()
    logger.info(
        "Hardware (fallback) — RAM: %.1fGB, Apple Silicon: %s, CUDA: %s → backend: %s",
        profile.ram_gb, profile.has_apple_silicon, profile.has_cuda,
        profile.recommended_backend
    )
    return profile.recommended_backend


def get_model_recommendations() -> list[str]:
    """
    Returns a ranked list of models that fit the current hardware.
    Uses llmfit if available, otherwise uses RAM-based heuristics.
    """
    llmfit_data = _run_llmfit()
    if llmfit_data:
        models = llmfit_data.get("recommended_models", [])
        if models:
            logger.info("llmfit recommends: %s", models[:5])
            return models

    profile = _detect_hardware_fallback()
    return profile.recommended_models


def auto_select_model(preferred: str | None = None) -> str:
    """
    Select the best available model. Returns `preferred` if it fits hardware,
    otherwise falls back to the top llmfit recommendation.
    """
    recommendations = get_model_recommendations()
    if preferred and preferred in recommendations:
        return preferred
    if recommendations:
        selected = recommendations[0]
        logger.info("llmfit auto-selected model: %s", selected)
        return selected
    return preferred or "qwen3:8b"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backend = detect_best_backend()
    models  = get_model_recommendations()
    print(f"Backend : {backend}")
    print(f"Models  : {models}")
    print(f"Selected: {auto_select_model()}")
