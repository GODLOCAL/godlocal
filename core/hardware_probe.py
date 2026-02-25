"""
core/hardware_probe.py
Hardware-aware model scorer — inspired by llmfit (github.com/AlexsJones/llmfit).

llmfit pattern (@hasantoxr tweet, 2026-02-25):
  CLI tool: detects RAM/CPU/GPU → scores 206 models → tells you exactly what runs.
  "No benchmark hunting. No Reddit threads. No OOM crashes mid-generation."

GodLocal adaptation:
  - Runs at startup (one-time, <100ms)
  - Detects: RAM, CPU cores, VRAM (CUDA/Metal), disk free
  - Scores GodLocal model roster against detected hardware
  - Returns HardwareProfile + ModelFitReport (what's runnable + recommended tier)
  - TieredRouter calls probe_and_configure() once at init to disable unavailable tiers
  - Also used by setup scripts and `/status/warrior` endpoint

Scoring formula (adapted from llmfit):
  fit_score = min(available_ram / required_ram, 2.0)   # 1.0 = exact fit, >1 = headroom
  speed_score = cpu_cores / 4                           # normalized to 4-core baseline
  overall = 0.6 * fit_score + 0.4 * speed_score         # quality-weighted

Model roster (GodLocal specific):
  MICRO tier  — BitNet b1.58 2B  (0.4 GB),  LFM2.5-1.2B (2.4 GB FP32 / 1.2 GB FP16)
  FAST tier   — Qwen3-1.7B Ollama (1.2 GB), Qwen3-8B Ollama (5 GB)
  FULL tier   — Qwen3-32B Ollama (20 GB),   llama3.3-70B (via AirLLM layer-offload)
  GIANT tier  — AirLLM 70B+ (requires 4GB+ VRAM or 64GB+ RAM)

Usage:
  from core.hardware_probe import get_hardware_probe
  probe = get_hardware_probe()
  report = probe.scan()
  print(report.summary())
  # -> "RAM 16.0GB | VRAM 8.0GB | Runnable: bitnet, lfm2, qwen3-1.7b | Tier ceiling: FAST"

  # Auto-configure TieredRouter
  probe.configure_router()
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model catalog (GodLocal roster)
# --------------------------------------------------------------------------- #

class ModelTier(str, Enum):
    MICRO = "micro"
    FAST  = "fast"
    FULL  = "full"
    GIANT = "giant"


@dataclass
class ModelSpec:
    id:           str          # internal key
    name:         str          # human label
    tier:         ModelTier
    ram_gb:       float        # minimum RAM required
    vram_gb:      float = 0.0  # minimum VRAM (0 = CPU only)
    disk_gb:      float = 0.0  # download size
    description:  str  = ""
    install_hint: str  = ""


MODEL_CATALOG: list[ModelSpec] = [
    # MICRO — local CPU, always free
    ModelSpec(
        id="bitnet",
        name="BitNet b1.58 2B",
        tier=ModelTier.MICRO,
        ram_gb=0.5,
        disk_gb=0.4,
        description="1.58-bit quantized, 0.4GB GGUF, 40% faster than INT4 on CPU",
        install_hint="pip install llama-cpp-python && huggingface-cli download microsoft/bitnet-b1.58-2B-4T --include '*.gguf' --local-dir models/",
    ),
    ModelSpec(
        id="lfm2-fp16",
        name="LFM2.5-1.2B Thinking (FP16 ONNX)",
        tier=ModelTier.MICRO,
        ram_gb=1.5,
        vram_gb=1.5,  # GPU mode; CPU mode needs ~1.5GB RAM too
        disk_gb=1.2,
        description="LiquidAI ONNX, 200+ tok/s on GPU; thinking mode for chain-of-thought",
        install_hint="optimum-cli export onnx --model liquid-ai/lfm2.5-1.2b-thinking models/lfm2/",
    ),
    ModelSpec(
        id="lfm2-fp32",
        name="LFM2.5-1.2B Thinking (FP32 ONNX)",
        tier=ModelTier.MICRO,
        ram_gb=2.8,
        disk_gb=2.4,
        description="LiquidAI ONNX FP32 — higher precision, needs more RAM",
        install_hint="optimum-cli export onnx --model liquid-ai/lfm2.5-1.2b-thinking --dtype fp32 models/lfm2/",
    ),
    # FAST — small Ollama models
    ModelSpec(
        id="qwen3-1.7b",
        name="Qwen3-1.7B (Ollama)",
        tier=ModelTier.FAST,
        ram_gb=1.5,
        disk_gb=1.2,
        description="Fastest Ollama model; good for classify/translate/short-gen",
        install_hint="ollama pull qwen3:1.7b",
    ),
    ModelSpec(
        id="qwen3-8b",
        name="Qwen3-8B (Ollama)",
        tier=ModelTier.FAST,
        ram_gb=5.5,
        disk_gb=5.0,
        description="Good quality/speed balance; plan/analyze tasks",
        install_hint="ollama pull qwen3:8b",
    ),
    ModelSpec(
        id="llama3.1-8b",
        name="Llama 3.1-8B (Ollama)",
        tier=ModelTier.FAST,
        ram_gb=5.5,
        disk_gb=4.7,
        description="Meta Llama 3.1 instruction; reliable FAST fallback",
        install_hint="ollama pull llama3.1:8b",
    ),
    # FULL — larger Ollama models
    ModelSpec(
        id="qwen3-32b",
        name="Qwen3-32B (Ollama)",
        tier=ModelTier.FULL,
        ram_gb=22.0,
        disk_gb=20.0,
        description="High quality codegen + reasoning; requires 24GB+ RAM",
        install_hint="ollama pull qwen3:32b",
    ),
    ModelSpec(
        id="llama3.3-70b-q4",
        name="Llama 3.3 70B Q4 (Ollama)",
        tier=ModelTier.FULL,
        ram_gb=42.0,
        disk_gb=40.0,
        description="70B Q4 quantized; requires 48GB RAM or split CPU/GPU",
        install_hint="ollama pull llama3.3:70b",
    ),
    # GIANT — AirLLM layer-offload
    ModelSpec(
        id="airllm-70b",
        name="AirLLM 70B (layer offload)",
        tier=ModelTier.GIANT,
        ram_gb=8.0,   # AirLLM only needs ~8GB RAM (loads layers sequentially)
        vram_gb=4.0,  # 4GB VRAM minimum for layer-offload speed
        disk_gb=40.0,
        description="AirLLM layer-by-layer offload; slow but fits 4GB VRAM",
        install_hint="pip install airllm",
    ),
]


# --------------------------------------------------------------------------- #
# Hardware detection
# --------------------------------------------------------------------------- #

@dataclass
class HardwareProfile:
    ram_total_gb:  float
    ram_free_gb:   float
    cpu_cores:     int
    cpu_model:     str
    vram_gb:       float         # 0.0 if no dedicated GPU
    gpu_model:     str           # "" if none
    disk_free_gb:  float
    platform:      str           # linux / darwin / win32
    is_apple_silicon: bool = False
    cuda_available:   bool = False
    metal_available:  bool = False

    def __str__(self) -> str:
        gpu_info = f" | GPU {self.gpu_model} {self.vram_gb:.1f}GB" if self.vram_gb > 0 else ""
        accel = []
        if self.cuda_available:   accel.append("CUDA")
        if self.metal_available:  accel.append("Metal")
        if self.is_apple_silicon: accel.append("ANE")
        accel_str = f" [{'+'.join(accel)}]" if accel else ""
        return (
            f"RAM {self.ram_total_gb:.1f}GB (free {self.ram_free_gb:.1f}GB) | "
            f"CPU {self.cpu_cores}c {self.cpu_model}{gpu_info}{accel_str} | "
            f"Disk free {self.disk_free_gb:.1f}GB"
        )


@dataclass
class ModelFitScore:
    model:       ModelSpec
    fit_score:   float   # 0..2, 1.0 = exact fit
    speed_score: float   # 0..2
    overall:     float   # 0..1
    runnable:    bool
    reason:      str     # why not runnable (if applicable)
    installed:   bool = False   # model files found on disk


@dataclass
class ModelFitReport:
    hardware:    HardwareProfile
    scores:      list[ModelFitScore] = field(default_factory=list)

    @property
    def runnable(self) -> list[ModelFitScore]:
        return [s for s in self.scores if s.runnable]

    @property
    def installed_runnable(self) -> list[ModelFitScore]:
        return [s for s in self.runnable if s.installed]

    @property
    def tier_ceiling(self) -> ModelTier:
        """Highest tier with at least one runnable model."""
        order = [ModelTier.GIANT, ModelTier.FULL, ModelTier.FAST, ModelTier.MICRO]
        for tier in order:
            if any(s.tier == tier for s in self.runnable):
                return tier
        return ModelTier.MICRO

    def summary(self) -> str:
        hw = str(self.hardware)
        runnable_ids = [s.model.id for s in self.runnable]
        installed_ids = [s.model.id for s in self.installed_runnable]
        return (
            f"{hw}\n"
            f"Runnable: {', '.join(runnable_ids) or 'none'}\n"
            f"Installed: {', '.join(installed_ids) or 'none'}\n"
            f"Tier ceiling: {self.tier_ceiling.value.upper()}"
        )

    def recommendations(self) -> list[ModelFitScore]:
        """Top-1 per tier, sorted by overall score."""
        best: dict[ModelTier, ModelFitScore] = {}
        for s in sorted(self.runnable, key=lambda x: x.overall, reverse=True):
            if s.model.tier not in best:
                best[s.model.tier] = s
        return list(best.values())


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #

class HardwareProbe:
    """
    One-shot hardware scanner + model scorer.

    Inspired by llmfit scoring formula:
      fit_score  = min(available_ram / required_ram, 2.0)
      speed_score = cpu_cores / 4
      overall    = 0.6 * fit_score + 0.4 * speed_score

    Extensions for GodLocal:
      - VRAM check (CUDA / Metal) for GPU-accelerated MICRO models
      - Disk space check (download feasibility)
      - Installation check (model files present on disk)
      - Apple Silicon detection (ANE / CoreML / Metal)
    """

    def __init__(self) -> None:
        self._profile: Optional[HardwareProfile] = None
        self._report:  Optional[ModelFitReport]  = None

    def scan(self, force: bool = False) -> ModelFitReport:
        """Run hardware detection + model scoring. Cached after first call."""
        if self._report and not force:
            return self._report
        hw = self._detect_hardware()
        self._profile = hw
        scores = [self._score_model(m, hw) for m in MODEL_CATALOG]
        scores.sort(key=lambda s: s.overall, reverse=True)
        self._report = ModelFitReport(hardware=hw, scores=scores)
        logger.info("[HardwareProbe] %s", self._report.summary())
        return self._report

    def configure_router(self) -> None:
        """
        Probe hardware and set env flags to disable unavailable tiers/models.

        Called once at TieredRouter init. Prevents OOM by not even attempting
        models that won't fit in RAM/VRAM.
        """
        report = self.scan()
        runnable_ids = {s.model.id for s in report.runnable}

        # Disable BitNet if won't fit
        if "bitnet" not in runnable_ids:
            os.environ.setdefault("BITNET_ENABLED", "false")
            logger.info("[HardwareProbe] BitNet disabled — insufficient RAM")

        # Disable LFM2 if won't fit
        if "lfm2-fp16" not in runnable_ids and "lfm2-fp32" not in runnable_ids:
            os.environ.setdefault("LFM2_ENABLED", "false")
            logger.info("[HardwareProbe] LFM2 disabled — insufficient RAM/VRAM")

        # Disable AirLLM GIANT if won't fit
        if "airllm-70b" not in runnable_ids:
            os.environ.setdefault("AIRLLM_ENABLED", "false")
            logger.info("[HardwareProbe] AirLLM disabled — insufficient RAM/VRAM/disk")

        ceiling = report.tier_ceiling
        logger.info(
            "[HardwareProbe] Tier ceiling: %s | Runnable: %s",
            ceiling.value.upper(),
            ", ".join(runnable_ids) or "none",
        )

    # ------------------------------------------------------------------ #
    # Hardware detection
    # ------------------------------------------------------------------ #

    def _detect_hardware(self) -> HardwareProfile:
        import platform
        plat = platform.system().lower()

        # RAM
        ram_total, ram_free = self._get_ram_gb()

        # CPU
        cpu_cores = os.cpu_count() or 1
        cpu_model = self._get_cpu_model()

        # VRAM + GPU
        vram_gb, gpu_model, cuda_avail = self._get_gpu_info()

        # Apple Silicon
        is_apple_silicon = (plat == "darwin" and platform.machine() == "arm64")
        metal_avail = is_apple_silicon

        # Unified memory on Apple Silicon counts as both RAM and VRAM
        if is_apple_silicon and vram_gb == 0:
            vram_gb = ram_total * 0.75  # ~75% accessible to GPU on M-series

        # Disk
        disk_free_gb = self._get_disk_free_gb()

        return HardwareProfile(
            ram_total_gb=ram_total,
            ram_free_gb=ram_free,
            cpu_cores=cpu_cores,
            cpu_model=cpu_model,
            vram_gb=vram_gb,
            gpu_model=gpu_model,
            disk_free_gb=disk_free_gb,
            platform=plat,
            is_apple_silicon=is_apple_silicon,
            cuda_available=cuda_avail,
            metal_available=metal_avail,
        )

    @staticmethod
    def _get_ram_gb() -> tuple[float, float]:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.total / 1e9, vm.available / 1e9
        except ImportError:
            pass
        # Fallback: read /proc/meminfo on Linux
        try:
            mem = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":")
                    mem[k.strip()] = int(v.strip().split()[0]) * 1024  # kB -> bytes
            total = mem.get("MemTotal", 0) / 1e9
            free  = mem.get("MemAvailable", 0) / 1e9
            return total, free
        except Exception:
            pass
        return 8.0, 4.0  # conservative default

    @staticmethod
    def _get_cpu_model() -> str:
        try:
            import platform
            return platform.processor() or platform.machine()
        except Exception:
            return "unknown"

    @staticmethod
    def _get_gpu_info() -> tuple[float, str, bool]:
        """Returns (vram_gb, gpu_model_name, cuda_available)."""
        # Try CUDA via torch (optional)
        try:
            import torch
            if torch.cuda.is_available():
                idx   = torch.cuda.current_device()
                name  = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                vram  = props.total_memory / 1e9
                return vram, name, True
        except ImportError:
            pass

        # Try nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=3, stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                parts = out.split(",")
                name  = parts[0].strip()
                vram  = float(parts[1].strip()) / 1024  # MiB -> GB
                return vram, name, True
        except Exception:
            pass

        return 0.0, "", False

    @staticmethod
    def _get_disk_free_gb() -> float:
        try:
            stat = shutil.disk_usage(".")
            return stat.free / 1e9
        except Exception:
            return 50.0

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _score_model(self, m: ModelSpec, hw: HardwareProfile) -> ModelFitScore:
        """
        Score a model against detected hardware.
        Adapted from llmfit's scoring formula.
        """
        # --- Runnability checks ---
        # Use free RAM for fitting (not total — OS + other procs take ~1-2GB)
        effective_ram = max(hw.ram_free_gb, hw.ram_total_gb * 0.6)

        if m.vram_gb > 0 and not hw.cuda_available and not hw.metal_available:
            # GPU model, no GPU — check if it can run on CPU instead
            cpu_fallback_ram = m.ram_gb * 1.5  # needs more RAM without GPU
            if effective_ram < cpu_fallback_ram:
                return ModelFitScore(
                    model=m, fit_score=0, speed_score=0, overall=0,
                    runnable=False,
                    reason=f"Needs {m.vram_gb:.1f}GB VRAM or {cpu_fallback_ram:.1f}GB RAM (CPU fallback)",
                    installed=self._is_installed(m),
                )

        if effective_ram < m.ram_gb:
            return ModelFitScore(
                model=m, fit_score=0, speed_score=0, overall=0,
                runnable=False,
                reason=f"Needs {m.ram_gb:.1f}GB RAM, only {effective_ram:.1f}GB free",
                installed=self._is_installed(m),
            )

        # Disk check (soft warning, not hard block)
        disk_ok = hw.disk_free_gb >= m.disk_gb

        # --- Scores ---
        fit_score   = min(effective_ram / max(m.ram_gb, 0.1), 2.0)
        speed_score = min(hw.cpu_cores / 4.0, 2.0)

        # GPU bonus for models that benefit from VRAM
        if m.vram_gb > 0 and (hw.cuda_available or hw.metal_available):
            effective_vram = hw.vram_gb
            if effective_vram >= m.vram_gb:
                speed_score = min(speed_score * 1.5, 2.0)  # 50% speed bonus with GPU

        # Apple Silicon ANE bonus for small models
        if hw.is_apple_silicon and m.tier in (ModelTier.MICRO, ModelTier.FAST):
            speed_score = min(speed_score * 1.3, 2.0)  # ANE acceleration

        overall = min(0.6 * fit_score + 0.4 * speed_score, 1.0)

        reason = "" if disk_ok else f"Disk: needs {m.disk_gb:.1f}GB, {hw.disk_free_gb:.1f}GB free (download may fail)"
        return ModelFitScore(
            model=m,
            fit_score=fit_score,
            speed_score=speed_score,
            overall=overall,
            runnable=True,
            reason=reason,
            installed=self._is_installed(m),
        )

    @staticmethod
    def _is_installed(m: ModelSpec) -> bool:
        """Quick check if model files exist on disk."""
        checks = {
            "bitnet":    lambda: any([
                os.path.exists("models/bitnet-b1.58-2B.gguf"),
                os.path.exists("models/bitnet-b1.58-2B-4T.gguf"),
                bool(os.listdir("models")) if os.path.isdir("models") else False,
            ]),
            "lfm2-fp16": lambda: os.path.isdir("models/lfm2"),
            "lfm2-fp32": lambda: os.path.isdir("models/lfm2"),
            "qwen3-1.7b": lambda: _ollama_has("qwen3:1.7b"),
            "qwen3-8b":   lambda: _ollama_has("qwen3:8b"),
            "qwen3-32b":  lambda: _ollama_has("qwen3:32b"),
            "llama3.1-8b":   lambda: _ollama_has("llama3.1:8b"),
            "llama3.3-70b-q4": lambda: _ollama_has("llama3.3:70b"),
            "airllm-70b": lambda: shutil.which("airllm") is not None,
        }
        checker = checks.get(m.id)
        if not checker:
            return False
        try:
            return checker()
        except Exception:
            return False


def _ollama_has(tag: str) -> bool:
    """Check if Ollama has a model pulled."""
    try:
        out = subprocess.check_output(
            ["ollama", "list"], timeout=3, stderr=subprocess.DEVNULL
        ).decode()
        return tag.split(":")[0] in out
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #

_instance: Optional[HardwareProbe] = None


def get_hardware_probe() -> HardwareProbe:
    global _instance
    if _instance is None:
        _instance = HardwareProbe()
    return _instance


def probe_and_configure() -> ModelFitReport:
    """Convenience: scan + configure router in one call."""
    probe = get_hardware_probe()
    probe.configure_router()
    return probe.scan()
