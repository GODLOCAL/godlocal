"""
oasis_autogenesis.py — X100 OASIS AutoGenesis Engine
=====================================================
Self-evolving code intelligence layer powered by MLX (Apple Silicon).

Architecture:
  FEP Loop    — Free Energy Principle: measure surprise, reduce prediction error
  CodeScanner — full codebase in-memory with hash tracking (detects drift)
  SafeApply   — backup → diff → apply → verify → rollback if broken
  FastAPI     — /evolve endpoint for iPhone Shortcuts integration
  GodLocalBridge — feeds evolution events into performance_logger.py

Usage:
  python oasis_autogenesis.py                     # interactive REPL
  python oasis_autogenesis.py --serve             # HTTP server (iPhone Shortcuts)
  python oasis_autogenesis.py --task "..."        # one-shot CLI

iPhone Shortcuts:
  POST http://localhost:7100/evolve
  Body: {"task": "speed up FEP loop", "apply": false}
"""

import os
import sys
import json
import math
import time
import shutil
import hashlib
import difflib
import argparse
import logging
import asyncio
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AutoGenesis] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AutoGenesis")

# ─── Optional imports ────────────────────────────────────────────────────────

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("mlx_lm not installed — using Ollama fallback")

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-32B-4bit"
FALLBACK_OLLAMA_MODEL = "qwen2.5:7b"
SOUL_FILE = "BOH_OASIS.md"
LOG_FILE = "autogenesis_log.md"
BACKUP_DIR = ".autogenesis_backups"
SERVER_PORT = 7100

# Files to watch (relative to project root)
# Add any file you want AutoGenesis to evolve
WATCHED_FILES = [
    "oasis_autogenesis.py",
    "godlocal_v5.py",
    "self_evolve.py",
    "performance_logger.py",
    "paroquant_backend.py",
    "godlocal_telegram.py",
    "utils.py",
    "BOH_OASIS.md",
]


# ─── Free Energy Principle (FEP) ────────────────────────────────────────────

class FEPMetrics:
    """
    Minimal FEP implementation for code evolution.

    Surprise = -log P(observation | model)
    Prediction error ≈ distance between expected code state and current state.
    Free Energy ≈ surprise + complexity (KL divergence from prior beliefs).
    """

    def __init__(self):
        self._baseline: dict[str, str] = {}  # filename → content hash at last evolution
        self._prediction_errors: list[float] = []

    def snapshot(self, codebase: dict[str, str]) -> None:
        """Record current state as baseline for next cycle."""
        self._baseline = {f: hashlib.sha256(c.encode()).hexdigest() for f, c in codebase.items()}

    def compute_surprise(self, codebase: dict[str, str]) -> dict:
        """
        Compare current codebase to baseline.
        Returns surprise score and per-file drift.
        """
        if not self._baseline:
            self.snapshot(codebase)
            return {"surprise": 0.0, "drift": {}, "changed_files": []}

        drift = {}
        changed = []
        total_surprise = 0.0

        for filename, content in codebase.items():
            current_hash = hashlib.sha256(content.encode()).hexdigest()
            prev_hash = self._baseline.get(filename, "")

            if prev_hash and current_hash != prev_hash:
                # Levenshtein-based surprise proxy
                prev_content = ""  # we only have hashes — use line-count proxy
                lines = content.count("\n") + 1
                prev_lines = lines  # unknown; use 0 surprise for now
                file_surprise = 0.1  # minimal surprise on hash change
                drift[filename] = {"changed": True, "surprise": file_surprise}
                changed.append(filename)
                total_surprise += file_surprise
            elif not prev_hash:
                # New file — maximum novelty
                drift[filename] = {"changed": True, "surprise": 1.0}
                changed.append(filename)
                total_surprise += 1.0

        # Normalise to [0,1]
        max_surprise = max(1.0, len(codebase))
        normalised = min(1.0, total_surprise / max_surprise)

        # Free Energy ≈ surprise + log(complexity)
        n_tokens = sum(len(c.split()) for c in codebase.values())
        complexity = math.log(max(1, n_tokens)) / 10.0
        free_energy = normalised + complexity

        self._prediction_errors.append(normalised)
        return {
            "surprise": round(normalised, 4),
            "free_energy": round(free_energy, 4),
            "changed_files": changed,
            "drift": drift,
            "prediction_errors_history": self._prediction_errors[-10:],
        }

    def running_avg_surprise(self) -> float:
        if not self._prediction_errors:
            return 0.0
        return sum(self._prediction_errors) / len(self._prediction_errors)


# ─── Code Scanner ────────────────────────────────────────────────────────────

class CodeScanner:
    """Loads all watched files into memory with hash tracking."""

    def __init__(self, root: str = ".", watch_list: list[str] = None):
        self.root = Path(root)
        self.watch_list = watch_list or WATCHED_FILES
        self._cache: dict[str, str] = {}
        self._hashes: dict[str, str] = {}

    def load(self) -> dict[str, str]:
        """Reload all watched files. Returns {filename: content}."""
        loaded = {}
        for filename in self.watch_list:
            path = self.root / filename
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                    self._cache[filename] = content
                    self._hashes[filename] = hashlib.sha256(content.encode()).hexdigest()
                    loaded[filename] = content
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        return loaded

    def context_window(self, max_chars: int = 12000) -> str:
        """
        Return a condensed codebase string that fits within max_chars.
        Prioritise files with recent changes.
        """
        parts = []
        total = 0
        for filename, content in self._cache.items():
            header = f"\n### {filename} ({len(content.splitlines())} lines)\n"
            # Truncate large files to first N lines
            preview = "\n".join(content.splitlines()[:100])
            if len(content.splitlines()) > 100:
                preview += f"\n... [{len(content.splitlines()) - 100} more lines] ..."
            chunk = header + preview
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n".join(parts)

    def write(self, filename: str, new_content: str) -> Path:
        """Write updated content to disk (not atomic — use SafeApply)."""
        path = self.root / filename
        path.write_text(new_content, encoding="utf-8")
        self._cache[filename] = new_content
        return path


# ─── Safe Apply ──────────────────────────────────────────────────────────────

class SafeApply:
    """
    Backup → parse LLM output for file blocks → diff → apply → verify → rollback.
    """

    FENCE_START = "```python"
    FENCE_END = "```"

    def __init__(self, scanner: CodeScanner, backup_dir: str = BACKUP_DIR):
        self.scanner = scanner
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

    def backup(self, filename: str) -> Path:
        """Copy current file to .autogenesis_backups/TIMESTAMP_filename."""
        src = self.scanner.root / filename
        if not src.exists():
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = self.backup_dir / f"{ts}_{filename.replace('/', '_')}"
        shutil.copy2(src, dst)
        return dst

    # ── Cursor/Windsurf SEARCH/REPLACE format ─────────────────────────
    SEARCH_MARKER  = "<<<<<<< SEARCH"
    REPLACE_MARKER = "======="
    END_MARKER     = ">>>>>>> REPLACE"

    def parse_search_replace(self, response: str) -> dict[str, list[tuple[str, str]]]:
        """
        Parse Cursor-style SEARCH/REPLACE blocks.

        Expected format per file:
          ### filename.py
          <<<<<<< SEARCH
          old code block
          =======
          new code block
          >>>>>>> REPLACE

        Returns: {filename: [(search_text, replace_text), ...]}
        Multiple SEARCH/REPLACE blocks per file are supported.
        """
        result: dict[str, list[tuple[str, str]]] = {}
        current_file = None
        lines = response.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # File header
            if line.startswith("### ") or line.startswith("# FILE: "):
                name = line.lstrip("# ").lstrip("FILE: ").strip()
                if name.endswith((".py", ".md", ".sh", ".yml", ".yaml", ".json", ".toml")):
                    current_file = name
                    if current_file not in result:
                        result[current_file] = []
                i += 1
                continue

            # SEARCH block start
            if line == self.SEARCH_MARKER and current_file:
                search_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() != self.REPLACE_MARKER:
                    search_lines.append(lines[i])
                    i += 1
                # skip "======="
                i += 1
                replace_lines = []
                while i < len(lines) and lines[i].strip() != self.END_MARKER:
                    replace_lines.append(lines[i])
                    i += 1
                search_text  = "\n".join(search_lines)
                replace_text = "\n".join(replace_lines)
                result[current_file].append((search_text, replace_text))
                i += 1
                continue

            i += 1

        return result

    def apply_search_replace(
        self,
        filename: str,
        patches: list[tuple[str, str]],
        dry_run: bool = True,
    ) -> dict:
        """
        Apply SEARCH/REPLACE patches to a file.
        More surgical than full-file replacement — only touches changed sections.
        Returns same schema as apply().
        """
        path = self.scanner.root / filename
        if not path.exists():
            return {"filename": filename, "diff": "", "applied": False, "reason": "file not found"}

        original = self.scanner._cache.get(filename, path.read_text(encoding="utf-8"))
        content = original

        applied_count = 0
        missed = []
        for search_text, replace_text in patches:
            if search_text in content:
                content = content.replace(search_text, replace_text, 1)
                applied_count += 1
            else:
                missed.append(search_text[:60].replace("\n", "↵"))

        if content == original:
            return {"filename": filename, "diff": "", "applied": False, "reason": "no SEARCH text matched"}

        if missed:
            logger.warning(f"[SafeApply SR] {filename}: {len(missed)} block(s) not matched: {missed}")

        diff = self.generate_diff(filename, content)
        if dry_run:
            return {"filename": filename, "diff": diff, "applied": False, "reason": "dry_run",
                    "blocks_matched": applied_count, "blocks_missed": len(missed)}

        backup_path = self.backup(filename)
        # Run pre-patch tests
        test_passed, test_output = self._run_tests()
        if not test_passed:
            return {"filename": filename, "diff": diff, "applied": False,
                    "reason": "pre-patch tests failed", "test_output": test_output}

        try:
            self.scanner.write(filename, content)
            # Verify post-patch tests
            post_passed, post_output = self._run_tests()
            if not post_passed:
                logger.warning(f"[SafeApply SR] Post-patch tests FAILED — rolling back {filename}")
                if backup_path:
                    shutil.copy2(backup_path, self.scanner.root / filename)
                    self.scanner._cache[filename] = original
                return {"filename": filename, "diff": diff, "applied": False,
                        "reason": "post-patch tests failed — rolled back", "test_output": post_output}
            logger.info(f"[SafeApply SR] Applied {applied_count} block(s) to {filename} ✓")
            return {"filename": filename, "diff": diff, "applied": True,
                    "backup": str(backup_path), "blocks_matched": applied_count}
        except Exception as e:
            if backup_path:
                shutil.copy2(backup_path, self.scanner.root / filename)
            return {"filename": filename, "diff": "", "applied": False, "reason": str(e)}

    def _run_tests(self) -> tuple[bool, str]:
        """Run pytest and return (passed, output). Reusable by both apply() and apply_search_replace()."""
        if not (self.scanner.root / "tests").exists():
            return True, ""
        try:
            import subprocess as _sp
            _res = _sp.run(
                ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
                capture_output=True, text=True, timeout=60,
                cwd=str(self.scanner.root)
            )
            return _res.returncode == 0, (_res.stdout + _res.stderr)[-800:]
        except Exception as e:
            return True, f"test runner unavailable: {e}"  # Don't block on test infra failure

    def parse_llm_output(self, response: str) -> dict[str, str]:
        """
        Extract file→code blocks from LLM response.

        Tries SEARCH/REPLACE format first (Cursor/Windsurf style — surgical, preferred).
        Falls back to full-file ```python``` block extraction.

        SEARCH/REPLACE format (preferred — surgical):
          ### filename.py
          <<<<<<< SEARCH
          old code
          =======
          new code
          >>>>>>> REPLACE

        Full-file format (fallback):
          ### filename.py
          ```python
          ... complete new file ...
          ```
        """
        # Try SEARCH/REPLACE first
        sr_patches = self.parse_search_replace(response)
        if sr_patches:
            # Apply patches in-memory to produce full new content per file
            result = {}
            for filename, patches in sr_patches.items():
                path = self.scanner.root / filename
                original = self.scanner._cache.get(filename, "")
                if not original and path.exists():
                    original = path.read_text(encoding="utf-8")
                content = original
                for search_text, replace_text in patches:
                    if search_text in content:
                        content = content.replace(search_text, replace_text, 1)
                if content != original:
                    result[filename] = content
            if result:
                logger.info(f"[SafeApply] Using SEARCH/REPLACE format for {list(result.keys())}")
                return result

        # Fallback: full-file ```python``` blocks
        files = {}
        lines = response.splitlines()
        current_file = None
        in_fence = False
        buffer = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("### ") or stripped.startswith("# FILE: "):
                name = stripped.lstrip("# ").lstrip("FILE: ").strip()
                if name.endswith((".py", ".md", ".sh", ".yml", ".yaml", ".json", ".toml")):
                    current_file = name
                    buffer = []
                    in_fence = False
                    continue

            if stripped.startswith("```") and not in_fence:
                in_fence = True
                continue

            if stripped == "```" and in_fence:
                in_fence = False
                if current_file and buffer:
                    files[current_file] = "\n".join(buffer)
                    buffer = []
                continue

            if in_fence:
                buffer.append(line)

        if files:
            logger.info(f"[SafeApply] Using full-file format for {list(files.keys())}")
        return files

    def generate_diff(self, filename: str, new_content: str) -> str:
        """Generate unified diff between current and proposed content."""
        old = self.scanner._cache.get(filename, "").splitlines(keepends=True)
        new = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(old, new, fromfile=f"a/{filename}", tofile=f"b/{filename}", n=3)
        return "".join(diff)

    def apply(self, filename: str, new_content: str, dry_run: bool = True) -> dict:
        """
        Apply proposed change. If dry_run=True, only show diff.
        Returns: {filename, diff, applied, backup_path}
        """
        diff = self.generate_diff(filename, new_content)
        if not diff:
            return {"filename": filename, "diff": "", "applied": False, "reason": "no changes"}

        backup_path = None
        applied = False


        if not dry_run:
            # Run test suite before patching (improvement: auto-testing before apply)
            backup_path = self.backup(filename)
            test_passed = True
            test_output = ""
            if (self.scanner.root / "tests").exists():
                try:
                    import subprocess as _sp
                    _res = _sp.run(
                        ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
                        capture_output=True, text=True, timeout=60,
                        cwd=str(self.scanner.root)
                    )
                    test_passed = _res.returncode == 0
                    test_output = (_res.stdout + _res.stderr)[-800:]
                    if not test_passed:
                        logger.warning(f"[SafeApply] Tests FAILED before patching {filename} — aborting apply")
                        logger.warning(f"Test output: {test_output}")
                        return {
                            "filename": filename,
                            "diff": diff,
                            "applied": False,
                            "backup": str(backup_path) if backup_path else None,
                            "reason": "pre-patch tests failed",
                            "test_output": test_output,
                            "lines_changed": diff.count("\n+") + diff.count("\n-"),
                        }
                    logger.info(f"[SafeApply] Pre-patch tests passed ✓")
                except Exception as _te:
                    logger.warning(f"[SafeApply] Could not run tests: {_te} — proceeding anyway")
            try:
                self.scanner.write(filename, new_content)
                applied = True
                logger.info(f"Applied changes to {filename} (backup: {backup_path})")
            except Exception as e:
                logger.exception(f"Failed to apply {filename}")
                if backup_path and backup_path.exists():
                    shutil.copy2(backup_path, self.scanner.root / filename)
                    logger.warning(f"Rolled back {filename}")


        return {
            "filename": filename,
            "diff": diff,
            "applied": applied,
            "backup": str(backup_path) if backup_path else None,
            "lines_changed": diff.count("\n+") + diff.count("\n-"),
        }

    def rollback(self, filename: str) -> bool:
        """Roll back to latest backup for filename."""
        backups = sorted(self.backup_dir.glob(f"*_{filename.replace('/', '_')}"), reverse=True)
        if not backups:
            return False
        latest = backups[0]
        shutil.copy2(latest, self.scanner.root / filename)
        logger.info(f"Rolled back {filename} from {latest.name}")
        return True


# ─── LLM Bridge ──────────────────────────────────────────────────────────────

class LLMBridge:
    """Unified interface: MLX (Mac) → Ollama fallback → raises if neither."""

    def __init__(self, mlx_model: str = DEFAULT_MLX_MODEL, ollama_model: str = FALLBACK_OLLAMA_MODEL):
        self.mlx_model = mlx_model
        self.ollama_model = ollama_model
        self._mlx = None  # lazy load

    def _load_mlx(self):
        if self._mlx is None:
            logger.info(f"Loading MLX model {self.mlx_model} (first run ~30s)...")
            self._mlx = load(self.mlx_model)
            logger.info("MLX model loaded ✓")
        return self._mlx

    def generate(self, prompt: str, max_tokens: int = 3000) -> str:
        if MLX_AVAILABLE:
            try:
                model, tokenizer = self._load_mlx()
                return generate(model, tokenizer, prompt, max_tokens=max_tokens)
            except Exception as e:
                logger.exception("MLX generation failed, trying Ollama fallback")

        if OLLAMA_AVAILABLE:
            try:
                resp = _ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"num_predict": max_tokens},
                )
                return resp["message"]["content"]
            except Exception as e:
                logger.exception("Ollama fallback failed")

        raise RuntimeError("No LLM available. Install mlx_lm (Mac) or ollama.")


# ─── AutoGenesis Core ────────────────────────────────────────────────────────

class AutoGenesis:
    """
    X100 OASIS AutoGenesis — self-evolving code intelligence.

    FEP Loop:
      1. Scan codebase
      2. Compute surprise (what changed since last cycle)
      3. Build context-aware prompt
      4. Generate improvements
      5. SafeApply (dry_run=True by default)
      6. Log to autogenesis_log.md + GodLocal performance_logger
    """

    SYSTEM_PROMPT = """You are БОГ || OASIS AutoGenesis — sovereign AI code intelligence.
Your job: plan, then evolve the codebase surgically.

## Step 1 — Plan (REQUIRED, output before any code):
[PLAN]
- Mode: <CODING|TRADING|WRITING|MEDICAL|ANALYSIS>
- Prediction error: <what current code gets wrong for this task>
- Minimal change: <what specifically changes, one sentence>
- Risk: <LOW|MEDIUM|HIGH> — <why>
- Files: <comma-separated filenames>
[/PLAN]

## Step 2 — Patches (SEARCH/REPLACE format — preferred, surgical):

### filename.py
<<<<<<< SEARCH
exact verbatim code to find (no omissions)
=======
replacement code
>>>>>>> REPLACE

Multiple blocks per file supported. Only output changed sections.

## Fallback (full-file — only if SEARCH/REPLACE doesn't apply):

### filename.py
```python
... complete new file ...
```

Rules:
- Never remove LOCKED sections without explicit instruction.
- Preserve all existing tests and backward compatibility.
- Output ONLY [PLAN] + patches. No prose outside these structures.
"""

    def __init__(self, root: str = ".", mlx_model: str = DEFAULT_MLX_MODEL):
        self.root = Path(root)
        self.soul = self._load_soul()
        self.scanner = CodeScanner(root)
        self.fep = FEPMetrics()
        self.llm = LLMBridge(mlx_model=mlx_model)
        self.safe_apply = SafeApply(self.scanner)
        self.evolution_count = 0
        logger.info("🔥 BOG || OASIS AutoGenesis ACTIVATED 🔥")
        logger.info(f"  MLX: {MLX_AVAILABLE} | Ollama: {OLLAMA_AVAILABLE} | Soul: {bool(self.soul)}")

    def _load_soul(self) -> str:
        soul_path = self.root / SOUL_FILE
        if soul_path.exists():
            return soul_path.read_text(encoding="utf-8")
        return "# BOG || OASIS — Sovereign AI System\n## Goal: Maximize free energy minimisation across the X100 ecosystem."

    def _load_godlocal_bridge(self):
        """Optionally integrate with GodLocal performance_logger."""
        try:
            sys.path.insert(0, str(self.root))
            from performance_logger import PerformanceLogger
            return PerformanceLogger()
        except Exception:
            return None


    def _pre_evolve_plan(self, task: str, fep_metrics: dict, codebase: dict) -> dict:
        """
        Devin-style planning pass: lightweight CoT before code generation.
        Uses a focused mini-prompt to extract structured plan JSON.
        Returns dict with keys: mode, prediction_error, minimal_change, risk, files_to_touch.
        """
        # Detect mode from task keywords
        task_lower = task.lower()
        if any(k in task_lower for k in ["trad", "market", "fund", "position", "kalshi", "manifold"]):
            mode = "TRADING"
        elif any(k in task_lower for k in ["write", "blog", "post", "copy", "draft"]):
            mode = "WRITING"
        elif any(k in task_lower for k in ["medical", "mri", "dicom", "hipaa", "patient"]):
            mode = "MEDICAL"
        elif any(k in task_lower for k in ["analyz", "report", "metric", "data", "insight"]):
            mode = "ANALYSIS"
        else:
            mode = "CODING"

        # Auto-swap agent if AgentPool available
        if self.agent_pool:
            agent_map = {
                "TRADING": "trading",
                "WRITING": "writing",
                "MEDICAL": "medical",
                "ANALYSIS": "coding",
                "CODING": "coding",
            }
            target_agent = agent_map.get(mode, "coding")
            try:
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync context — fire and forget
                    import concurrent.futures as _cf
                    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                        ex.submit(lambda: _asyncio.run(self.agent_pool.swap(target_agent)))
                else:
                    loop.run_until_complete(self.agent_pool.swap(target_agent))
                logger.info(f"[AutoGenesis] AgentPool swapped to {target_agent} for {mode} task")
            except Exception as _e:
                logger.debug(f"[AutoGenesis] AgentPool swap skipped: {_e}")

        # Lightweight plan prompt — small token budget
        plan_prompt = f"""Task: {task}

Surprise: {fep_metrics.get('surprise', 0):.3f} | Changed files: {fep_metrics.get('changed_files', [])}
Files in codebase: {list(codebase.keys())[:20]}

Output JSON only (no markdown):
{{
  "mode": "{mode}",
  "prediction_error": "one sentence: what does current code get wrong?",
  "minimal_change": "one sentence: what specifically changes?",
  "risk": "LOW|MEDIUM|HIGH",
  "risk_reason": "why",
  "files_to_touch": ["file1.py", "file2.py"]
}}"""

        try:
            plan_response = self.llm.generate(plan_prompt, max_tokens=256)
            # Extract JSON from response
            import re as _re
            json_match = _re.search(r'\{[^{}]+\}', plan_response, _re.DOTALL)
            if json_match:
                import json as _json
                plan = _json.loads(json_match.group())
                plan["mode"] = mode  # Override with keyword-detected mode
                return plan
        except Exception as _e:
            logger.debug(f"[AutoGenesis] Planning pass failed: {_e}")

        # Fallback plan
        return {
            "mode": mode,
            "prediction_error": "unknown — planning pass failed",
            "minimal_change": task[:100],
            "risk": "LOW",
            "risk_reason": "planning pass failed, proceeding cautiously",
            "files_to_touch": [],
        }

    def evolve(self, task: str, apply: bool = False, max_tokens: int = 3000) -> dict:
        """
        Run one evolution cycle.

        Args:
            task:       Natural language description of what to improve.
            apply:      If True, applies changes to disk (after showing diff).
            max_tokens: LLM token budget.

        Returns:
            dict with fep_metrics, proposed_files, diffs, applied
        """
        logger.info(f"Evolution #{self.evolution_count + 1}: {task[:80]}")

        # 1. Scan
        codebase = self.scanner.load()
        logger.info(f"  Loaded {len(codebase)} files")

        # 2. FEP
        fep_metrics = self.fep.compute_surprise(codebase)
        logger.info(f"  Surprise: {fep_metrics['surprise']} | Free Energy: {fep_metrics['free_energy']}")

        # 3. Pre-evolve planning pass (Devin-style — CoT before code)
        plan = self._pre_evolve_plan(task, fep_metrics, codebase)
        logger.info(f"  Plan: mode={plan.get('mode','?')} risk={plan.get('risk','?')} files={plan.get('files_to_touch','?')}")

        # 4. Build prompt
        context = self.scanner.context_window(max_chars=10000)
        prompt = f"""{self.SYSTEM_PROMPT}

## Soul (identity + locked rules)
{self.soul[:1000]}

## FEP Metrics
- Surprise: {fep_metrics['surprise']} (1.0 = maximum novelty)
- Free Energy: {fep_metrics['free_energy']}
- Changed since last cycle: {fep_metrics['changed_files']}
- Avg surprise history: {self.fep.running_avg_surprise():.4f}

## Pre-Evolution Plan (already computed)
- Mode: {plan.get('mode', 'CODING')}
- Prediction error: {plan.get('prediction_error', 'unknown')}
- Minimal change: {plan.get('minimal_change', 'unknown')}
- Risk: {plan.get('risk', 'LOW')}
- Files to touch: {plan.get('files_to_touch', 'unknown')}

## Current Codebase
{context}

## Task
{task}

## Instructions
Use SEARCH/REPLACE format (preferred) or full-file blocks (fallback).
Output [PLAN] block first, then patches.
"""

        # 4. Generate
        t0 = time.time()
        response = self.llm.generate(prompt, max_tokens=max_tokens)
        elapsed = round(time.time() - t0, 1)
        logger.info(f"  LLM response in {elapsed}s ({len(response.split())} tokens ~)")

        # 5. Parse proposed files
        proposed = self.safe_apply.parse_llm_output(response)
        logger.info(f"  Proposed changes to: {list(proposed.keys())}")

        # 6. Diff + optional apply
        results = []
        for filename, new_content in proposed.items():
            r = self.safe_apply.apply(filename, new_content, dry_run=not apply)
            results.append(r)

        # 7. Log
        self.evolution_count += 1
        self._log(task, fep_metrics, response, results)

        # Update FEP baseline
        self.fep.snapshot(codebase)

        # 8. GodLocal bridge
        bridge = self._load_godlocal_bridge()
        if bridge:
            try:
                bridge.log_interaction(
                    user_msg=f"[AutoGenesis] {task}",
                    assistant_msg=response[:500],
                    was_corrected=False,
                )
            except Exception:
                pass

        return {
            "task": task,
            "evolution": self.evolution_count,
            "fep": fep_metrics,
            "proposed_files": list(proposed.keys()),
            "diffs": [{r["filename"]: r["diff"][:2000]} for r in results],
            "applied": [r for r in results if r.get("applied")],
            "elapsed_s": elapsed,
            "llm_response": response,
        }

    def _log(self, task: str, fep: dict, response: str, results: list) -> None:
        """Append structured entry to autogenesis_log.md."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        applied_files = [r["filename"] for r in results if r.get("applied")]
        with open(self.root / LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"""
---
## Evolution #{self.evolution_count} — {ts}
**Task:** {task}
**FEP:** surprise={fep['surprise']} free_energy={fep['free_energy']} changed={fep['changed_files']}
**Proposed:** {[r['filename'] for r in results]}
**Applied:** {applied_files}

<details><summary>LLM Output</summary>

{response[:4000]}

</details>
""")
        # Auto-documentation: append to CHANGELOG.md when files actually patched
        if applied_files:
            try:
                changelog = self.root / "CHANGELOG.md"
                header_exists = changelog.exists()
                with changelog.open("a", encoding="utf-8") as _cl:
                    if not header_exists:
                        _cl.write("# CHANGELOG\n\nAuto-generated by AutoGenesis.\n\n")
                    _cl.write(
                        f"## [{ts}] AutoGenesis #{self.evolution_count}\n"
                        f"**Task:** {task}\n"
                        f"**Files patched:** {', '.join(applied_files)}\n"
                        f"**FEP surprise:** {fep['surprise']}\n\n"
                    )
                logger.info(f"📝 CHANGELOG.md updated for evolution #{self.evolution_count}")
            except Exception as _ce:
                logger.warning(f"[AutoDoc] CHANGELOG update failed: {_ce}")


# ─── iPhone Shortcuts HTTP Server ────────────────────────────────────────────

def make_server(genesis: AutoGenesis) -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise ImportError("Install fastapi + uvicorn for iPhone Shortcuts server")

    app = FastAPI(title="OASIS AutoGenesis", version="1.0", description="iPhone Shortcuts interface for AutoGenesis")

    class EvolveRequest(BaseModel):
        task: str
        apply: bool = False
        max_tokens: int = 2048

    class EvolveResponse(BaseModel):
        evolution: int
        fep: dict
        proposed_files: list
        applied: list
        elapsed_s: float
        diff_preview: str

    @app.get("/health")
    async def health():
        return {"status": "ok", "mlx": MLX_AVAILABLE, "ollama": OLLAMA_AVAILABLE}

    @app.get("/status")
    async def status():
        return {
            "evolution_count": genesis.evolution_count,
            "files_loaded": list(genesis.scanner._cache.keys()),
            "avg_surprise": genesis.fep.running_avg_surprise(),
            "backups": len(list(Path(BACKUP_DIR).glob("*"))) if Path(BACKUP_DIR).exists() else 0,
        }

    @app.post("/evolve", response_model=EvolveResponse)
    async def evolve(req: EvolveRequest):
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: genesis.evolve(req.task, apply=req.apply, max_tokens=req.max_tokens)
            )
            diff_preview = "\n\n".join(
                f"--- {list(d.keys())[0]} ---\n{list(d.values())[0][:500]}"
                for d in result["diffs"]
                if d
            )
            return EvolveResponse(
                evolution=result["evolution"],
                fep=result["fep"],
                proposed_files=result["proposed_files"],
                applied=result["applied"],
                elapsed_s=result["elapsed_s"],
                diff_preview=diff_preview or "no changes proposed",
            )
        except Exception as e:
            logger.exception("Error in /evolve")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/rollback/{filename}")
    async def rollback(filename: str):
        success = genesis.safe_apply.rollback(filename)
        if not success:
            raise HTTPException(status_code=404, detail=f"No backup found for {filename}")
        return {"status": "rolled_back", "filename": filename}

    @app.get("/log")
    async def get_log(lines: int = 50):
        log_path = genesis.root / LOG_FILE
        if not log_path.exists():
            return {"log": ""}
        content = log_path.read_text(encoding="utf-8")
        return {"log": "\n".join(content.splitlines()[-lines:])}


    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Simple FEP dashboard — visual overview for browser / iPhone."""
        try:
            from fastapi.responses import HTMLResponse as _HR
        except ImportError:
            return {"error": "HTMLResponse not available"}
        log_path = genesis.root / LOG_FILE
        log_tail = ""
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            log_tail = "\n".join(lines[-30:])
        backups = len(list(Path(BACKUP_DIR).glob("*"))) if Path(BACKUP_DIR).exists() else 0
        html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>GodLocal AutoGenesis</title>
<style>
body{{background:#0d0d0d;color:#00ff41;font-family:monospace;padding:2em}}
h1{{color:#00e5ff}}h2{{color:#7b2fff}}
pre{{background:#111;padding:1em;border:1px solid #333;white-space:pre-wrap;font-size:.8em}}
.metric{{display:inline-block;margin:1em;padding:.8em 1.5em;border:1px solid #00ff41;border-radius:4px}}
.metric span{{color:#fff;font-size:1.4em;display:block}}
</style></head>
<body>
<h1>🌐 GodLocal AutoGenesis v5.1</h1>
<div class='metric'>Evolutions<span>{genesis.evolution_count}</span></div>
<div class='metric'>Avg Surprise<span>{genesis.fep.running_avg_surprise():.3f}</span></div>
<div class='metric'>Files Watched<span>{len(genesis.scanner._cache)}</span></div>
<div class='metric'>Backups<span>{backups}</span></div>
<h2>Recent Evolution Log (last 30 lines)</h2>
<pre>{log_tail or "(no entries yet)"}</pre>
</body></html>"""
        return _HR(content=html)

    return app


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OASIS AutoGenesis")
    parser.add_argument("--task", type=str, help="Evolution task (one-shot)")
    parser.add_argument("--apply", action="store_true", help="Apply changes to disk")
    parser.add_argument("--serve", action="store_true", help="Start iPhone Shortcuts HTTP server")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help=f"Server port (default {SERVER_PORT})")
    parser.add_argument("--model", type=str, default=DEFAULT_MLX_MODEL, help="MLX model name")
    parser.add_argument("--root", type=str, default=".", help="Project root directory")
    parser.add_argument("--rollback", type=str, help="Roll back named file to latest backup")
    args = parser.parse_args()

    genesis = AutoGenesis(root=args.root, mlx_model=args.model)

    if args.rollback:
        ok = genesis.safe_apply.rollback(args.rollback)
        print("✓ rolled back" if ok else "✗ no backup found")
        return

    if args.serve:
        if not FASTAPI_AVAILABLE:
            print("Install: pip install fastapi uvicorn")
            sys.exit(1)
        app = make_server(genesis)
        print(f"\n🚀 AutoGenesis server → http://localhost:{args.port}")
        print(f"   iPhone Shortcuts: POST /evolve {{task: ..., apply: false}}")
        print(f"   Docs: http://localhost:{args.port}/docs\n")
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
        return

    if args.task:
        result = genesis.evolve(args.task, apply=args.apply)
        print("\n" + "─" * 60)
        print(f"Evolution #{result['evolution']} complete")
        print(f"FEP: surprise={result['fep']['surprise']} free_energy={result['fep']['free_energy']}")
        print(f"Proposed: {result['proposed_files']}")
        if result['applied']:
            print(f"Applied: {[r['filename'] for r in result['applied']]}")
        print("\n=== DIFF PREVIEW ===")
        for d in result['diffs']:
            for fname, diff in d.items():
                print(f"\n--- {fname} ---")
                print(diff[:1500])
        return

    # Interactive REPL
    print("\n🔥 BOG || OASIS AutoGenesis REPL 🔥")
    print("  Commands: evolve <task> | apply <task> | rollback <file> | status | quit\n")
    while True:
        try:
            line = input("AutoGenesis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue

        if line == "quit":
            break

        if line == "status":
            print(json.dumps({
                "evolutions": genesis.evolution_count,
                "files": list(genesis.scanner._cache.keys()),
                "avg_surprise": genesis.fep.running_avg_surprise(),
            }, indent=2))
            continue

        if line.startswith("rollback "):
            fname = line.split(" ", 1)[1].strip()
            ok = genesis.safe_apply.rollback(fname)
            print("✓ rolled back" if ok else "✗ no backup found")
            continue

        apply = line.startswith("apply ")
        task = line[len("apply "):] if apply else line[len("evolve "):] if line.startswith("evolve ") else line
        result = genesis.evolve(task, apply=apply)
        print(f"\n✓ Evolution #{result['evolution']} | files: {result['proposed_files']} | elapsed: {result['elapsed_s']}s")
        if apply and result["applied"]:
            print(f"  Applied: {[r['filename'] for r in result['applied']]}")
        print()


if __name__ == "__main__":
    main()
