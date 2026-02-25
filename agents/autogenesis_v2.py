"""agents/autogenesis_v2.py — AutoGenesis v2 for БОГ || OASIS v6
FEPMetrics: correction_rate + free_energy (real signal, not just hash)
CodeScanner: dynamic .py/.md discovery (no hardcoded WATCHED_FILES)
DockerSafeApply: build→patch→test→apply or rollback
Plan-and-Execute: [PLAN] JSON → SEARCH/REPLACE patches → revision loop
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.brain import Brain
from core.settings import settings

# ── Potpie integration (optional — gracefully degraded when server is down) ──
try:
    from extensions.xzero import get_connector as _get_xzero_connector
    _PotpieConnector = _get_xzero_connector("potpie")
    _POTPIE_AVAILABLE = True
except Exception:
    _POTPIE_AVAILABLE = False

logger = logging.getLogger(__name__)

PLAN_PROMPT = """You are a senior Python architect. Analyze the task and output a JSON [PLAN].

Task: {task}

Output ONLY valid JSON:
{{
  "mode": "REFACTOR|FEATURE|BUGFIX|DOCS",
  "target_files": ["list", "of", "files"],
  "changes": ["brief description of each change"],
  "risk": "low|medium|high",
  "test_command": "pytest tests/ -x -q"
}}"""

PATCH_PROMPT = """You are a precise code editor. Generate SEARCH/REPLACE patches.

[PLAN]
{plan}

Current file content:
```python
{content}
```

Task: {task}

Output patches in this format ONLY:
<<<SEARCH
exact lines to find
>>>
<<<REPLACE
replacement lines
>>>

Multiple patches allowed. Output nothing else."""


# ── FEP Metrics ───────────────────────────────────────────────────────────
@dataclass
class FEPState:
    correction_rate: float = 0.0   # was_corrected / total interactions
    free_energy: float = 1.0       # 1 - correction_rate (lower = better)
    total_interactions: int = 0
    corrections: int = 0
    evolutions: int = 0
    last_evolved_at: float = 0.0
    code_hashes: dict[str, str] = field(default_factory=dict)

    def update(self, corrected: bool) -> None:
        self.total_interactions += 1
        if corrected:
            self.corrections += 1
        self.correction_rate = self.corrections / max(self.total_interactions, 1)
        self.free_energy = 1.0 - self.correction_rate

    def file_changed(self, path: str, new_hash: str) -> bool:
        old = self.code_hashes.get(path)
        self.code_hashes[path] = new_hash
        return old != new_hash

    def to_dict(self) -> dict:
        return {
            "correction_rate": round(self.correction_rate, 4),
            "free_energy": round(self.free_energy, 4),
            "total_interactions": self.total_interactions,
            "corrections": self.corrections,
            "evolutions": self.evolutions,
            "last_evolved_at": self.last_evolved_at,
        }


# ── Code Scanner ──────────────────────────────────────────────────────────
class CodeScanner:
    """Dynamic .py/.md scanner — no hardcoded file lists."""

    IGNORE = {".git", "__pycache__", "node_modules", ".venv", "venv",
              "godlocal_data", "dist", ".mypy_cache"}

    def scan(self, root: str = ".") -> list[Path]:
        root_path = Path(root)
        files: list[Path] = []
        for p in root_path.rglob("*"):
            if any(part in self.IGNORE for part in p.parts):
                continue
            if p.suffix in (".py", ".md") and p.is_file():
                files.append(p)
        return sorted(files)

    def hash_file(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ── SEARCH/REPLACE Patcher ────────────────────────────────────────────────
class SearchReplacePatcher:
    PATCH_RE = re.compile(
        r"<<<SEARCH\n(.*?)>>>\n<<<REPLACE\n(.*?)>>>",
        re.DOTALL,
    )

    def apply(self, content: str, patch_text: str) -> tuple[str, int]:
        """Returns (patched_content, patch_count)."""
        result = content
        count = 0
        for m in self.PATCH_RE.finditer(patch_text):
            search  = m.group(1)
            replace = m.group(2)
            if search in result:
                result = result.replace(search, replace, 1)
                count += 1
            else:
                logger.warning("SEARCH block not found: %.60s…", search)
        return result, count


# ── Docker Safe Apply ─────────────────────────────────────────────────────
class DockerSafeApply:
    """Build sandbox image → apply patch in temp → run tests → commit or rollback."""

    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)
        self.image = settings.sandbox_image
        self.timeout = settings.sandbox_timeout

    def _docker_available(self) -> bool:
        try:
            subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
            return True
        except Exception:
            return False

    def run_tests(self, file_path: Path, patched_content: str) -> tuple[bool, str]:
        """Write patch to temp file, run pytest in Docker, restore on failure."""
        if not self._docker_available():
            logger.warning("Docker not available — running tests locally")
            return self._run_local_tests(file_path, patched_content)

        backup = file_path.read_text(encoding="utf-8")
        try:
            file_path.write_text(patched_content, encoding="utf-8")
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{self.root.resolve()}:/app:ro",
                    self.image,
                    "pytest", "tests/", "-x", "-q", "--tb=short",
                ],
                capture_output=True, text=True, timeout=self.timeout,
            )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            if not passed:
                file_path.write_text(backup, encoding="utf-8")
                logger.warning("Tests failed — rolled back %s", file_path)
            return passed, output
        except subprocess.TimeoutExpired:
            file_path.write_text(backup, encoding="utf-8")
            return False, "Timeout in Docker sandbox"
        except Exception as e:
            file_path.write_text(backup, encoding="utf-8")
            return False, str(e)

    def _run_local_tests(self, file_path: Path, patched_content: str) -> tuple[bool, str]:
        backup = file_path.read_text(encoding="utf-8")
        file_path.write_text(patched_content, encoding="utf-8")
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
            capture_output=True, text=True, cwd=str(self.root), timeout=self.timeout,
        )
        if result.returncode != 0:
            file_path.write_text(backup, encoding="utf-8")
        return result.returncode == 0, result.stdout + result.stderr


# ── AutoGenesis v2 ────────────────────────────────────────────────────────
class AutoGenesis:
    """Main evolution engine. Used in sleep_cycle Phase 4 and POST /evolve."""

    def __init__(self, root: str = ".") -> None:
        self.root = root
        self.fep = FEPState()
        self.scanner = CodeScanner()
        self.patcher = SearchReplacePatcher()
        self.safe_apply = DockerSafeApply(root)

    # Public API -----------------------------------------------------------
    def evolve(
        self,
        task: str,
        apply: bool | None = None,
        max_revisions: int | None = None,
    ) -> dict[str, Any]:
        """Synchronous entry point (sleep_cycle calls this in executor)."""
        apply = apply if apply is not None else settings.autogenesis_apply
        max_revisions = max_revisions or settings.autogenesis_max_revisions
        return asyncio.run(self._evolve_async(task, apply, max_revisions))

    async def evolve_async(
        self,
        task: str,
        apply: bool | None = None,
        max_revisions: int | None = None,
    ) -> dict[str, Any]:
        apply = apply if apply is not None else settings.autogenesis_apply
        max_revisions = max_revisions or settings.autogenesis_max_revisions
        return await self._evolve_async(task, apply, max_revisions)

    def record_correction(self, was_corrected: bool) -> None:
        self.fep.update(was_corrected)

    def fep_metrics(self) -> dict:
        return self.fep.to_dict()

    # Internal -------------------------------------------------------------
    async def _evolve_async(self, task: str, apply: bool, max_revisions: int) -> dict[str, Any]:
        brain = Brain.get()
        t0 = time.time()

        # Step 1: Plan
        logger.info("[AutoGenesis] Planning: %.80s", task)
        plan_raw = await brain.think(PLAN_PROMPT.format(task=task), max_tokens=512)
        plan = self._parse_plan(plan_raw)

        if plan.get("risk") == "high" and not apply:
            return {
                "status": "skipped",
                "reason": "high-risk plan — set apply=True to proceed",
                "plan": plan,
            }

        # Step 2a: Potpie codebase intelligence (optional — skipped if server down)
        potpie_context: str = ""
        if _POTPIE_AVAILABLE:
            try:
                _pc = _PotpieConnector()
                _health = await _pc.health_check()
                if _health.get("status") == "ok":
                    _q = f"What files and functions are relevant to this task: {task[:200]}"
                    _ans = await _pc.query_agent(
                        project_id="godlocal",
                        question=_q,
                        agent="qa",
                    )
                    potpie_context = _ans.get("content", "")
                    if potpie_context:
                        logger.info("[AutoGenesis] Potpie context: %.120s…", potpie_context)
            except Exception as _e:
                logger.debug("[AutoGenesis] Potpie unavailable: %s", _e)

        # Step 2b: Select target files
        target_paths = self._resolve_targets(plan.get("target_files", []))
        if not target_paths:
            return {"status": "no_targets", "plan": plan, "potpie_context": potpie_context}

        results: list[dict] = []
        for file_path in target_paths[:3]:  # max 3 files per evolution
            content = file_path.read_text(encoding="utf-8")
            patch_result = await self._patch_file(
                brain, task, plan, file_path, content, apply, max_revisions, potpie_context
            )
            results.append(patch_result)

        elapsed = round(time.time() - t0, 2)
        self.fep.evolutions += 1
        self.fep.last_evolved_at = time.time()

        return {
            "status": "evolved" if apply else "dry_run",
            "task": task,
            "plan": plan,
            "files": results,
            "elapsed_sec": elapsed,
            "fep": self.fep.to_dict(),
        }

    async def _patch_file(
        self,
        brain: Brain,
        task: str,
        plan: dict,
        file_path: Path,
        content: str,
        apply: bool,
        max_revisions: int,
        potpie_context: str = "",
    ) -> dict:
        revision = 0
        patches_applied = 0

        while revision < max_revisions:
            _potpie_note = (
                f"\n\n[Potpie codebase context]\n{potpie_context[:600]}"
                if potpie_context else ""
            )
            patch_text = await brain.think(
                PATCH_PROMPT.format(
                    plan=json.dumps(plan, indent=2),
                    content=content[:6000],  # trim to avoid token overflow
                    task=task + _potpie_note,
                ),
                max_tokens=1024,
            )
            patched, count = self.patcher.apply(content, patch_text)
            if count == 0:
                break
            patches_applied += count

            if apply:
                passed, test_out = self.safe_apply.run_tests(file_path, patched)
                if passed:
                    file_path.write_text(patched, encoding="utf-8")
                    logger.info("[AutoGenesis] Applied %d patches to %s", count, file_path)
                    return {"file": str(file_path), "patches": patches_applied, "applied": True, "tests": "passed"}
                else:
                    logger.warning("[AutoGenesis] Tests failed — revision %d", revision + 1)
                    revision += 1
                    continue
            else:
                # dry-run — just return diff preview
                return {
                    "file": str(file_path),
                    "patches": patches_applied,
                    "applied": False,
                    "diff_preview": patch_text[:500],
                }

        return {"file": str(file_path), "patches": patches_applied, "applied": False, "status": "max_revisions_reached"}

    def _parse_plan(self, raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"mode": "FEATURE", "target_files": [], "changes": [raw[:200]], "risk": "low"}

    def _resolve_targets(self, target_files: list[str]) -> list[Path]:
        root = Path(self.root)
        paths: list[Path] = []
        for f in target_files:
            p = root / f
            if p.exists() and p.suffix == ".py":
                paths.append(p)
        if not paths:
            # fallback: scan and pick recently modified .py
            all_py = sorted(
                (p for p in root.rglob("*.py") if "__pycache__" not in str(p)),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            paths = all_py[:2]
        return paths
