"""
extensions/sandbox/safe_apply.py — Docker Sandbox for Safe Evolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zero-touch safety layer: no AutoGenesis patch touches real code
until pytest passes in an isolated container.

Usage:
    from extensions.sandbox.safe_apply import DockerSafeApply
    sa = DockerSafeApply()
    ok = sa.apply("godlocal_v5.py", new_code)

Integration with self_evolve.py:
    In SelfEvolve.__init__:
        from extensions.sandbox.safe_apply import DockerSafeApply
        self.safe_apply = DockerSafeApply()
    Replace file writes:
        self.safe_apply.apply(file_path, new_code)
"""

from __future__ import annotations
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "godlocal-sandbox"
DOCKERFILE_DIR = Path(__file__).parent


class DockerSafeApply:
    """
    Runs pytest inside a Docker container against the patched code.
    Only writes to the real file if tests pass.

    Requires: Docker daemon running + image built (see README).
    Fallback: if Docker is unavailable, falls back to in-process pytest.
    """

    def __init__(self, image: str = SANDBOX_IMAGE, timeout: int = 120):
        self.image = image
        self.timeout = timeout
        self._docker_available = self._check_docker()
        if self._docker_available:
            self._ensure_image()

    def _check_docker(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            logger.warning("[DockerSafeApply] Docker not available — using fallback in-process pytest")
            return False

    def _ensure_image(self):
        """Build sandbox image if not present."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True, timeout=10
            )
            if result.returncode != 0:
                logger.info(f"[DockerSafeApply] Building {self.image}...")
                subprocess.run(
                    ["docker", "build", "-f",
                     str(DOCKERFILE_DIR / "Dockerfile.sandbox"),
                     "-t", self.image, str(DOCKERFILE_DIR)],
                    check=True, timeout=180
                )
                logger.info(f"[DockerSafeApply] Image {self.image} ready ✓")
        except Exception as e:
            logger.warning(f"[DockerSafeApply] Image build failed: {e}")
            self._docker_available = False

    def apply(self, file_path: str, new_code: str, project_root: str = ".") -> bool:
        """
        Stage new_code into a temp copy of the project, run pytest in Docker.
        If green → write to real file. Returns True on success.
        """
        project_root_path = Path(project_root).resolve()
        target = project_root_path / file_path

        if self._docker_available:
            return self._apply_docker(file_path, new_code, project_root_path)
        else:
            return self._apply_fallback(file_path, new_code, project_root_path)

    def _apply_docker(self, file_path: str, new_code: str, project_root: Path) -> bool:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Mirror project into sandbox
            shutil.copytree(str(project_root), str(tmp_path), dirs_exist_ok=True)
            (tmp_path / file_path).write_text(new_code, encoding="utf-8")

            try:
                result = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "-v", f"{tmp_path}:/sandbox",
                        "-w", "/sandbox",
                        self.image,
                        "python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"
                    ],
                    capture_output=True, text=True, timeout=self.timeout
                )
                passed = result.returncode == 0
                output = (result.stdout + result.stderr)[-1200:]
            except subprocess.TimeoutExpired:
                logger.error("[DockerSafeApply] Container timed out")
                return False
            except Exception as e:
                logger.error(f"[DockerSafeApply] Docker run failed: {e}")
                return False

        if passed:
            (project_root / file_path).write_text(new_code, encoding="utf-8")
            logger.info(f"✅ SAFE-APPLY OK (Docker): {file_path}")
            return True
        else:
            logger.warning(f"❌ SAFE-APPLY FAIL (Docker): {file_path}\n{output}")
            return False

    def _apply_fallback(self, file_path: str, new_code: str, project_root: Path) -> bool:
        """In-process pytest fallback when Docker is unavailable."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copytree(str(project_root), str(tmp_path), dirs_exist_ok=True)
            (tmp_path / file_path).write_text(new_code, encoding="utf-8")

            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
                capture_output=True, text=True, timeout=self.timeout,
                cwd=str(tmp_path)
            )
            passed = result.returncode == 0
            if passed:
                (project_root / file_path).write_text(new_code, encoding="utf-8")
                logger.info(f"✅ SAFE-APPLY OK (fallback): {file_path}")
                return True
            else:
                logger.warning(f"❌ SAFE-APPLY FAIL (fallback): {file_path}")
                return False
