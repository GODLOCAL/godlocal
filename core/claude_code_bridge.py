"""
core/claude_code_bridge.py
ClaudeCodeBridge — run Claude Code CLI locally against Ollama (zero cost, private).

Ollama v0.14+ implements Anthropic Messages API compatibility layer.
Claude Code doesn't care where its backend lives — it just needs:
  ANTHROPIC_BASE_URL=http://localhost:11434
  ANTHROPIC_AUTH_TOKEN=ollama (or any non-empty string)

This bridge spawns claude-code as an async subprocess for agentic tasks
(file edits, bash commands, multi-step refactoring) that go beyond raw
LLM completion — used by AutoGenesis V2 self-patching workflows.

Setup (once):
  npm install -g @anthropic-ai/claude-code
  # or: curl -fsSL https://claude.ai/install.sh | bash

Recommended models for code tasks (via Ollama):
  qwen3-coder — 128k ctx, excellent coding, 4.5 GB
  glm-4.7      — 128k ctx, fast iteration, 4.2 GB
  qwen3:8b     — already in our Ollama, good baseline

Usage:
  from core.claude_code_bridge import ClaudeCodeBridge
  result = await ClaudeCodeBridge().run_task("Add error handling to core/groq_connector.py")
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# Env-configurable
CLAUDE_OLLAMA_URL   = os.getenv("CLAUDE_OLLAMA_URL",   "http://localhost:11434")
CLAUDE_LOCAL_MODEL  = os.getenv("CLAUDE_LOCAL_MODEL",  "qwen3:8b")
CLAUDE_LOCAL_ENABLED = os.getenv("CLAUDE_LOCAL_ENABLED", "true").lower() == "true"

# Claude Code binary name
_CLAUDE_BIN = shutil.which("claude") or shutil.which("claude-code")


def is_available() -> bool:
    """True if claude-code CLI is installed and CLAUDE_LOCAL_ENABLED."""
    return bool(_CLAUDE_BIN) and CLAUDE_LOCAL_ENABLED


class ClaudeCodeBridge:
    """
    Spawn claude-code CLI as async subprocess, routing to local Ollama.
    Useful for AutoGenesis V2 self-patching and multi-file agentic tasks.
    """

    def __init__(
        self,
        model: str = CLAUDE_LOCAL_MODEL,
        ollama_url: str = CLAUDE_OLLAMA_URL,
        cwd: str = ".",
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.cwd = cwd
        self._env = {
            **os.environ,
            "ANTHROPIC_BASE_URL":   ollama_url,
            "ANTHROPIC_AUTH_TOKEN": "ollama",
            "ANTHROPIC_API_KEY":    "",       # overwrite any real key — use local
        }

    async def run_task(
        self,
        task: str,
        timeout: int = 120,
        extra_flags: Optional[list[str]] = None,
    ) -> str:
        """
        Run a task via claude-code CLI (non-interactive, headless mode).
        Returns stdout output.
        Raises RuntimeError if claude-code not installed or times out.
        """
        if not _CLAUDE_BIN:
            raise RuntimeError(
                "claude-code CLI not installed.
"
                "Install: npm install -g @anthropic-ai/claude-code
"
                "Or:      curl -fsSL https://claude.ai/install.sh | bash"
            )

        flags = [
            "--model", self.model,
            "--print",                          # non-interactive / print mode
            "--allow-dangerously-skip-permissions",  # needed for headless
        ]
        if extra_flags:
            flags.extend(extra_flags)

        cmd = [_CLAUDE_BIN] + flags + [task]
        logger.info(f"ClaudeCode[{self.model}] task: {task[:80]}...")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.cwd,
            env=self._env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"ClaudeCodeBridge timed out after {timeout}s")

        output = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 and not output:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ClaudeCode exited {proc.returncode}: {err[:300]}")

        logger.debug(f"ClaudeCode done: {len(output)} chars output")

        # SparkNet capture
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            asyncio.ensure_future(
                get_sparknet().capture(
                    "claude_code_local",
                    f"ClaudeCode/{self.model} agentic task: {task[:100]}",
                    tags=["claude_code", "local", "ollama", "agentic"],
                )
            )
        except Exception:
            pass

        return output

    async def patch_file(self, filepath: str, instruction: str) -> str:
        """Shortcut: apply instruction to a specific file."""
        task = f"In file {filepath}: {instruction}"
        return await self.run_task(task, extra_flags=["--dangerously-force-output"])
