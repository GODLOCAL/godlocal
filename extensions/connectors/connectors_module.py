"""
extensions/connectors/connectors_module.py — GodLocal ConnectorsModule
500+ service integrations via Composio SDK + built-in agents.

Architecture:
  ConnectorsModule
    ├── ComposioConnector   — Composio SDK wrapper (500+ apps)
    ├── built-in agents/
    │     ├── ReadmeGeneratorAgent
    │     └── (extensible — add more in agents/)
    └── FastAPI router      — /connectors/* endpoints

Usage:
  from extensions.connectors.connectors_module import ConnectorsModule
  connectors = ConnectorsModule()
  connectors.mount(app)          # mount FastAPI router
  readme = await connectors.readme_generator.generate("GODLOCAL/godlocal")
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import pkgutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("godlocal.connectors")

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from composio import ComposioToolSet, Action, App
    COMPOSIO_AVAILABLE = True
except ImportError:
    COMPOSIO_AVAILABLE = False
    logger.info("[ConnectorsModule] composio-core not installed — pip install composio-core")

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ── Composio connector ────────────────────────────────────────────────────────

class ComposioConnector:
    """
    Thin wrapper around Composio ToolSet.
    Handles auth, action execution, and tool search.

    Required env: COMPOSIO_API_KEY
    """

    def __init__(self, api_key: str | None = None):
        self.api_key   = api_key or os.getenv("COMPOSIO_API_KEY", "")
        self._toolset  = None

    def _get_toolset(self) -> "ComposioToolSet":
        if not COMPOSIO_AVAILABLE:
            raise RuntimeError("composio-core not installed: pip install composio-core")
        if self._toolset is None:
            self._toolset = ComposioToolSet(api_key=self.api_key or None)
        return self._toolset

    def execute(self, action: str, params: dict) -> dict:
        """Execute a Composio action by slug. Returns result dict."""
        ts = self._get_toolset()
        result = ts.execute_action(action=action, params=params)
        return result

    def get_tools(self, apps: list[str] | None = None, tags: list[str] | None = None) -> list:
        """Get tools for LLM function-calling (OpenAI / LangChain schema)."""
        ts = self._get_toolset()
        kwargs: dict[str, Any] = {}
        if apps:
            kwargs["apps"] = [App(a.upper()) for a in apps]
        if tags:
            kwargs["tags"] = tags
        return ts.get_tools(**kwargs)

    def search_actions(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search over 30,000+ Composio actions."""
        ts = self._get_toolset()
        try:
            actions = ts.find_actions_by_use_case(use_case=query, advanced=False)
            return [{"action": a.name, "app": a.app_name, "description": a.description}
                    for a in actions[:limit]]
        except Exception as e:
            logger.warning(f"[Composio] search_actions failed: {e}")
            return []

    @property
    def available(self) -> bool:
        return COMPOSIO_AVAILABLE and bool(self.api_key)


# ── Built-in agent base ───────────────────────────────────────────────────────

class BaseConnectorAgent:
    """Base class for built-in connector agents."""
    name: str = "base"
    description: str = ""

    def __init__(self, llm=None, composio: ComposioConnector | None = None):
        self.llm      = llm       # GodLocal LLMBridge instance
        self.composio = composio

    async def run(self, *args, **kwargs) -> dict:
        raise NotImplementedError


# ── Readme Generator Agent ────────────────────────────────────────────────────

class ReadmeGeneratorAgent(BaseConnectorAgent):
    """
    Generates a high-quality README.md for a GitHub repository.

    Sources:
      1. GitHub API (via Composio) — repo metadata, file tree, package.json / requirements.txt
      2. LLM synthesis — structured README in GodLocal style
      3. Optional: auto-commit to repo via Composio GITHUB_COMMIT_MULTIPLE_FILES

    Usage:
      agent = ReadmeGeneratorAgent(llm=bridge, composio=connector)
      result = await agent.run("GODLOCAL/godlocal")
      print(result["readme"])
    """
    name        = "readme_generator"
    description = "Generate high-quality README.md for any GitHub repo using AI + Composio"

    # README template sections
    TEMPLATE = """# {repo_name}

> {description}

## What it does

{what_it_does}

## Quick Start

```bash
{quick_start}
```

## Architecture

{architecture}

## Key Features

{features}

## Configuration

{configuration}

## License

{license}
"""

    PROMPT = """You are a senior technical writer generating a README.md for a GitHub repository.

## Repository Metadata
{metadata}

## File Tree (top-level)
{file_tree}

## Key Files Content
{key_files}

## Instructions
Generate a high-quality README.md. Use this exact structure:
1. Title + one-line description
2. What it does (2-3 sentences, concrete, no marketing fluff)
3. Quick Start (minimal working example, ≤5 commands)
4. Architecture (bullet points, key modules/files)
5. Key Features (max 6 bullets)
6. Configuration (env vars, config files)
7. License

Rules:
- Technical, precise, zero filler
- Match the project's actual tech stack
- Quick Start must actually work
- Output only the markdown README, no meta-commentary
"""

    async def run(
        self,
        repo: str,                        # "owner/repo"
        branch: str = "main",
        auto_commit: bool = False,        # if True, commits README to repo
    ) -> dict:
        """Generate README for repo. Returns {"readme": str, "committed": bool}."""
        owner, repo_name = repo.split("/", 1)

        # 1. Fetch repo metadata via Composio (or fallback to GitHub API)
        metadata = await self._fetch_metadata(owner, repo_name, branch)
        file_tree = metadata.get("file_tree", [])
        key_files = await self._fetch_key_files(owner, repo_name, branch, file_tree)

        # 2. Build prompt
        prompt = self.PROMPT.format(
            metadata=self._format_metadata(metadata),
            file_tree="\n".join(f"  {f}" for f in file_tree[:40]),
            key_files=key_files,
        )

        # 3. LLM generation
        if self.llm:
            readme = self.llm.generate(prompt, max_tokens=2000)
        else:
            readme = f"# {repo_name}\n\n> README auto-generated (LLM unavailable)\n"

        # 4. Optional auto-commit
        committed = False
        if auto_commit and self.composio and self.composio.available:
            try:
                self.composio.execute(
                    action="GITHUB_COMMIT_MULTIPLE_FILES",
                    params={
                        "owner": owner,
                        "repo": repo_name,
                        "branch": branch,
                        "message": "docs: auto-generate README.md via GodLocal ReadmeGeneratorAgent",
                        "upserts": [{"path": "README.md", "content": readme, "encoding": "utf-8"}],
                    }
                )
                committed = True
                logger.info(f"[ReadmeGenerator] Committed README to {repo}")
            except Exception as e:
                logger.warning(f"[ReadmeGenerator] auto-commit failed: {e}")

        return {"repo": repo, "readme": readme, "committed": committed}

    async def _fetch_metadata(self, owner: str, repo: str, branch: str) -> dict:
        """Fetch repo metadata: description, stars, topics, top-level file list."""
        meta: dict[str, Any] = {"owner": owner, "repo": repo, "branch": branch}
        if self.composio and self.composio.available:
            try:
                result = self.composio.execute(
                    action="GITHUB_GET_REPOSITORY",
                    params={"owner": owner, "repo": repo}
                )
                r = result.get("data", {})
                meta["description"] = r.get("description", "")
                meta["stars"]       = r.get("stargazers_count", 0)
                meta["language"]    = r.get("language", "")
                meta["topics"]      = r.get("topics", [])
                meta["license"]     = r.get("license", {}).get("name", "")
                meta["homepage"]    = r.get("homepage", "")
            except Exception as e:
                logger.debug(f"[ReadmeGenerator] metadata fetch failed: {e}")

            try:
                tree_result = self.composio.execute(
                    action="GITHUB_GET_REPOSITORY_CONTENT",
                    params={"owner": owner, "repo": repo, "path": "", "branch": branch}
                )
                items = tree_result.get("data", {}).get("content", [])
                meta["file_tree"] = [
                    f"{'/' if i.get('type') == 'dir' else ''}{i.get('name', '')}"
                    for i in (items if isinstance(items, list) else [])
                ]
            except Exception as e:
                logger.debug(f"[ReadmeGenerator] file tree fetch failed: {e}")

        return meta

    async def _fetch_key_files(
        self, owner: str, repo: str, branch: str, file_tree: list[str]
    ) -> str:
        """Fetch content of key files: requirements.txt, CLAUDE.md, pyproject.toml, etc."""
        KEY_FILES = ["requirements.txt", "pyproject.toml", "package.json", "CLAUDE.md", "Dockerfile"]
        present   = [f for f in KEY_FILES if f in file_tree or f.lstrip("/") in file_tree]
        chunks: list[str] = []

        for fname in present[:3]:  # max 3 files to stay within token budget
            if self.composio and self.composio.available:
                try:
                    result = self.composio.execute(
                        action="GITHUB_GET_REPOSITORY_CONTENT",
                        params={"owner": owner, "repo": repo, "path": fname, "branch": branch}
                    )
                    import base64 as _b64
                    raw = result.get("data", {}).get("content", {})
                    if isinstance(raw, dict) and raw.get("encoding") == "base64":
                        content = _b64.b64decode(
                            raw["content"].replace("\n", "")
                        ).decode("utf-8", errors="replace")[:800]
                        chunks.append(f"### {fname}\n```\n{content}\n```")
                except Exception as e:
                    logger.debug(f"[ReadmeGenerator] {fname} fetch failed: {e}")

        return "\n\n".join(chunks) if chunks else "(key files unavailable)"

    def _format_metadata(self, meta: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in meta.items() if k != "file_tree" and v)


# ── ConnectorsModule ──────────────────────────────────────────────────────────

class ConnectorsModule:
    """
    GodLocal ConnectorsModule — main entry point.

    Usage in godlocal_v5.py:
        from extensions.connectors.connectors_module import ConnectorsModule
        connectors = ConnectorsModule(llm=agent.llm)
        connectors.mount(app)      # registers /connectors/* FastAPI routes

    Or standalone:
        connectors = ConnectorsModule()
        result = asyncio.run(connectors.readme_generator.run("GODLOCAL/godlocal"))
    """

    def __init__(self, llm=None, composio_api_key: str | None = None):
        self.llm       = llm
        self.composio  = ComposioConnector(api_key=composio_api_key)
        self._agents: dict[str, BaseConnectorAgent] = {}
        self._router   = None

        # Register built-in agents
        self._register_agent(ReadmeGeneratorAgent(llm=self.llm, composio=self.composio))

        # Auto-discover agents in extensions/connectors/agents/
        self._autodiscover_agents()

        logger.info(
            f"[ConnectorsModule] Ready — "
            f"Composio={'✓' if self.composio.available else '✗ (set COMPOSIO_API_KEY)'} | "
            f"Agents: {list(self._agents.keys())}"
        )

    # ── Agent registry ────────────────────────────────────────────────────────

    def _register_agent(self, agent: BaseConnectorAgent) -> None:
        self._agents[agent.name] = agent
        logger.debug(f"[ConnectorsModule] Registered agent: {agent.name}")

    def _autodiscover_agents(self) -> None:
        """Auto-load any BaseConnectorAgent subclasses from extensions/connectors/agents/."""
        agents_dir = Path(__file__).parent / "agents"
        if not agents_dir.exists():
            return
        for _, mod_name, _ in pkgutil.iter_modules([str(agents_dir)]):
            try:
                mod = importlib.import_module(f"extensions.connectors.agents.{mod_name}")
                for attr_name in dir(mod):
                    cls = getattr(mod, attr_name)
                    if (
                        isinstance(cls, type)
                        and issubclass(cls, BaseConnectorAgent)
                        and cls is not BaseConnectorAgent
                        and cls.name != "base"
                    ):
                        self._register_agent(cls(llm=self.llm, composio=self.composio))
            except Exception as e:
                logger.warning(f"[ConnectorsModule] Failed to load agent {mod_name}: {e}")

    # ── Agent accessors ───────────────────────────────────────────────────────

    @property
    def readme_generator(self) -> ReadmeGeneratorAgent:
        return self._agents["readme_generator"]  # type: ignore

    def agent(self, name: str) -> BaseConnectorAgent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not registered. Available: {list(self._agents.keys())}")
        return self._agents[name]

    # ── Composio direct passthrough ───────────────────────────────────────────

    def execute(self, action: str, params: dict) -> dict:
        """Execute any Composio action directly."""
        return self.composio.execute(action, params)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search over Composio actions."""
        return self.composio.search_actions(query, limit)

    # ── FastAPI router ────────────────────────────────────────────────────────

    def mount(self, app) -> None:
        """Mount /connectors/* routes onto a FastAPI app."""
        if not FASTAPI_AVAILABLE:
            logger.warning("[ConnectorsModule] FastAPI not available — routes not mounted")
            return

        router = APIRouter(prefix="/connectors", tags=["connectors"])

        @router.get("/")
        async def list_agents():
            return {
                "agents":   list(self._agents.keys()),
                "composio": self.composio.available,
            }

        @router.post("/readme")
        async def generate_readme(repo: str, branch: str = "main", auto_commit: bool = False):
            """Generate README.md for a GitHub repository."""
            if "/" not in repo:
                raise HTTPException(400, "repo must be 'owner/repo'")
            try:
                result = await self.readme_generator.run(repo, branch, auto_commit)
                return result
            except Exception as e:
                raise HTTPException(500, str(e))

        @router.post("/execute")
        async def execute_action(action: str, params: dict = {}):
            """Execute a Composio action directly."""
            if not self.composio.available:
                raise HTTPException(503, "Composio not configured — set COMPOSIO_API_KEY")
            try:
                return self.composio.execute(action, params)
            except Exception as e:
                raise HTTPException(500, str(e))

        @router.get("/search")
        async def search_actions(q: str, limit: int = 10):
            """Semantic search over 30,000+ Composio actions."""
            if not self.composio.available:
                raise HTTPException(503, "Composio not configured — set COMPOSIO_API_KEY")
            return {"results": self.composio.search_actions(q, limit)}

        @router.post("/agent/{agent_name}/run")
        async def run_agent(agent_name: str, params: dict = {}):
            """Run any registered connector agent."""
            try:
                ag_instance = self.agent(agent_name)
            except KeyError as e:
                raise HTTPException(404, str(e))
            try:
                result = await ag_instance.run(**params)
                return result
            except Exception as e:
                raise HTTPException(500, str(e))

        app.include_router(router)
        self._router = router
        logger.info("[ConnectorsModule] Routes mounted: /connectors/*")
