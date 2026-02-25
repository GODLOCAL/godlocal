"""
extensions/xzero/potpie_connector.py
PotpieConnector — Codebase knowledge graph + AI agents for GodLocal.

Based on: Potpie — "Spec-driven development for large codebases"
GitHub: https://github.com/potpie-ai/potpie
Tweet: @sukh_saroy (2026-02-25) — "builds AI agents that understand your code"

What it does:
  Turns any codebase into a knowledge graph (Neo4j) — every file, class,
  function, and their relationships. Prebuilt agents for debugging, Q&A,
  code generation, spec writing, test generation, and feature planning.
  Exposes a FastAPI server at localhost:8001 with a REST API + custom agents.

Architecture:
  Potpie FastAPI (localhost:8001) → Neo4j knowledge graph
    ├── Prebuilt agents: debugging, qa, codegen, spec, test, feature
    ├── Custom agents via POST /api/v1/custom-agents/agents/auto
    ├── Conversation sessions (multi-turn memory)
    └── LLM: configurable — openai | ollama | anthropic | openrouter

GodLocal integration:
  - PotpieConnector wraps the Potpie HTTP API at localhost:8001
  - 8 tools: query_agent, create_custom_agent, list_agents, create_conversation,
             send_message, get_parsing_status, parse_repo, list_conversations
  - Works alongside GitNexusMCPConnector — complementary:
      GitNexus: call graphs, clusters, static analysis
      Potpie:   conversational agents, spec generation, test writing
  - AutoGenesisV2 hook: call parse_repo() once, then query_agent() before patching

Setup (one-time):
  git clone --recurse-submodules https://github.com/potpie-ai/potpie.git && cd potpie
  cp .env.template .env
  # Edit .env: LLM_PROVIDER=ollama, CHAT_MODEL=ollama_chat/qwen3:8b
  ./scripts/start.sh          # starts FastAPI + Neo4j + PostgreSQL + Redis + Celery

Usage:
  from extensions.xzero.potpie_connector import PotpieConnector
  pc = PotpieConnector()
  conv = await pc.run_tool("create_conversation", project_id="godlocal", agent="debugging")
  result = await pc.run_tool("send_message", conversation_id=conv["id"], message="Why does AgentPool fail silently?")

  # Or via get_connector lazy load:
  PotpieConnector = get_connector("potpie")
  pc = PotpieConnector()
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from extensions.xzero.cimd_connector_base import CIMDConnector, CIMDTool


# ── Response models ───────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    content: str
    conversation_id: str
    agent_type: str
    citations: list[dict]


@dataclass
class ConversationSession:
    id: str
    project_id: str
    agent_type: str
    created_at: str


# ── Connector ─────────────────────────────────────────────────────────────────

class PotpieConnector(CIMDConnector):
    """
    CIMD connector for Potpie local server.
    Gives GodLocal agents a full codebase-aware AI layer:
    - Conversational debugging with stacktrace analysis
    - Natural language codebase Q&A
    - Spec / PRD / architecture document generation
    - Test generation grounded in actual code structure
    - Feature planning from requirements to implementation

    Modes:
      HTTP mode (default): calls Potpie FastAPI at localhost:8001
      Configured by env vars below.

    Potpie agents:
      "debugging"  — stacktrace root-cause + fix path
      "qa"         — codebase Q&A and architecture explanation
      "codegen"    — feature/refactor code generation
      "spec"       — PRD, architecture docs, software specs
      "test"       — unit + integration test generation
      "feature"    — low-level implementation plan from requirement
    """

    name = "potpie"

    SERVER_PORT   = int(os.getenv("POTPIE_PORT", "8001"))
    SERVER_URL    = os.getenv("POTPIE_URL", f"http://localhost:{int(os.getenv('POTPIE_PORT', '8001'))}")
    DEFAULT_USER  = os.getenv("POTPIE_USER", "defaultuser")   # isDevelopmentMode user

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get(self, path: str, **params) -> dict:
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp required: pip install aiohttp")
        url = f"{self.SERVER_URL}{path}"
        async with aiohttp.ClientSession() as s:
            async with s.get(url, params=params or None) as r:
                r.raise_for_status()
                return await r.json()

    async def _post(self, path: str, payload: dict | None = None) -> dict:
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp required: pip install aiohttp")
        url = f"{self.SERVER_URL}{path}"
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload or {}) as r:
                r.raise_for_status()
                return await r.json()

    # ── Tools ─────────────────────────────────────────────────────────────────

    async def parse_repo(
        self,
        repo_url: str = "",
        branch: str = "main",
        local_path: str = "",
    ) -> dict:
        """
        Parse a repository into the Potpie knowledge graph.
        Use repo_url for GitHub repos or local_path for local codebases.
        Returns project_id to use in subsequent calls.
        """
        payload: dict[str, Any] = {"branch": branch}
        if repo_url:
            payload["repo_path"] = repo_url
        elif local_path:
            payload["repo_path"] = local_path
        else:
            payload["repo_path"] = "."
        result = await self._post("/api/v1/parse", payload)
        return result  # {"project_id": "...", "status": "..."}

    async def get_parsing_status(self, project_id: str) -> dict:
        """Check parsing/indexing status for a project."""
        return await self._get(f"/api/v1/parsing-status/{project_id}")

    async def list_agents(self) -> dict:
        """List available prebuilt + custom agents."""
        return await self._get("/api/v1/agents")

    async def create_conversation(
        self,
        project_id: str,
        agent: str = "qa",
        title: str = "",
    ) -> dict:
        """
        Start a new conversation session with a Potpie agent.
        agent: "debugging" | "qa" | "codegen" | "spec" | "test" | "feature"
        Returns conversation_id for use in send_message.
        """
        payload = {
            "project_id": project_id,
            "agent_ids": [agent],
            "title": title or f"GodLocal/{agent} session",
        }
        return await self._post("/api/v1/conversations", payload)

    async def send_message(
        self,
        conversation_id: str,
        message: str,
    ) -> dict:
        """
        Send a message to an active conversation.
        Returns agent response with content and citations.
        """
        return await self._post(
            f"/api/v1/conversations/{conversation_id}/message",
            {"content": message},
        )

    async def query_agent(
        self,
        project_id: str,
        question: str,
        agent: str = "qa",
    ) -> dict:
        """
        One-shot: create conversation + send message, return answer.
        Convenience wrapper for single Q&A without managing sessions.
        """
        conv = await self.create_conversation(project_id=project_id, agent=agent)
        conv_id = conv.get("id") or conv.get("conversation_id", "")
        if not conv_id:
            return {"error": "Failed to create conversation", "raw": conv}
        return await self.send_message(conversation_id=conv_id, message=question)

    async def create_custom_agent(self, prompt: str) -> dict:
        """
        Create a custom agent from a natural-language description.
        Example prompt: "An agent that takes stacktrace and returns root cause + fix"
        Returns agent_id for use in create_conversation.
        """
        return await self._post(
            "/api/v1/custom-agents/agents/auto",
            {"prompt": prompt},
        )

    async def list_conversations(self, project_id: str = "") -> dict:
        """List all conversation sessions, optionally filtered by project."""
        params = {"project_id": project_id} if project_id else {}
        return await self._get("/api/v1/conversations", **params)

    async def health_check(self) -> dict:
        """Check if Potpie server is running."""
        try:
            return await self._get("/health")
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    # ── CIMD tool registry ────────────────────────────────────────────────────

    @property
    def tools(self) -> list[CIMDTool]:  # type: ignore[override]
        return [
            CIMDTool(
                name="parse_repo",
                description="Parse a GitHub repo or local path into Potpie knowledge graph. Returns project_id.",
                fn=self.parse_repo,
                input_schema={
                    "type": "object",
                    "properties": {
                        "repo_url": {"type": "string", "description": "GitHub repo URL (e.g. https://github.com/GODLOCAL/godlocal)"},
                        "branch":   {"type": "string", "description": "Branch to index (default: main)"},
                        "local_path": {"type": "string", "description": "Absolute local path to repo (alternative to repo_url)"},
                    },
                },
            ),
            CIMDTool(
                name="get_parsing_status",
                description="Check indexing status for a parsed project.",
                fn=self.get_parsing_status,
                input_schema={
                    "type": "object",
                    "required": ["project_id"],
                    "properties": {"project_id": {"type": "string"}},
                },
            ),
            CIMDTool(
                name="list_agents",
                description="List all available Potpie agents (prebuilt + custom).",
                fn=self.list_agents,
                input_schema={"type": "object", "properties": {}},
            ),
            CIMDTool(
                name="create_conversation",
                description="Start a conversation with a Potpie agent. Returns conversation_id.",
                fn=self.create_conversation,
                input_schema={
                    "type": "object",
                    "required": ["project_id"],
                    "properties": {
                        "project_id": {"type": "string"},
                        "agent": {
                            "type": "string",
                            "enum": ["debugging", "qa", "codegen", "spec", "test", "feature"],
                            "description": "Agent type (default: qa)",
                        },
                        "title": {"type": "string", "description": "Optional session title"},
                    },
                },
            ),
            CIMDTool(
                name="send_message",
                description="Send a message to an active Potpie conversation. Returns agent response.",
                fn=self.send_message,
                input_schema={
                    "type": "object",
                    "required": ["conversation_id", "message"],
                    "properties": {
                        "conversation_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            ),
            CIMDTool(
                name="query_agent",
                description="One-shot codebase Q&A: creates session + asks question, returns answer. Best for single questions.",
                fn=self.query_agent,
                input_schema={
                    "type": "object",
                    "required": ["project_id", "question"],
                    "properties": {
                        "project_id": {"type": "string"},
                        "question":   {"type": "string"},
                        "agent": {
                            "type": "string",
                            "enum": ["debugging", "qa", "codegen", "spec", "test", "feature"],
                            "description": "Agent type (default: qa)",
                        },
                    },
                },
            ),
            CIMDTool(
                name="create_custom_agent",
                description="Create a custom Potpie agent from a natural-language prompt. Returns agent_id.",
                fn=self.create_custom_agent,
                input_schema={
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {"prompt": {"type": "string", "description": "What the agent should do"}},
                },
            ),
            CIMDTool(
                name="list_conversations",
                description="List all conversation sessions, optionally filtered by project_id.",
                fn=self.list_conversations,
                input_schema={
                    "type": "object",
                    "properties": {"project_id": {"type": "string", "description": "Filter by project (optional)"}},
                },
            ),
        ]

    # ── OpenAPI schema (CIMD discovery) ───────────────────────────────────────

    def openapi_schema(self) -> dict:
        paths: dict[str, Any] = {}
        for tool in self.tools:
            paths[f"/tools/{tool.name}/run"] = {
                "post": tool.to_openapi_operation()
            }
        return {
            "openapi": "3.1.0",
            "info": {
                "title": "GodLocal X-ZERO / potpie",
                "version": "1.0.0",
                "description": (
                    "Potpie codebase knowledge graph connector. "
                    "Discover tools via GET /openapi.json, "
                    "execute via POST /tools/{tool}/run. "
                    "Setup: git clone potpie-ai/potpie && ./scripts/start.sh"
                ),
            },
            "paths": paths,
        }
