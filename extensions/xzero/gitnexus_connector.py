"""
extensions/xzero/gitnexus_connector.py
GitNexusMCPConnector — Codebase knowledge graph for GodLocal agents.

Based on: GitNexus — "The Zero-Server Code Intelligence Engine"
GitHub: https://github.com/abhigyanpatwari/GitNexus
Viral: @ihtesham2005 / @roundtablespace (~2.5K bookmarks, 123K impressions, 2026-02-25)

What it does:
  Indexes any GitHub repo or local codebase into a knowledge graph:
  every dependency, call chain, cluster, execution flow.
  Exposes an MCP server (stdio) + local HTTP server for AI agent Q&A.

Architecture:
  gitnexus analyze → KuzuDB graph (Tree-sitter AST + HuggingFace embeddings)
  gitnexus mcp     → MCP stdio server (Cursor/Claude Code/Windsurf)
  gitnexus serve   → HTTP server at localhost:3399 (GodLocal integration point)

GodLocal integration:
  - GitNexusMCPConnector wraps the HTTP API exposed by `gitnexus serve`
  - 8 tools: analyze_repo, query_graph, search_code, get_call_chain,
             get_clusters, find_entry_points, explain_file, generate_wiki
  - Works with SkillOrchestraRouter: skill "analyze_onchain" → "code_python"
  - AutoGenesisV2 can call `analyze_repo` before patching to understand impact

Setup (one-time):
  npm install -g gitnexus
  gitnexus analyze .           # index godlocal repo
  gitnexus serve               # starts HTTP at localhost:3399

Usage:
  from extensions.xzero.gitnexus_connector import GitNexusMCPConnector
  gn = GitNexusMCPConnector()
  result = await gn.run_tool("query_graph", {"question": "How does AgentPool route tasks?"})
"""
from __future__ import annotations

import json
import os
import subprocess
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from extensions.xzero.cimd_connector_base import CIMDConnectorBase


# ── GitNexus HTTP server response models ─────────────────────────────────────

@dataclass
class GraphQueryResult:
    answer: str
    sources: list[dict]
    graph_context: dict


@dataclass
class CodeSearchResult:
    matches: list[dict]   # [{file, line, symbol, snippet, score}]
    total: int


# ── Connector ─────────────────────────────────────────────────────────────────

class GitNexusMCPConnector(CIMDConnectorBase):
    """
    CIMD connector for GitNexus local MCP/HTTP server.
    Gives GodLocal agents codebase-level understanding:
    - Query the call graph in natural language
    - Find where a function is called
    - Understand cluster/module relationships
    - Generate wiki docs

    The connector operates in two modes:
      HTTP mode (default): calls `gitnexus serve` REST API at localhost:3399
      CLI mode (fallback):  runs gitnexus CLI subprocess directly
    """

    SERVER_PORT = int(os.getenv("GITNEXUS_PORT", "3399"))
    SERVER_URL  = f"http://localhost:{SERVER_PORT}"
    REPO_PATH   = os.getenv("GITNEXUS_REPO_PATH", ".")
    NODE_BIN    = os.getenv("GITNEXUS_NODE_BIN", "gitnexus")

    # ── CIMDConnectorBase interface ──────────────────────────────────────────

    @classmethod
    def openapi_schema(cls) -> dict:
        return {
            "openapi": "3.1.0",
            "info": {
                "title": "GitNexus MCP Connector",
                "version": "1.0.0",
                "description": "Codebase knowledge graph and AI agent Q&A via GitNexus.",
            },
            "paths": {
                "/tools/analyze_repo":     {"post": {"summary": "Index or re-index a local repo"}},
                "/tools/query_graph":      {"post": {"summary": "Ask a question about the codebase"}},
                "/tools/search_code":      {"post": {"summary": "Hybrid BM25+semantic code search"}},
                "/tools/get_call_chain":   {"post": {"summary": "Trace execution flow from an entry point"}},
                "/tools/get_clusters":     {"post": {"summary": "List functional module clusters"}},
                "/tools/find_entry_points":{"post": {"summary": "Find all entry points in the repo"}},
                "/tools/explain_file":     {"post": {"summary": "Explain a file's role in the codebase"}},
                "/tools/generate_wiki":    {"post": {"summary": "Generate LLM-powered wiki documentation"}},
            },
        }

    @classmethod
    def registration_manifest(cls) -> dict:
        return {
            "name": "GitNexusMCPConnector",
            "id":   "gitnexus",
            "description": "Codebase knowledge graph — every dependency, call chain, cluster, entry point.",
            "env_vars": [],  # No API key required — local-only
            "tools": list(cls.openapi_schema()["paths"].keys()),
            "setup": [
                "npm install -g gitnexus",
                "gitnexus analyze .",
                "gitnexus serve   # starts localhost:3399",
            ],
        }

    async def run_tool(self, tool: str, params: dict) -> dict:
        """Route to the appropriate method."""
        dispatch = {
            "analyze_repo":      self._analyze_repo,
            "query_graph":       self._query_graph,
            "search_code":       self._search_code,
            "get_call_chain":    self._get_call_chain,
            "get_clusters":      self._get_clusters,
            "find_entry_points": self._find_entry_points,
            "explain_file":      self._explain_file,
            "generate_wiki":     self._generate_wiki,
        }
        handler = dispatch.get(tool)
        if not handler:
            return {"error": f"Unknown tool: {tool}"}
        return await handler(**params)

    # ── HTTP helper ──────────────────────────────────────────────────────────

    async def _http(self, endpoint: str, payload: dict) -> dict:
        """POST to gitnexus serve HTTP API. Falls back to CLI on connection error."""
        if not HAS_AIOHTTP:
            return await self._cli_fallback(endpoint, payload)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.SERVER_URL}{endpoint}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"error": f"HTTP {resp.status}", "detail": await resp.text()}
        except aiohttp.ClientConnectorError:
            # Server not running — fall back to CLI
            return await self._cli_fallback(endpoint, payload)

    async def _cli_fallback(self, endpoint: str, payload: dict) -> dict:
        """Run gitnexus CLI directly when HTTP server is not available."""
        tool = endpoint.split("/")[-1]
        cmd = [self.NODE_BIN]
        if tool == "analyze_repo":
            cmd += ["analyze", payload.get("path", self.REPO_PATH)]
            if payload.get("force"):
                cmd.append("--force")
        elif tool == "generate_wiki":
            cmd += ["wiki", payload.get("path", self.REPO_PATH)]
            if payload.get("model"):
                cmd += ["--model", payload["model"]]
        else:
            return {"error": f"CLI fallback not supported for tool: {tool}. Run `gitnexus serve` first."}

        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=300
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except FileNotFoundError:
            return {
                "error": "gitnexus not installed",
                "setup": "npm install -g gitnexus && gitnexus analyze . && gitnexus serve",
            }

    # ── Tool implementations ─────────────────────────────────────────────────

    async def _analyze_repo(
        self, path: str = ".", force: bool = False, skip_embeddings: bool = False
    ) -> dict:
        """
        Index a repo into KuzuDB knowledge graph.
        Phases: Structure → Parse → Resolve → Cluster → Process → Search index.
        """
        return await self._http("/api/analyze", {
            "path": path, "force": force, "skip_embeddings": skip_embeddings
        })

    async def _query_graph(self, question: str, repo_path: str = ".") -> dict:
        """
        Ask a natural language question about the codebase.
        Uses LangChain ReAct agent + BM25/semantic hybrid search + graph traversal.

        Examples:
          "How does AgentPool route tasks to agents?"
          "What calls sleep_cycle() and in what order?"
          "Where is MOONPAY_API_KEY read and validated?"
        """
        return await self._http("/api/query", {"question": question, "repo": repo_path})

    async def _search_code(self, query: str, top_k: int = 10, repo_path: str = ".") -> dict:
        """
        Hybrid code search: BM25 + semantic embeddings + RRF fusion.
        Returns ranked list of matching symbols with file/line/snippet.
        """
        return await self._http("/api/search", {
            "query": query, "top_k": top_k, "repo": repo_path
        })

    async def _get_call_chain(self, entry_point: str, repo_path: str = ".") -> dict:
        """
        Trace the execution flow from a function/method entry point.
        Returns full call graph from entry_point downwards.
        Use before patching — shows what breaks if you change a function.

        Example: entry_point="sleep_cycle" → traces all called functions
        """
        return await self._http("/api/call-chain", {
            "entry_point": entry_point, "repo": repo_path
        })

    async def _get_clusters(self, repo_path: str = ".") -> dict:
        """
        List functional module clusters (Graphology community detection).
        Shows how the codebase is organized into logical groups.
        """
        return await self._http("/api/clusters", {"repo": repo_path})

    async def _find_entry_points(self, repo_path: str = ".") -> dict:
        """
        Detect all entry points: main() functions, FastAPI routes,
        CLI commands, scheduled tasks, webhook handlers.
        """
        return await self._http("/api/entry-points", {"repo": repo_path})

    async def _explain_file(self, file_path: str, repo_path: str = ".") -> dict:
        """
        Explain a file's role: what it exports, what imports it,
        its cluster membership, and how it fits the overall architecture.

        Example: file_path="core/skill_orchestra.py"
        """
        return await self._http("/api/explain-file", {
            "file": file_path, "repo": repo_path
        })

    async def _generate_wiki(
        self, path: str = ".", model: str | None = None
    ) -> dict:
        """
        Generate LLM-powered documentation wiki for the entire codebase.
        Outputs structured markdown docs per module/cluster.
        """
        payload: dict = {"path": path}
        if model:
            payload["model"] = model
        return await self._http("/api/wiki", payload)

    # ── AutoGenesis integration ──────────────────────────────────────────────

    async def pre_patch_analysis(self, target_file: str) -> str:
        """
        Called by AutoGenesisV2 before applying a patch.
        Returns: impact summary (what calls target_file + its cluster).
        Prevents AutoGenesis from breaking dependencies unknowingly.
        """
        file_info = await self._explain_file(target_file)
        call_info = await self._get_call_chain(
            Path(target_file).stem   # function name heuristic from filename
        )
        return json.dumps({
            "file_role":    file_info.get("explanation", ""),
            "call_chain":   call_info.get("chain", []),
            "cluster":      file_info.get("cluster", ""),
            "dependents":   file_info.get("imported_by", []),
        }, indent=2)
