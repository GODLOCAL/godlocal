"""
cimd_connector_base.py
GodLocal X-ZERO — CIMD-native connector base
Based on: @RhysSullivan — REST API + CIMD (MCP dynamic registration spec)
          instead of CLI interfaces for agents

Every X-ZERO connector should subclass CIMDConnector:
- exposes /openapi.json  (tool discovery)
- exposes POST /tools/{tool_name}/run  (execution)
- supports dynamic registration via CIMD handshake (same spec as MCP)
- agent writes inline code to call any discovered tool

Usage:
    class JupiterConnector(CIMDConnector):
        name = "jupiter"
        tools = [swap_tool, price_tool]

    connector = JupiterConnector()
    connector.serve(port=9101)
"""

from __future__ import annotations
import json, inspect
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class CIMDTool:
    """Describes one callable tool exposed over REST."""
    name: str
    description: str
    fn: Callable
    input_schema: dict = field(default_factory=dict)

    def to_openapi_operation(self) -> dict:
        return {
            "summary": self.description,
            "operationId": self.name,
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": self.input_schema or {"type": "object"}
                    }
                }
            },
            "responses": {"200": {"description": "Tool result"}}
        }

    async def run(self, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return self.fn(**kwargs)


class CIMDConnector:
    """
    Base class for GodLocal CIMD-native connectors.

    CIMD = Client-Initiated Message Dispatch (same dynamic registration
    spec used by MCP). Each connector self-describes via OpenAPI so the
    GodLocal agent can discover and call any tool without a CLI wrapper.

    Subclass and define:
        name: str                   # connector identifier
        tools: list[CIMDTool]       # tools to expose
    """

    name: str = "base"
    tools: list[CIMDTool] = []

    # ── Discovery ──────────────────────────────────────────────────────────

    def openapi_schema(self) -> dict:
        """Returns OpenAPI 3.1 schema for dynamic agent discovery."""
        paths: dict[str, Any] = {}
        for tool in self.tools:
            paths[f"/tools/{tool.name}/run"] = {
                "post": tool.to_openapi_operation()
            }
        return {
            "openapi": "3.1.0",
            "info": {
                "title": f"GodLocal X-ZERO / {self.name}",
                "version": "1.0.0",
                "description": (
                    f"CIMD-native connector for {self.name}. "
                    "Discover tools via GET /openapi.json, "
                    "execute via POST /tools/{tool}/run"
                )
            },
            "paths": paths
        }

    # ── Registration handshake (CIMD / MCP-spec) ──────────────────────────

    def registration_manifest(self, base_url: str) -> dict:
        """
        CIMD dynamic registration payload.
        Agent sends this to the GodLocal connector registry so the
        LLM can discover and call the connector without a CLI.
        """
        return {
            "connector_id": self.name,
            "base_url": base_url,
            "openapi_url": f"{base_url}/openapi.json",
            "tools": [
                {"name": t.name, "description": t.description}
                for t in self.tools
            ]
        }

    # ── Execution ─────────────────────────────────────────────────────────

    async def run_tool(self, tool_name: str, **kwargs: Any) -> Any:
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.run(**kwargs)
        raise ValueError(f"Tool '{tool_name}' not found in {self.name} connector")

    # ── FastAPI mount helper ───────────────────────────────────────────────

    def mount(self, app: Any, prefix: str = "") -> None:
        """
        Mount this connector's routes onto a FastAPI app.

        Example:
            from fastapi import FastAPI
            app = FastAPI()
            connector = JupiterConnector()
            connector.mount(app, prefix="/connectors/jupiter")
        """
        try:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse

            route_prefix = prefix or f"/connectors/{self.name}"

            @app.get(f"{route_prefix}/openapi.json")
            async def _openapi():
                return JSONResponse(self.openapi_schema())

            for tool in self.tools:
                # Closure to capture tool reference correctly
                def make_route(t: CIMDTool):
                    async def _run(body: dict = {}):
                        result = await t.run(**body)
                        return {"result": result, "tool": t.name}
                    _run.__name__ = f"run_{t.name}"
                    return _run

                app.post(f"{route_prefix}/tools/{tool.name}/run")(make_route(tool))

        except ImportError:
            raise RuntimeError("FastAPI required: pip install fastapi")
