"""
extensions/connectors/agents/paper_canvas_agent.py — GodLocal PaperCanvasAgent
Integrates with Paper Desktop via MCP (Model Context Protocol).

Paper Desktop = HTML/CSS canvas that any agent (Cursor/Claude Code/Codex) can
read and write. PaperCanvasAgent bridges GodLocal's evolution loop with Paper:

  • evolve_to_canvas(html)  — write HTML artefact to Paper canvas (live preview)
  • pull_from_canvas()      — read current HTML from canvas back into repo
  • push_dashboard()        — push FEP / sleep_cycle metrics as live Paper widget
  • render_component(spec)  — ask LLM to generate a UI component, write to canvas

MCP connection (Paper Desktop must be running):
  Default MCP server URL: ws://localhost:3333/mcp   (Paper Desktop default)
  Override via env:       PAPER_MCP_URL

Usage:
  from extensions.connectors.agents.paper_canvas_agent import PaperCanvasAgent
  agent = PaperCanvasAgent(llm=bridge)
  await agent.run(action="push_dashboard", data={"fep": 0.12, "evolution": 42})

Auto-discovered by ConnectorsModule on startup.
Route: POST /connectors/agent/paper_canvas/run
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from extensions.connectors.connectors_module import BaseConnectorAgent, ComposioConnector

logger = logging.getLogger("godlocal.paper")

# ── MCP client (mcp>=1.0.0, optional) ─────────────────────────────────────────
try:
    from mcp import ClientSession
    from mcp.client.websocket import websocket_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.info("[PaperCanvas] mcp package not installed — pip install mcp")


PAPER_MCP_URL = os.getenv("PAPER_MCP_URL", "ws://localhost:3333/mcp")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _call_paper(tool: str, args: dict) -> dict:
    """Send a single MCP tool call to Paper Desktop."""
    if not MCP_AVAILABLE:
        raise RuntimeError("mcp package not installed: pip install mcp>=1.0.0")
    async with websocket_client(PAPER_MCP_URL) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result.model_dump() if hasattr(result, "model_dump") else dict(result)


# ── PaperCanvasAgent ──────────────────────────────────────────────────────────

class PaperCanvasAgent(BaseConnectorAgent):
    """
    Write/read GodLocal artefacts on Paper Desktop canvas via MCP.

    Supported actions (pass as `action` param to run()):
      push_dashboard   — render FEP + sleep_cycle metrics as HTML widget on canvas
      push_html        — write arbitrary HTML to canvas (pass html= kwarg)
      pull_canvas      — read current canvas HTML back
      render_component — LLM generates UI component from spec, writes to canvas
      push_readme      — push repo README to canvas for visual editing
    """

    name        = "paper_canvas"
    description = "Read/write GodLocal UI artefacts on Paper Desktop canvas via MCP"

    # ── Component templates ────────────────────────────────────────────────────

    DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0a0a0a; color: #00FF41;
    font-family: 'JetBrains Mono', monospace;
    padding: 24px; min-height: 100vh;
  }}
  .header {{ font-size: 11px; letter-spacing: 3px; color: #00E5FF; margin-bottom: 20px; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .metric {{
    border: 1px solid #1a2a1a; padding: 16px; border-radius: 4px;
    background: #0d160d;
  }}
  .metric-label {{ font-size: 10px; color: #4a8a4a; text-transform: uppercase; letter-spacing: 2px; }}
  .metric-value {{ font-size: 28px; font-weight: bold; margin-top: 4px; }}
  .metric-sub {{ font-size: 10px; color: #3a6a3a; margin-top: 2px; }}
  .fep {{ color: #00FF41; }}
  .evo {{ color: #00E5FF; }}
  .soul {{ color: #7B2FFF; }}
  .timestamp {{ font-size: 10px; color: #2a4a2a; margin-top: 20px; }}
  .bar-bg {{ background: #0d160d; border-radius: 2px; height: 4px; margin-top: 8px; }}
  .bar-fill {{ height: 4px; border-radius: 2px; background: #00FF41; }}
</style>
</head>
<body>
  <div class="header">⬡ GODLOCAL — AUTOGENESIS METRICS</div>
  <div class="metric-grid">
    <div class="metric">
      <div class="metric-label">FEP Surprise</div>
      <div class="metric-value fep">{surprise:.4f}</div>
      <div class="metric-sub">free energy: {free_energy:.4f}</div>
      <div class="bar-bg"><div class="bar-fill" style="width:{surprise_pct}%"></div></div>
    </div>
    <div class="metric">
      <div class="metric-label">Evolution Count</div>
      <div class="metric-value evo">{evolution}</div>
      <div class="metric-sub">patches applied: {patches}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Correction Rate</div>
      <div class="metric-value soul">{correction_rate:.1f}%</div>
      <div class="metric-sub">last 100 turns</div>
    </div>
    <div class="metric">
      <div class="metric-label">Sleep Cycle</div>
      <div class="metric-value fep">{sleep_count}</div>
      <div class="metric-sub">last: {last_sleep}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Memory Nodes</div>
      <div class="metric-value evo">{memory_nodes}</div>
      <div class="metric-sub">chromadb vectors</div>
    </div>
    <div class="metric">
      <div class="metric-label">Soul Version</div>
      <div class="metric-value soul">{soul_version}</div>
      <div class="metric-sub">patterns: {patterns}</div>
    </div>
  </div>
  <div class="timestamp">rendered by PaperCanvasAgent @ {timestamp}</div>
</body>
</html>"""

    COMPONENT_PROMPT = """You are a UI engineer building production-ready HTML/CSS components.
Style guide: black background (#0a0a0a), neon green (#00FF41), cyan (#00E5FF), purple (#7B2FFF),
monospace font, pixel/matrix aesthetic. No frameworks — vanilla HTML/CSS only.

Component spec:
{spec}

Output ONLY the complete HTML, no explanation, no markdown fences.
"""

    # ── Core: run() ──────────────────────────────────────────────────────────

    async def run(self, action: str = "push_dashboard", **kwargs) -> dict:
        """
        Dispatch to sub-action.

        Actions:
          push_dashboard(data={})  — push metrics widget to Paper canvas
          push_html(html=str)      — write raw HTML to canvas
          pull_canvas()            — read canvas HTML back
          render_component(spec=str) — LLM → component → canvas
          push_readme(readme=str)  — push README markdown to canvas
        """
        if not MCP_AVAILABLE:
            return {"error": "mcp package not installed (pip install mcp>=1.0.0)", "action": action}

        dispatch = {
            "push_dashboard":   self._push_dashboard,
            "push_html":        self._push_html,
            "pull_canvas":      self._pull_canvas,
            "render_component": self._render_component,
            "push_readme":      self._push_readme,
        }
        handler = dispatch.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}. Available: {list(dispatch.keys())}"}

        try:
            return await handler(**kwargs)
        except ConnectionRefusedError:
            return {
                "error": "Paper Desktop not running or MCP not enabled",
                "tip": f"Open Paper Desktop → Enable MCP server (defaults to {PAPER_MCP_URL})",
                "action": action,
            }
        except Exception as e:
            logger.error(f"[PaperCanvas] {action} failed: {e}")
            return {"error": str(e), "action": action}

    # ── Actions ──────────────────────────────────────────────────────────────

    async def _push_html(self, html: str, node_name: str = "GodLocal Widget") -> dict:
        """Write HTML string to a Paper canvas node."""
        result = await _call_paper("create_html_node", {
            "name":    node_name,
            "content": html,
        })
        logger.info(f"[PaperCanvas] push_html → node: {node_name}")
        return {"status": "ok", "node": node_name, "mcp_result": result}

    async def _push_dashboard(self, data: dict | None = None) -> dict:
        """Render FEP metrics as live HTML widget on Paper canvas."""
        from datetime import datetime, timezone
        d = data or {}
        html = self.DASHBOARD_TEMPLATE.format(
            surprise=d.get("surprise", 0.0),
            free_energy=d.get("free_energy", 0.0),
            surprise_pct=min(100, d.get("surprise", 0.0) * 1000),
            evolution=d.get("evolution", 0),
            patches=d.get("patches", 0),
            correction_rate=d.get("correction_rate", 0.0),
            sleep_count=d.get("sleep_count", 0),
            last_sleep=d.get("last_sleep", "—"),
            memory_nodes=d.get("memory_nodes", 0),
            soul_version=d.get("soul_version", "v1"),
            patterns=d.get("patterns", 0),
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )
        return await self._push_html(html, node_name="GodLocal Dashboard")

    async def _pull_canvas(self, node_name: str | None = None) -> dict:
        """Read current HTML from Paper canvas (optionally filter by node name)."""
        args = {}
        if node_name:
            args["name"] = node_name
        result = await _call_paper("get_html_nodes", args)
        logger.info(f"[PaperCanvas] pull_canvas → {len(str(result))} chars")
        return {"status": "ok", "content": result}

    async def _render_component(self, spec: str, node_name: str = "AI Component") -> dict:
        """Generate UI component via LLM and push to canvas."""
        if not self.llm:
            return {"error": "LLM not available — pass llm= to PaperCanvasAgent"}
        prompt = self.COMPONENT_PROMPT.format(spec=spec)
        html = self.llm.generate(prompt, max_tokens=1500)
        # Strip accidental markdown fences
        if "```html" in html:
            html = html.split("```html")[1].split("```")[0].strip()
        elif "```" in html:
            html = html.split("```")[1].split("```")[0].strip()
        result = await self._push_html(html, node_name=node_name)
        return {**result, "spec": spec, "html_len": len(html)}

    async def _push_readme(self, readme: str, repo: str = "GodLocal") -> dict:
        """Convert README markdown to styled HTML and push to canvas."""
        # Simple MD→HTML (no deps)
        html = _md_to_html(readme, title=repo)
        return await self._push_html(html, node_name=f"{repo} README")


# ── Minimal MD→HTML ───────────────────────────────────────────────────────────

def _md_to_html(md: str, title: str = "README") -> str:
    """Bare-bones markdown → HTML with GodLocal style."""
    import re
    html_lines = []
    for line in md.splitlines():
        line = line.rstrip()
        if line.startswith("# "):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith("## "):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith("### "):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith("- ") or line.startswith("* "):
            html_lines.append(f'<li>{line[2:]}</li>')
        elif line.startswith("```"):
            tag = "<pre><code>" if not any(l.startswith("</pre>") for l in html_lines[-3:]) else "</code></pre>"
            html_lines.append(tag)
        elif line.startswith("> "):
            html_lines.append(f'<blockquote>{line[2:]}</blockquote>')
        elif line == "":
            html_lines.append("<br>")
        else:
            # inline code
            line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
            # bold
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            html_lines.append(f'<p>{line}</p>')

    body = "\n".join(html_lines)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ background:#0a0a0a; color:#c0ffc0; font-family:'JetBrains Mono',monospace;
          padding:32px; max-width:860px; line-height:1.7; }}
  h1 {{ color:#00FF41; border-bottom:1px solid #1a3a1a; padding-bottom:8px; }}
  h2 {{ color:#00E5FF; margin-top:24px; }}
  h3 {{ color:#7B2FFF; }}
  code {{ background:#0d160d; color:#00FF41; padding:2px 6px; border-radius:3px; }}
  pre  {{ background:#0d160d; padding:16px; border-radius:4px; overflow-x:auto; }}
  pre code {{ background:none; padding:0; }}
  blockquote {{ border-left:3px solid #00E5FF; padding-left:16px; color:#4a8a8a; }}
  li {{ margin-left:20px; }}
  strong {{ color:#00E5FF; }}
</style>
</head>
<body>{body}</body>
</html>"""

