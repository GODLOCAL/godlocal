"""godlocal_v6.py — БОГ || OASIS v6 · Sovereign AI Studio
FastAPI + lifespan (replaces on_event) · Brain singleton · AgentPool · AutoGenesis
"""
from __future__ import annotations

import argparse
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader

from core.brain import Brain
from core.settings import settings
from agents.agent_pool import agent_pool
from agents.autogenesis_v2 import AutoGenesis
from models.schemas import (
    AgentSwapResponse,
    EvolveRequest,
    EvolveResponse,
    MemoryAddRequest,
    StatusResponse,
    ThinkRequest,
    ThinkResponse,
)
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_start_time = time.time()
_autogenesis = AutoGenesis(root=".")

# ── Auth ──────────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(_api_key_header)) -> None:
    if not settings.api_key:
        return  # auth disabled
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("БОГ || OASIS v6 starting…")
    brain = Brain.get()  # init singleton (loads soul + memory + LLM)

    # Start nightly scheduler
    from sleep_scheduler_v6 import start_scheduler
    scheduler_task = start_scheduler()

    # AutoGenesis Shortcuts server (optional)
    if settings.model:  # always True — just gate on env flag if needed
        pass

    logger.info("✅ Ready — model=%s soul=%s", settings.model, settings.soul_file)

    yield  # ← app runs here

    # Shutdown
    scheduler_task.cancel()
    logger.info("БОГ || OASIS v6 shutdown")


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="БОГ || OASIS v6",
    description="Sovereign Local AI Studio — your AI, your machine, getting smarter while you sleep.",
    version="6.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core endpoints ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    uptime = int(time.time() - _start_time)
    brain = Brain.get()
    fep = _autogenesis.fep_metrics()
    return f"""<!DOCTYPE html><html><head><title>БОГ || OASIS v6</title>
<meta charset="utf-8"><style>
body{{background:#0a0a0a;color:#00FF41;font-family:monospace;padding:2rem}}
h1{{color:#00E5FF}}span{{color:#7B2FFF}}
</style></head><body>
<h1>БОГ || OASIS <span>v6.0</span></h1>
<pre>model     {settings.model}
uptime    {uptime}s
soul      {settings.soul_file} ({'✅' if Path(settings.soul_file).exists() else '❌'})
memory    short={brain.memory.short.count()} long={brain.memory.long.count()}
fep       correction_rate={fep['correction_rate']} free_energy={fep['free_energy']}
agent     {agent_pool.status()['active'] or 'default'}
apply     {settings.autogenesis_apply}</pre>
</body></html>"""


@app.get("/status", response_model=StatusResponse)
async def get_status():
    brain = Brain.get()
    return StatusResponse(
        version="6.0.0",
        model=settings.model,
        soul_loaded=Path(settings.soul_file).exists(),
        memory={"short": brain.memory.short.count(), "long": brain.memory.long.count()},
        fep=_autogenesis.fep_metrics(),
        agents=agent_pool.status(),
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/think", response_model=ThinkResponse, dependencies=[Depends(verify_api_key)])
async def think(req: ThinkRequest):
    brain = Brain.get()
    response = await brain.think(req.task, max_tokens=req.max_tokens)
    return ThinkResponse(
        response=response,
        model=settings.model,
        tokens_approx=len(response.split()),
    )


@app.post("/evolve", response_model=EvolveResponse, dependencies=[Depends(verify_api_key)])
async def evolve(req: EvolveRequest):
    result = await _autogenesis.evolve_async(
        task=req.task,
        apply=req.apply,
        max_revisions=req.max_revisions,
    )
    return EvolveResponse(**result)


@app.post("/rollback/{filename}", dependencies=[Depends(verify_api_key)])
async def rollback(filename: str):
    backup = Path("godlocal_data/backups") / filename
    if not backup.exists():
        raise HTTPException(status_code=404, detail=f"No backup for {filename}")
    import shutil
    shutil.copy(backup, filename)
    return {"status": "rolled_back", "file": filename}


# ── Memory endpoints ──────────────────────────────────────────────────────
@app.post("/memory/add", dependencies=[Depends(verify_api_key)])
async def memory_add(req: MemoryAddRequest):
    brain = Brain.get()
    brain.memory.add(req.text, long=req.long)
    return {"status": "added", "collection": "long" if req.long else "short"}


@app.post("/memory/clear", dependencies=[Depends(verify_api_key)])
async def memory_clear():
    brain = Brain.get()
    pruned = brain.memory.prune()
    return {"pruned": pruned}


# ── Agent endpoints ───────────────────────────────────────────────────────
@app.get("/agent/status")
async def agent_status():
    return agent_pool.status()


@app.post("/agent/swap/{agent_type}", response_model=AgentSwapResponse, dependencies=[Depends(verify_api_key)])
async def swap_agent(agent_type: str):
    try:
        result = await agent_pool.swap(agent_type)
        return AgentSwapResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Mobile API (SwiftUI OasisApp) ─────────────────────────────────────────
@app.get("/mobile/status")
async def mobile_status():
    brain = Brain.get()
    return {
        "soul_loaded": Path(settings.soul_file).exists(),
        "model": settings.model,
        "memory_short": brain.memory.short.count(),
        "memory_long": brain.memory.long.count(),
        "fep": _autogenesis.fep_metrics(),
        "agent": agent_pool.status()["active"],
        "uptime_sec": round(time.time() - _start_time, 1),
    }


@app.post("/mobile/evolve", dependencies=[Depends(verify_api_key)])
async def mobile_evolve(task: str, apply: bool = False):
    result = await _autogenesis.evolve_async(task=task, apply=apply)
    return {
        "status": result["status"],
        "patches": sum(f.get("patches", 0) for f in result.get("files", [])),
        "fep": result.get("fep", {}),
    }


# ── Correction feedback (self-improve loop) ───────────────────────────────
@app.post("/feedback", dependencies=[Depends(verify_api_key)])
async def feedback(was_corrected: bool):
    _autogenesis.record_correction(was_corrected)
    return {"fep": _autogenesis.fep_metrics()}


# ── Entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="БОГ || OASIS v6")
    ap.add_argument("--host",      default=settings.api_host)
    ap.add_argument("--port",      type=int, default=settings.api_port)
    ap.add_argument("--reload",    action="store_true", help="uvicorn hot-reload (dev)")
    ap.add_argument("--dry-run",   action="store_true", help="Start without AutoGenesis writes")
    ap.add_argument("--verbose",   action="store_true", help="DEBUG logging")
    ap.add_argument("--model",     default=None, help="Override GODLOCAL_MODEL")
    args = ap.parse_args()

    if args.verbose:
        import logging as _l; _l.getLogger().setLevel(_l.DEBUG)
    if args.dry_run:
        settings.autogenesis_apply = False
        logger.warning("⚠️  DRY-RUN: AutoGenesis writes disabled")
    if args.model:
        settings.model = args.model

    print("""
╬══════════════════════════════════════════════════════════╖
║        БОГ || OASIS v6 — Sovereign AI Studio             ║
║  Your AI. Your machine. Getting smarter while you sleep. ║
╟──────────────────────────────────────────────────────────╢
║  Dashboard  http://localhost:{port}                       ║
║  Docs       http://localhost:{port}/docs                  ║
║  Status     http://localhost:{port}/status                ║
╚══════════════════════════════════════════════════════════╝
    """.format(port=args.port))

    uvicorn.run(
        "godlocal_v6:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )
