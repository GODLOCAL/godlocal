# GodLocal v5 — Sovereign AI Studio
# Copyright (C) 2026 Rostyslav Oliinyk
# AGPL-3.0 + Commercial Dual License
# See LICENSE and COMMERCIAL_LICENSE.md
#
# "Your AI. On your machine. Getting smarter while you sleep."
# https://github.com/GODLOCAL/godlocal

"""
GodLocal v5 capabilities:
  CORE    — AirLLM/Ollama, ChromaDB + SQLite memory, FastAPI REST
  MEMORY  — sleep_cycle() hippocampal replay (nightly consolidation)
  SOUL    — Personality-as-code via soul files (.md)
  SAFETY  — SafeExecutor (whitelisted shell)
  MEDICAL — MRIAnalyzer DICOM/NIfTI, HIPAA-by-architecture
  NEW v5  — ImageGen (Stable Diffusion/SDXL/Flux)
          — VideoGen (CogVideoX-2b, AnimateDiff)
          — AppGen  (DeepSeek-Coder / Qwen-Coder via Ollama)
          — AudioGen (Bark TTS + MusicGen)
"""

import os, json, time, asyncio, logging, hashlib, sqlite3, subprocess, threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    import ollama as _ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from airllm import AutoModel as AirLLMModel
    AIRLLM_AVAILABLE = True
except ImportError:
    AIRLLM_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import torch
    from diffusers import DiffusionPipeline, CogVideoXPipeline
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = IMAGE_GEN_AVAILABLE = VIDEO_GEN_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = IMAGE_GEN_AVAILABLE = VIDEO_GEN_AVAILABLE = False

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    import soundfile as sf
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False

try:
    from audiocraft.models import MusicGen
    import torchaudio
    MUSICGEN_AVAILABLE = True
except ImportError:
    MUSICGEN_AVAILABLE = False

try:
    import pydicom, nibabel as nib, numpy as np
    from transformers import pipeline as hf_pipeline
    MEDICAL_AVAILABLE = True
except ImportError:
    MEDICAL_AVAILABLE = False

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GodLocal] %(levelname)s: %(message)s")
logger = logging.getLogger("godlocal")


class GodLocalConfig:
    BASE_DIR     = Path("./godlocal_data")
    SOUL_DIR     = BASE_DIR / "souls"
    MEM_DIR      = BASE_DIR / "memory"
    LOG_DIR      = BASE_DIR / "logs"
    OUTPUTS_DIR  = BASE_DIR / "outputs"
    IMAGE_DIR    = OUTPUTS_DIR / "images"
    VIDEO_DIR    = OUTPUTS_DIR / "videos"
    APP_DIR      = OUTPUTS_DIR / "apps"
    AUDIO_DIR    = OUTPUTS_DIR / "audio"
    MEDICAL_DIR  = OUTPUTS_DIR / "medical"
    OLLAMA_MODEL = os.getenv("GODLOCAL_MODEL", "qwen2.5:7b")
    CODER_MODEL  = os.getenv("GODLOCAL_CODER_MODEL", "deepseek-coder:6.7b")
    IMAGE_MODEL  = os.getenv("GODLOCAL_IMAGE_MODEL", "stabilityai/sdxl-turbo")
    VIDEO_MODEL  = os.getenv("GODLOCAL_VIDEO_MODEL", "THUDM/CogVideoX-2b")
    MUSICGEN_MODEL = os.getenv("GODLOCAL_MUSIC_MODEL", "facebook/musicgen-small")
    SLEEP_CYCLE_HOUR = int(os.getenv("SLEEP_CYCLE_HOUR", "1"))
    SHORT_TERM_LIMIT = int(os.getenv("SHORT_TERM_LIMIT", "50"))
    LONG_TERM_THRESHOLD = float(os.getenv("LONG_TERM_THRESHOLD", "0.75"))
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    SHELL_WHITELIST = os.getenv(
        "SHELL_WHITELIST",
        "ls,cat,echo,pwd,python3,node,npm,pip,git,docker,curl,wget,mkdir,cp,mv,rm,find,grep,sed,awk,sort,head,tail,wc,date,env"
    ).split(",")

    @staticmethod
    def _device():
        if not DIFFUSERS_AVAILABLE: return "cpu"
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    @classmethod
    def init_dirs(cls):
        for d in [cls.BASE_DIR, cls.SOUL_DIR, cls.MEM_DIR, cls.LOG_DIR,
                  cls.OUTPUTS_DIR, cls.IMAGE_DIR, cls.VIDEO_DIR,
                  cls.APP_DIR, cls.AUDIO_DIR, cls.MEDICAL_DIR]:
            d.mkdir(parents=True, exist_ok=True)


CFG = GodLocalConfig()
DEFAULT_SOUL = """# default.md
You are GodLocal — a sovereign AI running entirely on local hardware.
You are private, fast, and getting smarter every day.
You remember everything the user tells you.
You never send data anywhere.
"""


class SoulEngine:
    def __init__(self):
        self.current_soul = "default"
        p = CFG.SOUL_DIR / "default.md"
        if not p.exists(): p.write_text(DEFAULT_SOUL)

    def load(self, name):
        p = CFG.SOUL_DIR / f"{name}.md"
        if not p.exists(): raise FileNotFoundError(f"Soul '{name}' not found")
        self.current_soul = name
        return p.read_text()

    def get_system_prompt(self):
        p = CFG.SOUL_DIR / f"{self.current_soul}.md"
        return p.read_text() if p.exists() else DEFAULT_SOUL

    def list_souls(self):
        return [p.stem for p in CFG.SOUL_DIR.glob("*.md")]

    def create_soul(self, name, content):
        p = CFG.SOUL_DIR / f"{name}.md"
        p.write_text(content)
        return p


class MemoryEngine:
    def __init__(self):
        self.db_path = str(CFG.MEM_DIR / "episodic.db")
        self._init_sqlite()
        if CHROMA_AVAILABLE:
            self._chroma = chromadb.PersistentClient(path=str(CFG.MEM_DIR / "chroma"))
            self.short_term = self._chroma.get_or_create_collection("short_term")
            self.long_term  = self._chroma.get_or_create_collection("long_term")
        else:
            self._chroma = None

    def _init_sqlite(self):
        c = sqlite3.connect(self.db_path)
        c.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT, timestamp TEXT,
                importance REAL DEFAULT 0.5, memory_type TEXT DEFAULT 'episodic',
                tags TEXT, promoted INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sleep_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, insights TEXT, patterns TEXT,
                pruned_count INTEGER, promoted_count INTEGER
            );
        """)
        c.commit(); c.close()

    def store(self, content, importance=0.5, tags=None, memory_type="episodic"):
        mid = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
        ts  = datetime.utcnow().isoformat()
        c   = sqlite3.connect(self.db_path)
        c.execute("INSERT OR IGNORE INTO memories VALUES (?,?,?,?,?,?,?)",
                  (mid, content, ts, importance, memory_type, json.dumps(tags or []), 0))
        c.commit(); c.close()
        if self._chroma:
            try:
                self.short_term.upsert(documents=[content], ids=[mid],
                    metadatas=[{"timestamp": ts, "importance": importance}])
            except Exception: pass
        return mid

    def recall(self, query, n=5):
        if not self._chroma: return []
        results = []
        for col, src in [(self.long_term, "long_term"), (self.short_term, "short_term")]:
            try:
                r = col.query(query_texts=[query], n_results=min(n, 10))
                results += [{"content": d, "source": src} for d in (r["documents"][0] or [])]
            except Exception: pass
        return results[:n]


class SafeExecutor:
    def run(self, command, cwd=None):
        cmd_name = command.strip().split()[0].split("/")[-1]
        if cmd_name not in CFG.SHELL_WHITELIST:
            return {"status": "blocked", "error": f"'{cmd_name}' not in whitelist"}
        try:
            r = subprocess.run(command, shell=True, capture_output=True,
                               text=True, timeout=30, cwd=cwd or str(CFG.BASE_DIR))
            return {"status": "ok", "stdout": r.stdout[:8096], "returncode": r.returncode}
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class SleepCycle:
    def __init__(self, memory, llm_fn):
        self.memory = memory
        self.llm    = llm_fn
        self._evolve = SelfEvolveEngine() if SELF_EVOLVE_AVAILABLE else None

    def run(self):
        logger.info("🌙 sleep_cycle() starting…")
        t0 = time.time()
        c  = sqlite3.connect(self.memory.db_path)
        rows = c.execute(
            "SELECT id, content, importance FROM memories WHERE promoted=0 ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
        if not rows:
            c.close()
            return {"status": "empty", "duration_s": round(time.time()-t0, 2)}
        promoted_ids = [r[0] for r in rows if r[2] >= CFG.LONG_TERM_THRESHOLD]
        prune_ids    = [r[0] for r in rows if r[2] < 0.2]
        for mid in promoted_ids:
            c.execute("UPDATE memories SET promoted=1 WHERE id=?", (mid,))
        if prune_ids:
            c.execute(f"DELETE FROM memories WHERE id IN ({','.join('?'*len(prune_ids))})", prune_ids)
        recent  = "\n".join(r[1][:200] for r in rows[:20])
        insight = self.llm(f"Extract 3-5 key insights from these AI agent memories:\n{recent}", max_tokens=300)
        report  = {"date": datetime.utcnow().date().isoformat(), "insights": insight,
                   "promoted_count": len(promoted_ids), "pruned_count": len(prune_ids),
                   "duration_s": round(time.time()-t0, 2)}
        c.execute("INSERT INTO sleep_reports (date,insights,patterns,pruned_count,promoted_count) VALUES (?,?,?,?,?)",
                  (report["date"], report["insights"], f"Processed {len(rows)} memories",
                   report["pruned_count"], report["promoted_count"]))
        c.commit(); c.close()
        logger.info(f"🌙 Done: +{len(promoted_ids)} promoted, -{len(prune_ids)} pruned")

        # Phase 2 — Self-Evolution: scan gaps created since last consolidation
        if self._evolve:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                evolve_result = loop.run_until_complete(
                    self._evolve.run_evolution_cycle(
                        llm_generate=self.llm,
                        max_gaps=5,
                        hours_back=24,
                    )
                )
                loop.close()
                report["self_evolve"] = {
                    "gaps_found":    evolve_result.gaps_found,
                    "gaps_resolved": evolve_result.gaps_resolved,
                    "topics":        evolve_result.topics_resolved,
                }
                logger.info(f"🧬 Self-evolution: {evolve_result.gaps_resolved}/{evolve_result.gaps_found} gaps resolved")
            except Exception as e:
                logger.warning(f"[SelfEvolve] skipped: {e}")
                report["self_evolve"] = {"error": str(e)}

        return report


class LLMEngine:
    def __init__(self):
        self.soul = SoulEngine()
        self._airllm_mdl = None

    def complete(self, prompt, model=None, max_tokens=1000):
        try:
            if not OLLAMA_AVAILABLE: raise RuntimeError("Ollama not available")
            r = _ollama_lib.chat(model=model or CFG.OLLAMA_MODEL,
                                  messages=[{"role": "user", "content": prompt}],
                                  options={"num_predict": max_tokens})
            return r["message"]["content"]
        except Exception:
            try:
                if not AIRLLM_AVAILABLE: raise RuntimeError("AirLLM not available")
                if self._airllm_mdl is None:
                    self._airllm_mdl = AirLLMModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
                r = self._airllm_mdl.generate([prompt], max_new_tokens=512)
                return r[0] if r else ""
            except Exception as e:
                return f"[LLM unavailable: {e}]"

    def chat(self, message, history=None, memory_context=None):
        system = self.soul.get_system_prompt()
        if memory_context: system += f"\n\n## Relevant memories:\n{memory_context}"
        ctx = "".join(f"{h['role'].title()}: {h['content']}\n" for h in (history or [])[-5:])
        full = f"{ctx}User: {message}" if ctx else message
        try:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": full}]
            r = _ollama_lib.chat(model=CFG.OLLAMA_MODEL, messages=msgs,
                                  options={"num_predict": 1000})
            return r["message"]["content"]
        except Exception as e:
            return f"[LLM unavailable: {e}]"


class AppGenerator:
    TEMPLATES = {
        "webapp": "Create a complete single-file HTML web app with inline CSS and JS.",
        "python": "Create a complete Python script with all imports and main block.",
        "api":    "Create a complete FastAPI REST API with endpoints and models.",
    }
    def __init__(self, llm_fn):
        self.llm = llm_fn

    def generate(self, description, app_type="webapp"):
        template = self.TEMPLATES.get(app_type, self.TEMPLATES["webapp"])
        code = self.llm(f"{template}\nDescription: {description}\nOutput ONLY code:",
                        model=CFG.CODER_MODEL, max_tokens=4000)
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        ext = {"webapp": "html", "python": "py", "api": "py"}.get(app_type, "txt")
        fname = f"app_{int(time.time())}.{ext}"
        out = CFG.APP_DIR / fname
        out.write_text(code)
        return {"status": "ok", "code": code, "path": str(out), "filename": fname}


class MRIAnalyzer:
    def analyze(self, file_path):
        if not MEDICAL_AVAILABLE:
            raise RuntimeError("pip install pydicom nibabel transformers")
        p = Path(file_path)
        if not p.exists(): raise FileNotFoundError(f"File not found: {file_path}")
        result = {"file": p.name, "format": p.suffix.upper(), "phi_egress": False}
        if p.suffix.lower() == ".dcm":
            dcm = pydicom.dcmread(str(p))
            result["modality"]  = str(getattr(dcm, "Modality", "Unknown"))
            result["body_part"] = str(getattr(dcm, "BodyPartExamined", "Unknown"))
        elif p.suffix.lower() in (".nii", ".gz"):
            img = nib.load(str(p))
            result["shape"] = list(img.shape)
        out = CFG.MEDICAL_DIR / f"report_{p.stem}_{int(time.time())}.json"
        out.write_text(json.dumps(result, indent=2))
        result["report_path"] = str(out)
        return result


class GodLocalAgent:
    def __init__(self):
        CFG.init_dirs()
        self.llm         = LLMEngine()
        self.memory      = MemoryEngine()
        self.soul        = self.llm.soul
        self.executor    = SafeExecutor()
        self.sleep_cycle = SleepCycle(self.memory, self.llm.complete)
        self.app_gen     = AppGenerator(self.llm.complete)
        self.mri         = MRIAnalyzer()
        self.history: List[Dict] = []
        logger.info("🚀 GodLocal v5 — Sovereign AI Studio — ready")

    def chat(self, message):
        mems   = self.memory.recall(message, n=3)
        ctx    = "\n".join(m["content"] for m in mems) or None
        answer = self.llm.chat(message, self.history, ctx)
        self.memory.store(f"User: {message}", importance=0.5)
        self.memory.store(f"Agent: {answer}", importance=0.4)
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def generate_app(self, description, app_type="webapp"):
        return self.app_gen.generate(description, app_type)

    def run_command(self, cmd):
        return self.executor.run(cmd)

    def run_sleep_cycle(self):
        return self.sleep_cycle.run()

    def status(self):
        return {
            "version": "v5",
            "soul": self.soul.current_soul,
            "capabilities": {
                "llm_ollama": OLLAMA_AVAILABLE, "llm_airllm": AIRLLM_AVAILABLE,
                "memory_chroma": CHROMA_AVAILABLE, "image_generation": IMAGE_GEN_AVAILABLE,
                "video_generation": VIDEO_GEN_AVAILABLE, "tts_bark": BARK_AVAILABLE,
                "music_musicgen": MUSICGEN_AVAILABLE, "app_generation": True,
                "safe_executor": True, "sleep_cycle": True, "self_evolve": SELF_EVOLVE_AVAILABLE, "medical_mri": MEDICAL_AVAILABLE,
            },
            "device": CFG._device(),
            "session_messages": len(self.history),
        }



try:
    from self_evolve import SelfEvolveEngine
    SELF_EVOLVE_AVAILABLE = True
except ImportError:
    SELF_EVOLVE_AVAILABLE = False

app = FastAPI(title="GodLocal v5",
              description="Sovereign AI Studio — Chat · Images · Video · Apps · Audio · Medical",
              version="5.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

agent: Optional[GodLocalAgent] = None


class ChatReq(BaseModel):
    message: str
    history: Optional[List[Dict]] = None

class AppReq(BaseModel):
    description: str
    app_type: str = "webapp"

class CmdReq(BaseModel):
    command: str

class SoulReq(BaseModel):
    soul_name: str
    content: Optional[str] = None


@app.on_event("startup")
async def startup():
    global agent
    agent = GodLocalAgent()
    if CFG.OUTPUTS_DIR.exists():
        try:
            app.mount("/outputs", StaticFiles(directory=str(CFG.OUTPUTS_DIR)), name="outputs")
        except Exception: pass
    if SCHEDULER_AVAILABLE:
        sched = AsyncIOScheduler()
        sched.add_job(lambda: agent.run_sleep_cycle(), "cron",
                      hour=CFG.SLEEP_CYCLE_HOUR, minute=0, id="sleep_cycle")
        sched.start()
        logger.info(f"⏰ sleep_cycle() scheduled at {CFG.SLEEP_CYCLE_HOUR}:00 UTC")


@app.get("/")
async def root():
    return {"name": "GodLocal", "version": "v5",
            "tagline": "Your AI. On your machine. Getting smarter while you sleep.",
            "docs": "/docs"}

@app.get("/health")
async def health(): return {"status": "ok"}

@app.get("/status")
async def status_route(): return agent.status()

@app.post("/chat")
async def chat(req: ChatReq):
    return {"response": agent.chat(req.message), "soul": agent.soul.current_soul}

@app.post("/create/app")
async def create_app(req: AppReq):
    try: return agent.generate_app(req.description, req.app_type)
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/execute")
async def execute(req: CmdReq): return agent.run_command(req.command)

@app.post("/sleep")
async def trigger_sleep(): return agent.run_sleep_cycle()

@app.get("/souls")
async def list_souls(): return {"souls": agent.soul.list_souls(), "current": agent.soul.current_soul}

@app.post("/souls/load")
async def load_soul(req: SoulReq):
    try:
        agent.soul.load(req.soul_name)
        return {"status": "ok", "soul": req.soul_name}
    except FileNotFoundError:
        raise HTTPException(404, f"Soul '{req.soul_name}' not found")

@app.post("/souls/create")
async def create_soul(req: SoulReq):
    if not req.content: raise HTTPException(400, "Content required")
    p = agent.soul.create_soul(req.soul_name, req.content)
    return {"status": "created", "soul": req.soul_name}

@app.get("/memory/search")
async def search_memory(q: str, n: int = 5):
    return {"results": agent.memory.recall(q, n=n)}

@app.post("/memory/store")
async def store_memory(content: str, importance: float = 0.5):
    mid = agent.memory.store(content, importance)
    return {"status": "stored", "id": mid}


if __name__ == "__main__":
    print("""
╬══════════════════════════════════════════════════════════╖
║        GodLocal v5 — Sovereign AI Studio                 ║
║  Your AI. Your machine. Getting smarter while you sleep. ║
╟══════════════════════════════════════════════════════════╢
║  Chat        → POST /chat                                ║
║  Apps        → POST /create/app    (DeepSeek-Coder)      ║
║  Shell       → POST /execute       (SafeExecutor)        ║
║  Medical     → Load godlocal_v5_modules.py               ║
║  Sleep       → POST /sleep         (hippocampal)         ║
║  Docs        →  GET /docs                                ║
║  GitHub      → github.com/GODLOCAL/godlocal               ║
╙══════════════════════════════════════════════════════════╒
""")
    uvicorn.run("godlocal_v5:app", host=CFG.API_HOST, port=CFG.API_PORT, reload=False)
