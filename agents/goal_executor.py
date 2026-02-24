"""
agents/goal_executor.py
SureThing-style execution algorithm for GodLocal v6
──────────────────────────────────────────────────
Flow A  — HITL Resume       (resume after human action)
Flow B  — Incoming Signal   (react to email / timer / event)
Flow C  — Standard          (goal → plan → execute → stop at human)

Core loop:
  Goal → decompose_plan() → TaskChain
       → run_chain()      → execute AI tasks sequentially
       → STOP at human    → persist state
       → resume(task_id)  → continue from HITL result
"""

from __future__ import annotations

import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("godlocal.goal_executor")


# ── Types ────────────────────────────────────────────────────────────────────

class Executor(str, Enum):
    AI    = "ai"
    HUMAN = "human"

class TaskStatus(str, Enum):
    PENDING    = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    SKIPPED    = "skipped"
    FAILED     = "failed"
    AWAITING   = "awaiting_user_action"

class Signal(str, Enum):
    EMAIL   = "email"
    TIMER   = "timer"
    CHAT    = "chat_message"
    HITL    = "hitl"
    EVENT   = "event"

@dataclass
class Task:
    title:      str
    executor:   Executor                = Executor.AI
    action:     str                     = ""
    why_human:  str                     = ""
    status:     TaskStatus              = TaskStatus.PENDING
    task_id:    str                     = field(default_factory=lambda: str(uuid.uuid4())[:8])
    result:     Optional[dict[str,Any]] = None
    created_at: str                     = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["executor"] = self.executor.value
        d["status"]   = self.status.value
        return d

@dataclass
class GoalChain:
    goal:       str
    chain_id:   str                  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tasks:      list[Task]           = field(default_factory=list)
    created_at: str                  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def next_pending(self) -> Optional[Task]:
        return next((t for t in self.tasks if t.status == TaskStatus.PENDING), None)

    @property
    def is_complete(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
                   for t in self.tasks)

    def get_task(self, task_id: str) -> Optional[Task]:
        return next((t for t in self.tasks if t.task_id == task_id), None)

    def to_dict(self) -> dict:
        return {
            "chain_id":   self.chain_id,
            "goal":       self.goal,
            "tasks":      [t.to_dict() for t in self.tasks],
            "created_at": self.created_at,
        }


# ── AIRunner protocol ────────────────────────────────────────────────────────
# Inject your Brain / LLMBridge here — see example at bottom of file.

AIRunner = Callable[[str, str], Coroutine[Any, Any, str]]


# ── GoalExecutor ─────────────────────────────────────────────────────────────

class GoalExecutor:
    """
    Drop-in execution layer.  Usage:

        executor = GoalExecutor(ai_runner=brain.think, state_dir=Path("data/goals"))

        # Flow C — new goal
        chain = await executor.run("Deploy WebAgent to production")

        # Flow A — HITL result came back
        await executor.resume(chain_id="abc12345", task_id="cd56ef78",
                               result={"action": "confirmed", "note": "LGTM"})

        # Flow B — incoming signal
        await executor.on_signal(Signal.EMAIL,
                                  payload={"from": "user@x.com", "body": "..."})
    """

    def __init__(
        self,
        ai_runner: AIRunner,
        state_dir: Path = Path("data/goals"),
        hitl_callback: Optional[Callable[[Task], Coroutine]] = None,
    ):
        self.ai       = ai_runner
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.hitl_cb  = hitl_callback   # called when a human task is reached
        self._chains: dict[str, GoalChain] = {}
        self._load_persisted()

    # ── Persistence ──────────────────────────────────────────────────────

    def _path(self, chain_id: str) -> Path:
        return self.state_dir / f"{chain_id}.json"

    def _save(self, chain: GoalChain) -> None:
        self._path(chain.chain_id).write_text(
            json.dumps(chain.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_persisted(self) -> None:
        for f in self.state_dir.glob("*.json"):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                tasks = [
                    Task(
                        title=t["title"], executor=Executor(t["executor"]),
                        action=t.get("action",""), why_human=t.get("why_human",""),
                        status=TaskStatus(t["status"]),
                        task_id=t["task_id"], result=t.get("result"),
                        created_at=t["created_at"],
                    )
                    for t in raw["tasks"]
                ]
                chain = GoalChain(goal=raw["goal"], chain_id=raw["chain_id"],
                                   tasks=tasks, created_at=raw["created_at"])
                self._chains[chain.chain_id] = chain
            except Exception as e:
                logger.warning("Failed to load chain %s: %s", f.stem, e)

    # ── Flow C: Standard ─────────────────────────────────────────────────

    async def run(self, goal: str, tasks: Optional[list[Task]] = None) -> GoalChain:
        """
        Main entry point.
        - If `tasks` provided: use them (manual chain).
        - Otherwise: call LLM to decompose goal into tasks.
        """
        if tasks is None:
            tasks = await self._decompose(goal)

        chain = GoalChain(goal=goal, tasks=tasks)
        self._chains[chain.chain_id] = chain
        self._save(chain)

        logger.info("[Flow C] Goal: %s | Tasks: %d", goal, len(tasks))
        await self._run_chain(chain)
        return chain

    async def _decompose(self, goal: str) -> list[Task]:
        """Ask LLM to break goal into AI + human task chain."""
        prompt = (
            f"Break this goal into a minimal ordered task chain.\n"
            f"Goal: {goal}\n\n"
            f"Rules:\n"
            f"- Each task: title, executor (ai|human), action (runbook), why_human (if human)\n"
            f"- Human tasks = tasks that need approval, review, or a real-world action\n"
            f"- AI tasks come BEFORE human checkpoints\n"
            f"- Be concise, max 5 tasks\n\n"
            f"Return JSON array: [{{'title':'...','executor':'ai|human',"
            f"'action':'...','why_human':'...'}}]"
        )
        raw = await self.ai(prompt, "goal_decompose")
        try:
            # Extract JSON array from LLM response
            start = raw.index("["); end = raw.rindex("]") + 1
            items = json.loads(raw[start:end])
            return [
                Task(
                    title=i.get("title","Task"),
                    executor=Executor(i.get("executor","ai")),
                    action=i.get("action",""),
                    why_human=i.get("why_human",""),
                )
                for i in items
            ]
        except Exception as e:
            logger.warning("Decompose parse failed (%s), using single task", e)
            return [Task(title=goal, executor=Executor.AI, action=goal)]

    async def _run_chain(self, chain: GoalChain) -> None:
        """Execute AI tasks sequentially; stop at first human task."""
        for task in chain.tasks:
            if task.status != TaskStatus.PENDING:
                continue

            if task.executor == Executor.HUMAN:
                # ── STOP — human checkpoint ───────────────────────────
                task.status = TaskStatus.AWAITING
                self._save(chain)
                logger.info("[HITL] STOP at: %s | why: %s", task.title, task.why_human)
                if self.hitl_cb:
                    await self.hitl_cb(task)
                return      # ← execution halts here

            # ── AI task ───────────────────────────────────────────────
            task.status = TaskStatus.IN_PROGRESS
            self._save(chain)
            logger.info("[AI] Running: %s", task.title)
            try:
                result_text = await self.ai(task.action, f"task:{task.task_id}")
                task.status = TaskStatus.COMPLETED
                task.result = {"output": result_text}
                logger.info("[AI] Done: %s", task.title)
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.result = {"error": str(e)}
                logger.error("[AI] Failed: %s | %s", task.title, e)
                self._save(chain)
                return      # stop chain on failure
            self._save(chain)

        if chain.is_complete:
            logger.info("[Flow C] Chain complete: %s", chain.chain_id)

    # ── Flow A: HITL Resume ──────────────────────────────────────────────

    async def resume(self, chain_id: str, task_id: str,
                      result: dict[str, Any]) -> Optional[GoalChain]:
        """
        Call this when a human task is completed.

        result dict examples:
          {"action": "confirmed"}
          {"action": "changes", "note": "Tone too formal"}
          {"action": "cancelled"}
        """
        chain = self._chains.get(chain_id)
        if not chain:
            logger.warning("[Flow A] Chain not found: %s", chain_id)
            return None

        task = chain.get_task(task_id)
        if not task:
            logger.warning("[Flow A] Task not found: %s in %s", task_id, chain_id)
            return None

        action = result.get("action", "confirmed")
        logger.info("[Flow A] Resume: %s | action: %s", task.title, action)

        if action == "confirmed":
            task.status = TaskStatus.COMPLETED
            task.result = result
            self._save(chain)
            await self._run_chain(chain)   # continue from next task

        elif action == "changes":
            # Re-queue as AI task to incorporate feedback, then re-stop at human
            note = result.get("note", "")
            revised_action = f"Revise previous output. Feedback: {note}\nOriginal: {task.action}"
            revised = Task(
                title=f"[Revised] {task.title}",
                executor=Executor.AI,
                action=revised_action,
            )
            review = Task(
                title=task.title,
                executor=Executor.HUMAN,
                action=task.action,
                why_human=task.why_human,
            )
            idx = chain.tasks.index(task)
            task.status = TaskStatus.SKIPPED
            chain.tasks[idx+1:idx+1] = [revised, review]
            self._save(chain)
            await self._run_chain(chain)

        elif action == "cancelled":
            task.status = TaskStatus.SKIPPED
            task.result = {"reason": "User cancelled"}
            self._save(chain)
            logger.info("[Flow A] Chain cancelled at: %s", task.title)

        return chain

    # ── Flow B: Incoming Signal ──────────────────────────────────────────

    async def on_signal(self, signal: Signal, payload: dict[str, Any],
                         chain_id: Optional[str] = None) -> Optional[str]:
        """
        React to an incoming signal (email, timer, event).
        Returns a response string or None.
        """
        logger.info("[Flow B] Signal: %s", signal.value)

        context = json.dumps(payload, ensure_ascii=False)[:1200]
        prompt = (
            f"You received a {signal.value} signal.\n"
            f"Payload: {context}\n\n"
            f"Determine: (1) Is this actionable? (2) What single action should be taken?\n"
            f"Reply in JSON: {{\"actionable\": true/false, \"action\": \"...\", \"reply\": \"...\"}}"
        )
        raw = await self.ai(prompt, f"signal:{signal.value}")

        try:
            start = raw.index("{"); end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
        except Exception:
            return raw

        if parsed.get("actionable") and parsed.get("action"):
            goal = parsed["action"]
            logger.info("[Flow B] Spawning goal from signal: %s", goal)
            chain = await self.run(goal)
            if chain_id:
                self._chains[chain.chain_id] = chain

        return parsed.get("reply")

    # ── State helpers ─────────────────────────────────────────────────────

    def list_chains(self, active_only: bool = False) -> list[dict]:
        chains = list(self._chains.values())
        if active_only:
            chains = [c for c in chains if not c.is_complete]
        return [c.to_dict() for c in chains]

    def get_chain(self, chain_id: str) -> Optional[dict]:
        c = self._chains.get(chain_id)
        return c.to_dict() if c else None

    def skip_task(self, chain_id: str, task_id: str, reason: str = "") -> None:
        """Mark a task as skipped (obsolete / superseded)."""
        chain = self._chains.get(chain_id)
        if chain:
            task = chain.get_task(task_id)
            if task:
                task.status = TaskStatus.SKIPPED
                task.result = {"reason": reason}
                self._save(chain)
                logger.info("[Hygiene] Skipped %s: %s", task.title, reason)


# ── FastAPI integration ───────────────────────────────────────────────────────

def register_routes(app, executor: "GoalExecutor") -> None:
    """
    Mount GoalExecutor routes onto an existing FastAPI app.
    Call this from godlocal_v6.py after creating the executor.

    Example:
        from agents.goal_executor import GoalExecutor, register_routes, Task, Executor
        executor = GoalExecutor(ai_runner=brain.think)
        register_routes(app, executor)
    """
    from fastapi import Body
    from fastapi.responses import JSONResponse

    @app.post("/goals/run")
    async def run_goal(goal: str = Body(..., embed=True),
                        tasks: list[dict] = Body(default=None, embed=True)):
        task_objs = None
        if tasks:
            task_objs = [
                Task(
                    title=t["title"],
                    executor=Executor(t.get("executor","ai")),
                    action=t.get("action",""),
                    why_human=t.get("why_human",""),
                )
                for t in tasks
            ]
        chain = await executor.run(goal, task_objs)
        return JSONResponse(chain.to_dict())

    @app.post("/goals/{chain_id}/resume/{task_id}")
    async def resume_goal(chain_id: str, task_id: str,
                           result: dict = Body(...)):
        chain = await executor.resume(chain_id, task_id, result)
        if not chain:
            return JSONResponse({"error": "chain or task not found"}, status_code=404)
        return JSONResponse(chain.to_dict())

    @app.get("/goals")
    async def list_goals(active_only: bool = False):
        return JSONResponse(executor.list_chains(active_only))

    @app.get("/goals/{chain_id}")
    async def get_goal(chain_id: str):
        c = executor.get_chain(chain_id)
        if not c:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(c)

    @app.post("/goals/{chain_id}/signal")
    async def signal_goal(chain_id: str,
                           signal: str = Body(..., embed=True),
                           payload: dict = Body(default={}, embed=True)):
        from agents.goal_executor import Signal
        try:
            sig = Signal(signal)
        except ValueError:
            return JSONResponse({"error": f"unknown signal: {signal}"}, status_code=400)
        reply = await executor.on_signal(sig, payload, chain_id=chain_id)
        return JSONResponse({"reply": reply})


# ── Example wiring (godlocal_v6.py) ─────────────────────────────────────────
# 
# from agents.goal_executor import GoalExecutor, register_routes
#
# async def hitl_notify(task):
#     """Called when execution hits a human checkpoint."""
#     logger.info("⏸  Human needed: %s — %s", task.title, task.why_human)
#     # e.g. push to Telegram, UI websocket, etc.
#
# executor = GoalExecutor(
#     ai_runner=brain.think,          # async (prompt, context_id) -> str
#     state_dir=Path("data/goals"),
#     hitl_callback=hitl_notify,
# )
# register_routes(app, executor)
#
# # Run a goal:
# chain = await executor.run("Write and send project update email")
# # → AI drafts email → STOPS for human review
# # → User confirms via POST /goals/{chain_id}/resume/{task_id} {"action":"confirmed"}
# # → AI sends email
