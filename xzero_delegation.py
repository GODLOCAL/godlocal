"""
xzero_delegation.py — Intelligent AI Delegation layer for X-ZERO
═══════════════════════════════════════════════════════════════════
Implements Google DeepMind's "Intelligent AI Delegation" principles
for the Eliza Cloud (delegator) → Picobot (delegatee) pipeline.

Principles applied:
  1. Dynamic Assessment   — evaluate capability/risk/reversibility before any trade
  2. Adaptive Execution   — retry → fallback → Telegram escalation on failure
  3. Structural Transparency — every action logged with verifiable proof (tx hash)
  4. Trust Calibration    — [LOCKED] position limits LLM cannot override
  5. Systemic Resilience  — health checks, state backup, circuit breaker

Drop this file next to godlocal_v5.py. Import in X-ZERO orchestrator.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# [LOCKED] — Trust Calibration boundaries. LLM cannot override these values.
# To change limits: edit manually and restart. Remove [LOCKED] comment to allow
# AI updates (not recommended for financial parameters).
# ──────────────────────────────────────────────────────────────────────────────
LOCKED_LIMITS = {
    "max_position_sol":      1.0,     # [LOCKED] Max SOL per single trade
    "max_daily_trades":      20,      # [LOCKED] Circuit breaker — trades per day
    "max_daily_loss_sol":    0.5,     # [LOCKED] Daily loss limit before halt
    "min_balance_sol":       0.05,    # [LOCKED] Reserve — never trade below this
    "max_slippage_bps":      300,     # [LOCKED] 3% max slippage on Jupiter
    "reversibility_window_s": 30,     # [LOCKED] Window to abort before on-chain confirm
    "require_human_above_sol": 0.5,   # [LOCKED] Trades ≥ this SOL need Telegram approval
}

DATA_DIR = Path("godlocal_data/xzero")
STATE_FILE = DATA_DIR / "delegation_state.json"
AUDIT_LOG  = DATA_DIR / "delegation_audit.jsonl"
CIRCUIT_FILE = DATA_DIR / "circuit_breaker.json"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "log_archives").mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeIntent:
    """What Eliza Cloud wants Picobot to do."""
    action:      str            # "swap" | "limit" | "dca" | "stop_loss" | "lp"
    token_in:    str            # e.g. "SOL"
    token_out:   str            # e.g. "USDC"
    amount_sol:  float          # SOL value of the trade
    reason:      str            # LLM-generated rationale (logged for auditability)
    slippage_bps: int = 100     # default 1%
    urgency:     str = "normal" # "normal" | "urgent" | "cancel_only"
    session_id:  str = ""


@dataclass
class DelegationResult:
    """What actually happened — structural transparency artifact."""
    intent:          TradeIntent
    status:          str         # "executed" | "rejected" | "escalated" | "rolled_back"
    tx_hash:         Optional[str] = None
    rejection_reason: Optional[str] = None
    escalated_to:    Optional[str] = None   # "telegram" | "human"
    retries:         int = 0
    duration_ms:     int = 0
    assessment:      dict = field(default_factory=dict)
    timestamp:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dynamic Assessment
# ──────────────────────────────────────────────────────────────────────────────

class DynamicAssessor:
    """
    Principle 1: Before delegating, evaluate capability + risk + reversibility.
    "Not just who has the tool? But: who should be trusted with this specific
    task under these constraints?" — DeepMind paper
    """

    def assess(self, intent: TradeIntent, picobot_health: dict, portfolio: dict) -> dict:
        issues = []
        risk_score = 0.0  # 0.0 (safe) → 1.0 (halt)

        balance = portfolio.get("balance_sol", 0.0)
        daily_loss = portfolio.get("daily_loss_sol", 0.0)
        daily_trades = portfolio.get("daily_trades", 0)

        # ── Capability check ────────────────────────────────────────────────
        if not picobot_health.get("alive"):
            issues.append("Picobot unreachable")
            risk_score = 1.0

        if not picobot_health.get("solana_rpc_ok"):
            issues.append("Solana RPC unavailable")
            risk_score = max(risk_score, 0.9)

        # ── Resource check ──────────────────────────────────────────────────
        if balance < LOCKED_LIMITS["min_balance_sol"] + intent.amount_sol:
            issues.append(f"Insufficient balance: {balance:.4f} SOL (need {intent.amount_sol + LOCKED_LIMITS['min_balance_sol']:.4f})")
            risk_score = max(risk_score, 1.0)

        if intent.amount_sol > LOCKED_LIMITS["max_position_sol"]:
            issues.append(f"Position {intent.amount_sol} SOL exceeds LOCKED max {LOCKED_LIMITS['max_position_sol']} SOL")
            risk_score = max(risk_score, 1.0)

        # ── Circuit breaker ─────────────────────────────────────────────────
        if daily_trades >= LOCKED_LIMITS["max_daily_trades"]:
            issues.append(f"Daily trade limit reached ({daily_trades}/{LOCKED_LIMITS['max_daily_trades']})")
            risk_score = max(risk_score, 1.0)

        if daily_loss >= LOCKED_LIMITS["max_daily_loss_sol"]:
            issues.append(f"Daily loss limit hit ({daily_loss:.4f}/{LOCKED_LIMITS['max_daily_loss_sol']:.4f} SOL)")
            risk_score = max(risk_score, 1.0)

        # ── Reversibility check ─────────────────────────────────────────────
        reversible = intent.action in ("dca", "limit")  # on-chain swaps are irreversible
        if not reversible and intent.amount_sol >= LOCKED_LIMITS["require_human_above_sol"]:
            issues.append(f"Irreversible trade ≥{LOCKED_LIMITS['require_human_above_sol']} SOL — requires human approval")
            risk_score = max(risk_score, 0.8)

        # ── Slippage check ──────────────────────────────────────────────────
        if intent.slippage_bps > LOCKED_LIMITS["max_slippage_bps"]:
            issues.append(f"Slippage {intent.slippage_bps}bps exceeds LOCKED max {LOCKED_LIMITS['max_slippage_bps']}bps")
            intent.slippage_bps = LOCKED_LIMITS["max_slippage_bps"]

        verdict = "go" if risk_score < 0.5 else ("escalate" if risk_score < 1.0 else "halt")

        return {
            "verdict":     verdict,
            "risk_score":  round(risk_score, 2),
            "issues":      issues,
            "reversible":  reversible,
            "balance_sol": balance,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Adaptive Execution  +  3. Structural Transparency
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveExecutor:
    """
    Principle 2: If delegatee underperforms — reassign, escalate, restructure.
    Principle 3: Every execution produces a verifiable proof artifact.
    """

    MAX_RETRIES = 2
    RETRY_DELAY_S = 3

    def __init__(
        self,
        picobot_execute: Callable[[TradeIntent], dict],
        telegram_notify: Callable[[str], None],
    ):
        self._execute  = picobot_execute
        self._telegram = telegram_notify

    def run(self, intent: TradeIntent, assessment: dict) -> DelegationResult:
        t0 = time.monotonic()
        result = DelegationResult(intent=intent, assessment=assessment, status="pending")

        # ── Halt path ───────────────────────────────────────────────────────
        if assessment["verdict"] == "halt":
            result.status = "rejected"
            result.rejection_reason = "; ".join(assessment["issues"])
            logger.warning(f"[HALT] {result.rejection_reason}")
            self._telegram(f"🛑 XZERO HALT: {result.rejection_reason}")
            _write_audit(result)
            return result

        # ── Human-escalation path ───────────────────────────────────────────
        if assessment["verdict"] == "escalate":
            result.status = "escalated"
            result.escalated_to = "telegram"
            msg = (
                f"⚠️ X-ZERO needs approval\n"
                f"Action: {intent.action} {intent.amount_sol} SOL {intent.token_in}→{intent.token_out}\n"
                f"Reason: {intent.reason}\n"
                f"Issues: {'; '.join(assessment['issues'])}\n"
                f"Reply APPROVE or REJECT"
            )
            self._telegram(msg)
            logger.info(f"[ESCALATE] Sent to Telegram for human decision")
            _write_audit(result)
            return result

        # ── Adaptive execution with retry ───────────────────────────────────
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                exec_result = self._execute(intent)
                if exec_result.get("status") == "ok":
                    result.status   = "executed"
                    result.tx_hash  = exec_result.get("tx_hash")
                    result.retries  = attempt
                    logger.info(f"[EXECUTE] ✅ tx={result.tx_hash} attempt={attempt}")
                    break
                else:
                    last_error = exec_result.get("error", "unknown")
                    logger.warning(f"[EXECUTE] attempt {attempt} failed: {last_error}")

                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.RETRY_DELAY_S)
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[EXECUTE] attempt {attempt} exception: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_S)
        else:
            # All retries exhausted — escalate to human
            result.status = "escalated"
            result.escalated_to = "telegram"
            result.rejection_reason = f"All {self.MAX_RETRIES+1} attempts failed: {last_error}"
            self._telegram(
                f"⚠️ X-ZERO execution failed after {self.MAX_RETRIES+1} attempts\n"
                f"{intent.action} {intent.amount_sol} SOL: {last_error}\nRequires manual check."
            )
            logger.error(f"[ESCALATE] All retries failed, escalated to Telegram")

        result.duration_ms = int((time.monotonic() - t0) * 1000)
        _write_audit(result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. Systemic Resilience — health checks, circuit breaker, state backup
# ──────────────────────────────────────────────────────────────────────────────

class ResilienceGuard:
    """
    Principle 5: Efficiency without redundancy = fragility.
    Monitors Picobot health, maintains daily state, rotates audit log.
    """

    def check_picobot(self, picobot_status_fn: Callable[[], dict]) -> dict:
        """Ping Picobot health endpoint. Returns health dict."""
        try:
            s = picobot_status_fn()
            return {
                "alive":          True,
                "solana_rpc_ok":  s.get("rpc_ok", True),
                "memory_mb":      s.get("memory_mb", 0),
                "uptime_s":       s.get("uptime_s", 0),
                "last_checked":   datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"alive": False, "error": str(e)}

    def load_portfolio(self) -> dict:
        """Load current state from persistent file (survives restarts)."""
        _ensure_dirs()
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "balance_sol": 0.0,
            "daily_loss_sol": 0.0,
            "daily_trades": 0,
            "last_reset_date": datetime.utcnow().date().isoformat(),
        }

    def persist_portfolio(self, portfolio: dict):
        _ensure_dirs()
        # Daily reset
        today = datetime.utcnow().date().isoformat()
        if portfolio.get("last_reset_date") != today:
            portfolio["daily_loss_sol"]   = 0.0
            portfolio["daily_trades"]     = 0
            portfolio["last_reset_date"]  = today
        STATE_FILE.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")

    def rotate_audit_if_needed(self, max_mb: float = 5.0):
        """Mirror of GodLocal's _rotate_log_if_needed — same resilience pattern."""
        if not AUDIT_LOG.exists():
            return
        size_mb = AUDIT_LOG.stat().st_size / (1024 * 1024)
        if size_mb < max_mb:
            return
        import shutil
        archive_dir = DATA_DIR / "log_archives"
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(AUDIT_LOG, archive_dir / f"delegation_audit_{ts}.jsonl")
        AUDIT_LOG.write_text("", encoding="utf-8")
        archives = sorted(archive_dir.glob("delegation_audit_*.jsonl"))
        for old in archives[:-3]:
            old.unlink()
        logger.info(f"[Resilience] Audit log rotated ({size_mb:.1f}MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Top-level Delegation Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class XZeroDelegator:
    """
    Eliza Cloud → Picobot delegation pipeline.
    Wraps all 5 DeepMind principles into a single .delegate() call.

    Usage:
        delegator = XZeroDelegator(
            picobot_execute=picobot.execute_trade,
            picobot_status=picobot.get_status,
            telegram_notify=tg.send_message,
        )
        result = delegator.delegate(TradeIntent(
            action="swap", token_in="SOL", token_out="USDC",
            amount_sol=0.1, reason="DCA signal: RSI oversold"
        ))
    """

    def __init__(
        self,
        picobot_execute:  Callable[[TradeIntent], dict],
        picobot_status:   Callable[[], dict],
        telegram_notify:  Callable[[str], None],
    ):
        self.assessor  = DynamicAssessor()
        self.executor  = AdaptiveExecutor(picobot_execute, telegram_notify)
        self.resilience = ResilienceGuard()
        self._status_fn = picobot_status
        self._notify    = telegram_notify

    def delegate(self, intent: TradeIntent) -> DelegationResult:
        _ensure_dirs()
        self.resilience.rotate_audit_if_needed()

        # Load current portfolio state
        portfolio = self.resilience.load_portfolio()

        # Dynamic Assessment
        health = self.resilience.check_picobot(self._status_fn)
        assessment = self.assessor.assess(intent, health, portfolio)
        logger.info(
            f"[Assess] {intent.action} {intent.amount_sol}SOL → "
            f"verdict={assessment['verdict']} risk={assessment['risk_score']}"
        )

        # Adaptive Execution (includes transparency logging)
        result = self.executor.run(intent, assessment)

        # Update portfolio state after execution
        if result.status == "executed":
            portfolio["daily_trades"] = portfolio.get("daily_trades", 0) + 1
            # Caller should update balance_sol + daily_loss_sol from actual tx result
        self.resilience.persist_portfolio(portfolio)

        return result

    def status(self) -> dict:
        """Health snapshot — mirrors GodLocal's /status endpoint pattern."""
        health = self.resilience.check_picobot(self._status_fn)
        portfolio = self.resilience.load_portfolio()
        return {
            "picobot": health,
            "portfolio": portfolio,
            "locked_limits": LOCKED_LIMITS,
            "audit_log_kb": round(AUDIT_LOG.stat().st_size / 1024, 1) if AUDIT_LOG.exists() else 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_audit(result: DelegationResult):
    """Principle 3: Structural Transparency — verifiable, append-only audit trail."""
    _ensure_dirs()
    try:
        with AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error(f"[Audit] write failed: {e}")


def get_audit_tail(n: int = 10) -> list[dict]:
    """Return last n audit records for inspection."""
    if not AUDIT_LOG.exists():
        return []
    lines = [l for l in AUDIT_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [json.loads(l) for l in lines[-n:]]
