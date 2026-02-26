"""
xzero Circuit Breaker
=====================
Protects the trading wallet from runaway losses.

Policy (all configurable via env vars):
  XZERO_MAX_DRAWDOWN_PCT   — max portfolio drawdown before kill (default 10%)
  XZERO_MAX_CONSECUTIVE_L  — max consecutive losing trades before pause (default 3)
  XZERO_DAILY_LOSS_SOL     — max SOL lost in 24h window (default 0.5 SOL)
  XZERO_KILL_SWITCH        — set to 'true' to hard-stop all trades immediately

Usage:
    from core.xzero_circuit_breaker import circuit_breaker as cb
    if not cb.allow_trade():
        return  # blocked
    # ... execute trade ...
    cb.record_result(pnl_sol=-0.05)  # negative = loss

Status endpoint (add to FastAPI):
    from core.xzero_circuit_breaker import circuit_breaker
    @app.get('/status/circuit_breaker')
    def cb_status(): return circuit_breaker.status()
"""
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List

log = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    max_drawdown_pct: float = float(os.getenv("XZERO_MAX_DRAWDOWN_PCT", "10.0"))
    max_consecutive_losses: int = int(os.getenv("XZERO_MAX_CONSECUTIVE_L", "3"))
    daily_loss_sol_limit: float = float(os.getenv("XZERO_DAILY_LOSS_SOL", "0.5"))

    _consecutive_losses: int = field(default=0, init=False, repr=False)
    _trade_log: List[dict] = field(default_factory=list, init=False, repr=False)
    _peak_balance_sol: float = field(default=0.0, init=False, repr=False)
    _current_balance_sol: float = field(default=0.0, init=False, repr=False)
    _tripped: bool = field(default=False, init=False, repr=False)
    _trip_reason: str = field(default="", init=False, repr=False)

    @property
    def kill_switch(self) -> bool:
        return os.getenv("XZERO_KILL_SWITCH", "false").lower() == "true"

    def set_balance(self, sol: float) -> None:
        """Call at startup with current wallet balance."""
        self._current_balance_sol = sol
        if sol > self._peak_balance_sol:
            self._peak_balance_sol = sol

    def allow_trade(self, signal_urgency: float = 0.5) -> bool:
        """Returns True if trade is allowed. Call before every trade."""
        if self.kill_switch:
            log.warning("[CircuitBreaker] KILL SWITCH active — trade blocked.")
            return False
        if self._tripped:
            log.warning(f"[CircuitBreaker] TRIPPED ({self._trip_reason}) — trade blocked.")
            return False
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"{self._consecutive_losses} consecutive losses")
            return False
        daily_loss = self._daily_loss_sol()
        if daily_loss >= self.daily_loss_sol_limit:
            self._trip(f"daily loss {daily_loss:.4f} SOL >= limit {self.daily_loss_sol_limit}")
            return False
        if self._peak_balance_sol > 0:
            drawdown_pct = (self._peak_balance_sol - self._current_balance_sol) / self._peak_balance_sol * 100
            if drawdown_pct >= self.max_drawdown_pct:
                self._trip(f"drawdown {drawdown_pct:.2f}% >= limit {self.max_drawdown_pct}%")
                return False
        return True

    def record_result(self, pnl_sol: float) -> None:
        """Call after every trade."""
        self._trade_log.append({"pnl": pnl_sol, "ts": time.time()})
        self._current_balance_sol += pnl_sol
        if self._current_balance_sol > self._peak_balance_sol:
            self._peak_balance_sol = self._current_balance_sol
        if pnl_sol < 0:
            self._consecutive_losses += 1
            log.info(f"[CircuitBreaker] Loss #{self._consecutive_losses}: {pnl_sol:.4f} SOL")
        else:
            self._consecutive_losses = 0
            log.info(f"[CircuitBreaker] Win: +{pnl_sol:.4f} SOL — streak reset.")

    def reset(self) -> None:
        """Manual reset after reviewing losses."""
        self._tripped = False
        self._trip_reason = ""
        self._consecutive_losses = 0
        log.warning("[CircuitBreaker] Manually reset.")

    def status(self) -> dict:
        return {
            "tripped": self._tripped,
            "trip_reason": self._trip_reason,
            "consecutive_losses": self._consecutive_losses,
            "daily_loss_sol": round(self._daily_loss_sol(), 4),
            "daily_loss_limit_sol": self.daily_loss_sol_limit,
            "drawdown_pct": round(
                (self._peak_balance_sol - self._current_balance_sol)
                / max(self._peak_balance_sol, 0.0001) * 100, 2
            ),
            "max_drawdown_pct": self.max_drawdown_pct,
            "kill_switch": self.kill_switch,
        }

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self._trip_reason = reason
        log.error(f"[CircuitBreaker] TRIPPED: {reason}. All trading halted. Call cb.reset() to resume.")

    def _daily_loss_sol(self) -> float:
        cutoff = time.time() - 86400
        return abs(sum(t["pnl"] for t in self._trade_log if t["ts"] >= cutoff and t["pnl"] < 0))


# Global singleton
circuit_breaker = CircuitBreaker()
