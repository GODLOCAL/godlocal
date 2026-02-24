"""
extensions/xzero/base.py — Abstract base for all X-ZERO market connectors.
Each connector implements market_digest() and assess_for_trade().
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class XZeroConnector(ABC):
    """Base class for X-ZERO market connectors."""
    NAME: str = "base"

    @abstractmethod
    def market_digest(self) -> list[dict]:
        """Return list of active market opportunities."""
        ...

    @abstractmethod
    def assess_for_trade(self, opp: dict) -> dict:
        """
        Assess single opportunity for trade.
        Returns: {action: "bet"|"skip", amount: float, reason: str}
        """
        ...

    def safe_digest(self) -> list[dict]:
        """market_digest() with error handling — safe for sleep_cycle."""
        try:
            return self.market_digest()
        except Exception as e:
            logger.warning(f"[{self.NAME}] market_digest failed: {e}")
            return []
