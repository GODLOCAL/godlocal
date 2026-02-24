"""
extensions/xzero — X-ZERO market connectors.
Each connector: READ-ONLY market digest + trade signal assessment.
No connector executes real trades without explicit LOCKED_LIMITS approval.
"""
from .base import XZeroConnector
from .manifold_connector import ManifoldConnector
from .kalshi_connector import KalshiConnector
from .hyperliquid_connector import HyperliquidConnector
from .jupiter_enhanced import JupiterEnhancedConnector

CONNECTORS = {
    "manifold":   ManifoldConnector,
    "kalshi":     KalshiConnector,
    "hyperliquid": HyperliquidConnector,
    "jupiter":    JupiterEnhancedConnector,
}

__all__ = [
    "XZeroConnector",
    "ManifoldConnector",
    "KalshiConnector",
    "HyperliquidConnector",
    "JupiterEnhancedConnector",
    "CONNECTORS",
]
