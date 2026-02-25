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

# v7.0.0 — CONNECTOR_REGISTRY with lazy loading
from .sparknet_connector     import SparkNetConnector, get_sparknet   # stdlib-only, safe to import
from .solscan_free_connector import SolscanFreeConnector                # no API key needed

CONNECTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "manifold":     ("extensions.xzero.manifold_connector",     "ManifoldConnector"),
    "kalshi":       ("extensions.xzero.kalshi_connector",       "KalshiConnector"),
    "hyperliquid":  ("extensions.xzero.hyperliquid_connector",  "HyperliquidConnector"),
    "jupiter":      ("extensions.xzero.jupiter_enhanced",       "JupiterEnhancedConnector"),
    "polyterm":     ("extensions.xzero.polyterm_connector",     "PolytermConnector"),
    "gitnexus":     ("extensions.xzero.gitnexus_connector",     "GitNexusMCPConnector"),
    "moonpay":      ("extensions.xzero.moonpay_agents_connector", "MoonPayAgentsConnector"),
    "solscan":      ("extensions.xzero.solscan_free_connector", "SolscanFreeConnector"),
    "vinext":       ("extensions.xzero.vinext_connector",       "VinextReplicateConnector"),
    "delegation":   ("extensions.xzero.xzero_delegation",      "XZeroDelegation"),
    "sparknet":     ("extensions.xzero.sparknet_connector",     "SparkNetConnector"),
    "potpie":       ("extensions.xzero.potpie_connector",       "PotpieConnector"),
    "glint":        ("extensions.xzero.glint_signal_bus",         "GlintSignalBus"),       # GlintIntel multi-source signal aggregator
    "apify":        ("extensions.xzero.apify_mcp_connector",     "ApifyMCPConnector"),    # 15k+ web scraping & OSINT actors
    "polymarket":   ("extensions.xzero.polymarket_connector",     "PolymarketConnector"),  # with closed-candle gating
}

def get_connector(name: str):
    """Load any connector by name (lazy import, avoids heavy startup cost).

    Example:
        polyterm = get_connector("polyterm")()
        pulse    = await polyterm.solana_prediction_pulse()

        potpie   = get_connector("potpie")()
        result   = await potpie.query_agent(project_id="godlocal", question="How does AgentPool route tasks?")
    """
    if name not in CONNECTOR_REGISTRY:
        raise KeyError(f"Unknown: {name!r}. Available: {sorted(CONNECTOR_REGISTRY)}")
    module, cls = CONNECTOR_REGISTRY[name]
    import importlib
    return getattr(importlib.import_module(module), cls)

__all__ = [
    "XZroConnector",
    "ManifoldConnector",
    "KalshiConnector",
    "HyperliquidConnector",
    "JupiterEnhancedConnector",
    "SparkNetConnector",
    "SolscanFreeConnector",
    "get_sparknet",
    "get_connector",
    "ApifyMCPConnector",
    "CONNECTOR_REGISTRY",
    "SteerlingConnector",
    "PotpieConnector",
    "GlintSignalBus",
    "get_signal_bus",
    "ClosedCandleGate",
    "get_candle_gate",
]
