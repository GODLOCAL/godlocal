"""
extensions/ — Optional GodLocal modules

These modules extend GodLocal with domain-specific capabilities.
They are NOT imported by the core and must be explicitly opted-in.

Available extensions:
  xzero/ — X-ZERO trading delegation layer (Eliza → Picobot, Solana)
    - xzero_delegation.py    DynamicAssessor, AdaptiveExecutor, ResilienceGuard
    - polymarket_connector.py  Polymarket READ-ONLY prediction market signals

Usage:
    from extensions.xzero.xzero_delegation import DynamicAssessor, ResilienceGuard
    from extensions.xzero.polymarket_connector import PolymarketConnector
"""
