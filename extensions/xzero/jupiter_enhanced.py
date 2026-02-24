"""
extensions/xzero/jupiter_enhanced.py — Jupiter Solana Swap Connector (v2)
Enhanced over original jupiter_connector: adds best-route scan, price impact guard.
API: https://station.jup.ag/docs/apis/swap-api
"""
from __future__ import annotations
import logging
from typing import Optional
import requests
from .base import XZeroConnector

logger = logging.getLogger(__name__)

JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API  = "https://quote-api.jup.ag/v6/swap"
SOL_MINT  = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
MAX_PRICE_IMPACT_BPS = 100  # 1% max price impact


class JupiterEnhancedConnector(XZeroConnector):
    NAME = "jupiter_enhanced"

    def __init__(self, wallet_pubkey: Optional[str] = None, daily_limit_sol: float = 1.0):
        self.wallet_pubkey = wallet_pubkey
        self.daily_limit_sol = daily_limit_sol

    def get_quote(self, input_mint: str, output_mint: str, amount_lamports: int) -> Optional[dict]:
        """Get best swap route quote."""
        try:
            resp = requests.get(JUPITER_QUOTE_API, params={
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount_lamports,
                "slippageBps": 50,
            }, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"[Jupiter] Quote failed: {e}")
            return None

    def market_digest(self) -> list[dict]:
        """
        Sample SOL→USDC and USDC→SOL quotes as market signal.
        Returns current rate + price impact for both directions.
        """
        sol_amount = int(0.1 * 1e9)  # 0.1 SOL
        q_sell = self.get_quote(SOL_MINT, USDC_MINT, sol_amount)
        q_buy  = self.get_quote(USDC_MINT, SOL_MINT, int(10 * 1e6))  # 10 USDC

        digest = []
        if q_sell:
            digest.append({
                "id": "sol_usdc",
                "direction": "SOL→USDC",
                "out_amount": q_sell.get("outAmount"),
                "price_impact_pct": q_sell.get("priceImpactPct"),
                "route_plan": len(q_sell.get("routePlan", [])),
                "platform": "jupiter",
            })
        if q_buy:
            digest.append({
                "id": "usdc_sol",
                "direction": "USDC→SOL",
                "out_amount": q_buy.get("outAmount"),
                "price_impact_pct": q_buy.get("priceImpactPct"),
                "route_plan": len(q_buy.get("routePlan", [])),
                "platform": "jupiter",
            })
        return digest

    def assess_for_trade(self, opp: dict) -> dict:
        """Gate on price impact — reject if >MAX_PRICE_IMPACT_BPS."""
        impact_pct = float(opp.get("price_impact_pct") or 0)
        impact_bps = impact_pct * 100

        if impact_bps > MAX_PRICE_IMPACT_BPS:
            return {"action": "skip", "amount": 0,
                    "reason": f"Price impact too high: {impact_bps:.0f}bps > {MAX_PRICE_IMPACT_BPS}bps",
                    "platform": "jupiter"}
        return {
            "action": "swap_ready",
            "amount": self.daily_limit_sol * 0.1,
            "reason": f"Impact acceptable: {impact_bps:.1f}bps",
            "platform": "jupiter",
        }
