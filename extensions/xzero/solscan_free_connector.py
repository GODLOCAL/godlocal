"""
extensions/xzero/solscan_free_connector.py
X-ZERO Connector wrapping the free-solscan-api package.
Gives GodLocal agents zero-cost access to Solana on-chain data
(Solscan Premium API reversed-engineered by @paoloanzn).

No API key required. No rate limits.

Source: github.com/paoloanzn/free-solscan-api
WARNING: Uses Solscan internal website API — not officially supported.
         May break if Solscan updates their frontend bundle.
"""
from __future__ import annotations
import random, string, httpx
from typing import Any
from .cimd_connector_base import CIMDConnectorBase


# ── Token generator (ported from Solscan JS bundle) ─────────────────────────

def _generate_sol_aut() -> str:
    """Replicate Solscan's client-side generateRandomString()."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789==--"
    t = "".join(random.choice(chars) for _ in range(15))
    r = "".join(random.choice(chars) for _ in range(15))
    combined = t + r
    n = random.randint(0, 31)
    return combined[:n] + "B9dls0fK" + combined[n:]


# ── Connector ────────────────────────────────────────────────────────────────

class SolscanFreeConnector(CIMDConnectorBase):
    """
    Zero-cost Solana on-chain data via Solscan internal API.

    Useful for X100 OASIS:
      - Monitor $X100 token holders, transfers, price
      - Track agent wallets: portfolio, DeFi activity, balance history
      - Look up swap transactions (Jupiter/Raydium)
      - Whale detection: top_address_transfers
    """

    name = "solscan_free"
    BASE = "https://api-v2.solscan.io/v2"
    HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://solscan.io/",
        "Origin": "https://solscan.io",
    }

    # ── CIMDConnectorBase ────────────────────────────────────────────────────
    def openapi_schema(self) -> dict:
        return {
            "openapi": "3.0.0",
            "info": {"title": "Solscan Free (X-ZERO)", "version": "1.0"},
            "paths": {
                "/transaction": {
                    "get": {"summary": "Transaction detail", "parameters": [{"name": "tx", "in": "query", "required": True, "schema": {"type": "string"}}]}
                },
                "/transactions": {
                    "get": {"summary": "Account tx history", "parameters": [
                        {"name": "address", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                        {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 40}},
                    ]}
                },
                "/token_data": {
                    "get": {"summary": "Token metadata + price", "parameters": [{"name": "token_address", "in": "query", "required": True, "schema": {"type": "string"}}]}
                },
                "/token_holders": {
                    "get": {"summary": "Token holders list", "parameters": [
                        {"name": "address", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                        {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 100}},
                    ]}
                },
                "/token_holders_total": {
                    "get": {"summary": "Total holder count", "parameters": [{"name": "address", "in": "query", "required": True, "schema": {"type": "string"}}]}
                },
                "/account_info": {
                    "get": {"summary": "Account/wallet info", "parameters": [{"name": "address", "in": "query", "required": True, "schema": {"type": "string"}}]}
                },
                "/portfolio": {
                    "get": {"summary": "Wallet token portfolio", "parameters": [
                        {"name": "address", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "hide_zero", "in": "query", "schema": {"type": "boolean", "default": True}},
                    ]}
                },
                "/balance_history": {
                    "get": {"summary": "SOL balance history", "parameters": [{"name": "address", "in": "query", "required": True, "schema": {"type": "string"}}]}
                },
                "/transfers": {
                    "get": {"summary": "Account transfers", "parameters": [
                        {"name": "address", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                        {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 100}},
                    ]}
                },
                "/defi_activities": {
                    "get": {"summary": "DeFi swap/LP activity", "parameters": [
                        {"name": "address", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                        {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 100}},
                    ]}
                },
                "/top_address_transfers": {
                    "get": {"summary": "Top whale wallets for token", "parameters": [
                        {"name": "address", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "range_days", "in": "query", "schema": {"type": "integer", "default": 7}},
                    ]}
                },
            },
        }

    def registration_manifest(self) -> dict:
        return {
            "id": "solscan_free",
            "name": "Solscan Free (Solana Explorer)",
            "description": (
                "Zero-cost Solana on-chain data: transactions, token holders, "
                "DeFi activity, wallet portfolio, whale tracking. "
                "No API key required. Uses Solscan internal website API."
            ),
            "tools": [
                "transaction", "transactions", "token_data", "token_holders",
                "token_holders_total", "account_info", "portfolio",
                "balance_history", "transfers", "defi_activities",
                "top_address_transfers",
            ],
            "auth": {"type": "none"},
        }

    def run_tool(self, tool: str, params: dict) -> Any:
        DISPATCH = {
            "transaction":           self.transaction,
            "transactions":          self.transactions,
            "token_data":            self.token_data,
            "token_holders":         self.token_holders,
            "token_holders_total":   self.token_holders_total,
            "account_info":          self.account_info,
            "portfolio":             self.portfolio,
            "balance_history":       self.balance_history,
            "transfers":             self.transfers,
            "defi_activities":       self.defi_activities,
            "top_address_transfers": self.top_address_transfers,
        }
        if tool not in DISPATCH:
            raise ValueError(f"Unknown tool: {tool}")
        return DISPATCH[tool](**params)

    # ── Internal HTTP helper ─────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None) -> dict:
        headers = {**self.HEADERS, "sol-aut": _generate_sol_aut()}
        with httpx.Client(timeout=30) as client:
            r = client.get(f"{self.BASE}{path}", params=params or {}, headers=headers)
            r.raise_for_status()
            return r.json()

    # ── Public methods ───────────────────────────────────────────────────────
    def transaction(self, tx: str) -> dict:
        return self._get("/transaction/detail", {"tx": tx})

    def transactions(self, address: str, page: int = 1, page_size: int = 40) -> dict:
        return self._get("/account/transactions", {
            "address": address, "page": page, "page_size": page_size
        })

    def token_data(self, token_address: str) -> dict:
        return self._get("/token/meta", {"address": token_address})

    def token_holders(self, address: str, page: int = 1, page_size: int = 100) -> dict:
        return self._get("/token/holders", {
            "address": address, "page": page, "page_size": page_size
        })

    def token_holders_total(self, address: str) -> dict:
        return self._get("/token/holders/total", {"address": address})

    def account_info(self, address: str) -> dict:
        return self._get("/account/detail", {"address": address})

    def portfolio(
        self, address: str, type: str = "token",
        page: int = 1, page_size: int = 100, hide_zero: bool = True
    ) -> dict:
        return self._get("/account/portfolio", {
            "address": address, "type": type,
            "page": page, "page_size": page_size,
            "hide_zero": str(hide_zero).lower(),
        })

    def balance_history(self, address: str) -> dict:
        return self._get("/account/balance_change_activities", {"address": address})

    def transfers(
        self, address: str, remove_spam: bool = True,
        exclude_amount_zero: bool = True, page: int = 1, page_size: int = 100
    ) -> dict:
        return self._get("/account/transfer", {
            "address": address,
            "remove_spam": str(remove_spam).lower(),
            "exclude_amount_zero": str(exclude_amount_zero).lower(),
            "page": page, "page_size": page_size,
        })

    def defi_activities(self, address: str, page: int = 1, page_size: int = 100) -> dict:
        return self._get("/account/defi_activities", {
            "address": address, "page": page, "page_size": page_size
        })

    def top_address_transfers(self, address: str, range_days: int = 7) -> dict:
        return self._get("/token/top_traders", {
            "address": address, "range_days": range_days
        })
