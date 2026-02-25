"""
extensions/xzero/moonpay_agents_connector.py
X-ZERO Connector wrapping MoonPay Agents CLI.

MoonPay Agents (launched Feb 25, 2026 by @maxwallenberg) gives AI agents
non-custodial wallets + fiat rails + 54 crypto tools via @moonpay/cli.

For X100 OASIS:
  - Agent wallet funding: fiat → crypto via Apple Pay / Venmo / PayPal
  - Onramp / Swap / Trade / Offramp — full lifecycle
  - Virtual Accounts (GBP / EUR / USD) for company Squads multisig
  - Auto-fund X-ZERO agent wallets before Jupiter swaps

Install CLI (one-time):
    npm install -g @moonpay/cli

Source: moonpay.com/agents | @maxwallenberg tweet 2026-02-25
Requires: MOONPAY_API_KEY (publishable key from moonpay.com dashboard)
          MOONPAY_SECRET_KEY (for signed requests / virtual accounts)
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import httpx

from .cimd_connector_base import CIMDConnectorBase


class MoonPayAgentsConnector(CIMDConnectorBase):
    """
    GodLocal connector for MoonPay Agents infrastructure.

    Provides X-ZERO agents with:
    - Non-custodial wallet provisioning
    - Fiat → crypto onramp (Apple Pay / Venmo / PayPal / bank)
    - Cross-chain swaps
    - Crypto → fiat offramp
    - Virtual Accounts (GBP/EUR/USD bank rails)
    - Portfolio / transaction history

    Auth: RSA signature or MOONPAY_SECRET_KEY HMAC depending on endpoint.
    """

    name = "moonpay_agents"
    BASE_API = "https://api.moonpay.com"
    WIDGET_BASE = "https://buy.moonpay.com"

    def __init__(self):
        self.api_key = os.environ.get("MOONPAY_API_KEY", "")
        self.secret_key = os.environ.get("MOONPAY_SECRET_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "MOONPAY_API_KEY not set. Get it from moonpay.com/dashboard → Developers → API Keys"
            )

    # ── CIMDConnectorBase ────────────────────────────────────────────────────
    def openapi_schema(self) -> dict:
        return {
            "openapi": "3.0.0",
            "info": {"title": "MoonPay Agents (X-ZERO)", "version": "1.0"},
            "paths": {
                "/onramp_url": {"get": {"summary": "Generate fiat → crypto onramp widget URL"}},
                "/offramp_url": {"get": {"summary": "Generate crypto → fiat offramp widget URL"}},
                "/swap_url": {"get": {"summary": "Generate swap widget URL"}},
                "/virtual_accounts": {"get": {"summary": "List virtual accounts (GBP/EUR/USD)"}},
                "/virtual_account_transactions": {"get": {"summary": "Virtual account transaction history"}},
                "/currencies": {"get": {"summary": "List supported currencies"}},
                "/buy_quote": {"get": {"summary": "Get buy price quote"}},
                "/sell_quote": {"get": {"summary": "Get sell price quote"}},
                "/transactions": {"get": {"summary": "Agent transaction history"}},
                "/cli_exec": {"post": {"summary": "Execute @moonpay/cli command"}},
            },
        }

    def registration_manifest(self) -> dict:
        return {
            "id": "moonpay_agents",
            "name": "MoonPay Agents (Fiat ↔ Crypto)",
            "description": (
                "Non-custodial infrastructure for AI agent wallets. "
                "Onramp (fiat → crypto), swap, trade, offramp (crypto → fiat). "
                "Virtual Accounts: GBP/EUR/USD bank rails 24/7. "
                "54 crypto tools via @moonpay/cli. "
                "Requires MOONPAY_API_KEY env var."
            ),
            "tools": [
                "onramp_url", "offramp_url", "swap_url",
                "virtual_accounts", "virtual_account_transactions",
                "currencies", "buy_quote", "sell_quote",
                "transactions", "cli_exec",
            ],
            "auth": {"type": "api_key", "env": "MOONPAY_API_KEY"},
        }

    def run_tool(self, tool: str, params: dict) -> Any:
        DISPATCH = {
            "onramp_url":                   self.onramp_url,
            "offramp_url":                  self.offramp_url,
            "swap_url":                     self.swap_url,
            "virtual_accounts":             self.virtual_accounts,
            "virtual_account_transactions": self.virtual_account_transactions,
            "currencies":                   self.currencies,
            "buy_quote":                    self.buy_quote,
            "sell_quote":                   self.sell_quote,
            "transactions":                 self.transactions,
            "cli_exec":                     self.cli_exec,
        }
        if tool not in DISPATCH:
            raise ValueError(f"Unknown tool: {tool}")
        return DISPATCH[tool](**params)

    # ── Widget URL builders (signed) ─────────────────────────────────────────

    def onramp_url(
        self,
        wallet_address: str,
        currency_code: str = "sol",
        base_currency_code: str = "usd",
        base_currency_amount: Optional[float] = None,
        color_code: str = "00FF41",           # X100 neon green
    ) -> dict:
        """
        Generate signed onramp widget URL.
        User opens URL in browser → pays with Apple Pay / Venmo / PayPal → agent receives crypto.
        """
        params: dict = {
            "apiKey": self.api_key,
            "currencyCode": currency_code,
            "walletAddress": wallet_address,
            "colorCode": f"%23{color_code}",
        }
        if base_currency_amount:
            params["baseCurrencyAmount"] = base_currency_amount
        if base_currency_code:
            params["baseCurrencyCode"] = base_currency_code

        url = self._sign_widget_url(self.WIDGET_BASE, params)
        return {"url": url, "wallet_address": wallet_address, "currency": currency_code}

    def offramp_url(
        self,
        wallet_address: str,
        currency_code: str = "sol",
        base_currency_code: str = "usd",
    ) -> dict:
        """Generate signed offramp URL. Agent sells crypto → fiat to bank."""
        params = {
            "apiKey": self.api_key,
            "cryptoCurrencyCode": currency_code,
            "walletAddress": wallet_address,
            "baseCurrencyCode": base_currency_code,
        }
        url = self._sign_widget_url("https://sell.moonpay.com", params)
        return {"url": url, "wallet_address": wallet_address}

    def swap_url(
        self,
        wallet_address: str,
        from_currency: str = "usdc",
        to_currency: str = "sol",
    ) -> dict:
        """Generate signed swap widget URL."""
        params = {
            "apiKey": self.api_key,
            "baseCurrencyCode": from_currency,
            "quoteCurrencyCode": to_currency,
            "walletAddress": wallet_address,
        }
        url = self._sign_widget_url("https://swap.moonpay.com", params)
        return {"url": url, "from": from_currency, "to": to_currency}

    # ── REST API ─────────────────────────────────────────────────────────────

    def currencies(self) -> dict:
        """List all supported currencies (fiat + crypto)."""
        return self._get("/v3/currencies")

    def buy_quote(
        self,
        currency_code: str = "sol",
        base_currency_code: str = "usd",
        base_currency_amount: float = 100.0,
    ) -> dict:
        """Get real-time buy price quote."""
        return self._get("/v3/currencies/{}/buy_quote".format(currency_code), {
            "baseCurrencyCode": base_currency_code,
            "baseCurrencyAmount": base_currency_amount,
            "apiKey": self.api_key,
        })

    def sell_quote(
        self,
        base_currency_code: str = "sol",
        quoted_currency_code: str = "usd",
        base_currency_amount: float = 1.0,
    ) -> dict:
        """Get real-time sell/offramp price quote."""
        return self._get("/v3/currencies/{}/sell_quote".format(base_currency_code), {
            "quotedCurrencyCode": quoted_currency_code,
            "baseCurrencyAmount": base_currency_amount,
            "apiKey": self.api_key,
        })

    def transactions(self, limit: int = 50, offset: int = 0) -> dict:
        """Agent transaction history (onramps + offramps)."""
        return self._get_signed("/v1/transactions", {
            "limit": limit, "offset": offset
        })

    # ── Virtual Accounts (GBP / EUR / USD bank rails) ────────────────────────

    def virtual_accounts(self) -> dict:
        """
        List virtual accounts for this API key.
        Use for X100 company Squads multisig — receive fiat directly.
        """
        return self._get_signed("/v1/virtual-accounts")

    def virtual_account_transactions(
        self,
        account_id: str,
        limit: int = 50,
    ) -> dict:
        """Get transaction history for a virtual account."""
        return self._get_signed(f"/v1/virtual-accounts/{account_id}/transactions", {
            "limit": limit
        })

    # ── MoonPay CLI wrapper ──────────────────────────────────────────────────

    def cli_exec(self, command: str, timeout: int = 30) -> dict:
        """
        Execute @moonpay/cli command.

        Requires: npm install -g @moonpay/cli

        Examples:
            cli_exec("moonpay wallet create")
            cli_exec("moonpay onramp --currency sol --amount 100")
            cli_exec("moonpay swap --from usdc --to sol --amount 50")

        SECURITY: Only whitelisted commands are allowed.
        """
        ALLOWED_PREFIXES = [
            "moonpay wallet",
            "moonpay onramp",
            "moonpay offramp",
            "moonpay swap",
            "moonpay balance",
            "moonpay quote",
            "moonpay account",
        ]
        if not any(command.startswith(p) for p in ALLOWED_PREFIXES):
            raise PermissionError(
                f"CLI command not whitelisted: {command!r}. "
                f"Allowed prefixes: {ALLOWED_PREFIXES}"
            )
        env = {**os.environ, "MOONPAY_API_KEY": self.api_key}
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "command": command,
        }

    # ── Internal HTTP helpers ────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict:
        p = {"apiKey": self.api_key, **(params or {})}
        with httpx.Client(timeout=20) as c:
            r = c.get(f"{self.BASE_API}{path}", params=p)
            r.raise_for_status()
            return r.json()

    def _get_signed(self, path: str, params: dict | None = None) -> dict:
        """Signed request for protected endpoints (Virtual Accounts, etc.)."""
        if not self.secret_key:
            raise EnvironmentError(
                "MOONPAY_SECRET_KEY required for this endpoint. "
                "Get it from moonpay.com/dashboard."
            )
        timestamp = int(time.time() * 1000)
        p = {"apiKey": self.api_key, "timestamp": timestamp, **(params or {})}
        query = urllib.parse.urlencode(p)
        sig = hmac.new(
            self.secret_key.encode(),
            f"{path}?{query}".encode(),
            hashlib.sha256,
        ).hexdigest()
        p["signature"] = sig
        with httpx.Client(timeout=20) as c:
            r = c.get(f"{self.BASE_API}{path}", params=p)
            r.raise_for_status()
            return r.json()

    def _sign_widget_url(self, base_url: str, params: dict) -> str:
        """
        Sign a widget URL with MOONPAY_SECRET_KEY (HMAC-SHA256).
        Required for production widget embeds.
        """
        query = urllib.parse.urlencode(params)
        widget_url = f"?{query}"
        if self.secret_key:
            sig = hmac.new(
                self.secret_key.encode(),
                widget_url.encode(),
                hashlib.sha256,
            ).hexdigest()
            query += f"&signature={sig}"
        return f"{base_url}?{query}"
