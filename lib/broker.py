#!/usr/bin/env python3
"""
lib/broker.py — Alpaca API wrapper for the finance3 trading agent.

Handles all communication with the Alpaca paper-trading API:
  - Market status checks
  - Account / position queries
  - Order submission
  - Price data retrieval

All dollar amounts and quantities are Python floats/ints (not Alpaca's
internal Decimal strings) for ease of downstream arithmetic.
"""

from __future__ import annotations

import time
import logging
from datetime import datetime
from typing import Optional

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------
# Alpaca uses '/' for tickers that yfinance/Wikipedia represent with '-'.
# e.g. yfinance: "BRK-B"  →  Alpaca: "BRK/B"
# All broker methods call _to_alpaca() before hitting the API, and
# _from_alpaca() when returning symbols to the rest of the codebase, so
# the translation is invisible to callers.

_TO_ALPACA: dict[str, str] = {
    "BRK-A": "BRK/A",
    "BRK-B": "BRK/B",
    "BF-A":  "BF/A",
    "BF-B":  "BF/B",
}
_FROM_ALPACA: dict[str, str] = {v: k for k, v in _TO_ALPACA.items()}


def _to_alpaca(symbol: str) -> str:
    """Convert internal hyphen-format symbol to Alpaca's slash format."""
    return _TO_ALPACA.get(symbol, symbol)


def _from_alpaca(symbol: str) -> str:
    """Convert Alpaca's slash-format symbol back to internal hyphen format."""
    return _FROM_ALPACA.get(symbol, symbol)


class AlpacaBroker:
    """Thin wrapper around the Alpaca REST API (paper trading)."""

    def __init__(self, api_key: str, api_secret: str, base_url: str) -> None:
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
            api_version="v2",
        )
        logger.info("AlpacaBroker initialised — endpoint: %s", base_url)

    # ── Market status ─────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Return True if the US stock market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except APIError as exc:
            logger.error("Failed to fetch market clock: %s", exc)
            return False

    def get_next_market_open(self) -> datetime:
        """Return the next market open time (UTC-aware datetime)."""
        clock = self.api.get_clock()
        return clock.next_open.item()  # returns Python datetime

    # ── Account ───────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """
        Return a simplified account snapshot.

        Returns:
            {
                "cash": float,
                "portfolio_value": float,
                "buying_power": float,
                "equity": float,
            }
        """
        acct = self.api.get_account()
        return {
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
        }

    # ── Positions ─────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """
        Return all current open positions.

        Returns list of:
            {
                "symbol": str,
                "qty": int,
                "avg_entry_price": float,
                "current_price": float,
                "market_value": float,
                "unrealized_pl": float,
                "unrealized_pl_pct": float,  # e.g. -0.083 means -8.3%
            }
        """
        try:
            positions = self.api.list_positions()
        except APIError as exc:
            logger.error("Failed to fetch positions: %s", exc)
            return []

        result = []
        for p in positions:
            result.append(
                {
                    "symbol": _from_alpaca(p.symbol),
                    "qty": int(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_pl_pct": float(p.unrealized_plpc),  # already a fraction
                }
            )
        return result

    def get_position(self, symbol: str) -> Optional[dict]:
        """Return a single position dict, or None if not held."""
        try:
            p = self.api.get_position(_to_alpaca(symbol))
            return {
                "symbol": _from_alpaca(p.symbol),
                "qty": int(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_pl_pct": float(p.unrealized_plpc),
            }
        except APIError:
            return None

    # ── Prices ─────────────────────────────────────────────────────────────

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Return the latest trade price for a single symbol.
        Returns None on failure.
        """
        try:
            trade = self.api.get_latest_trade(_to_alpaca(symbol))
            return float(trade.price)
        except APIError as exc:
            logger.warning("Could not fetch price for %s: %s", symbol, exc)
            return None

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Return a {symbol: price} dict for a list of symbols.
        Symbols that fail are omitted from the result.
        """
        prices: dict[str, float] = {}
        # Batch in groups of 50 to avoid URL length issues
        for i in range(0, len(symbols), 50):
            batch = symbols[i : i + 50]
            alpaca_batch = [_to_alpaca(s) for s in batch]
            try:
                trades = self.api.get_latest_trades(alpaca_batch)
                for alpaca_sym, trade in trades.items():
                    prices[_from_alpaca(alpaca_sym)] = float(trade.price)
            except APIError as exc:
                logger.warning("Batch price fetch failed: %s", exc)
                # Fall back to individual fetches for this batch
                for sym in batch:
                    p = self.get_current_price(sym)
                    if p is not None:
                        prices[sym] = p
        return prices

    # ── Orders ────────────────────────────────────────────────────────────

    def submit_market_order(
        self, symbol: str, qty: int, side: str, time_in_force: str = "day"
    ) -> Optional[dict]:
        """
        Submit a market order.

        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            qty:    Number of shares (positive integer)
            side:   "buy" or "sell"
            time_in_force: "day" (default) or "gtc"

        Returns:
            Order dict on success, None on failure.
        """
        if qty <= 0:
            logger.warning("Skipping %s %s — qty=%d is invalid", side, symbol, qty)
            return None

        try:
            order = self.api.submit_order(
                symbol=_to_alpaca(symbol),
                qty=qty,
                side=side,
                type="market",
                time_in_force=time_in_force,
            )
            logger.info(
                "ORDER SUBMITTED: %s %d shares of %s | id=%s",
                side.upper(),
                qty,
                symbol,
                order.id,
            )
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": int(order.qty),
                "side": order.side,
                "status": order.status,
                "submitted_at": str(order.submitted_at),
            }
        except APIError as exc:
            logger.error(
                "Order failed: %s %d %s —", side, qty, symbol, exc
            )
            return None

    def cancel_all_orders(self) -> None:
        """Cancel all open orders (called during stop-loss or halt scenarios)."""
        try:
            self.api.cancel_all_orders()
            logger.info("All open orders cancelled.")
        except APIError as exc:
            logger.error("Failed to cancel orders: %s", exc)

    def wait_for_order_fill(self, order_id: str, timeout: int = 30) -> bool:
        """
        Poll until an order is filled or timeout expires.

        Args:
            order_id: Alpaca order ID string
            timeout:  Max seconds to wait

        Returns:
            True if filled, False if timed out or failed.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                order = self.api.get_order(order_id)
                if order.status == "filled":
                    return True
                if order.status in ("canceled", "expired", "rejected"):
                    logger.warning("Order %s ended with status: %s", order_id, order.status)
                    return False
            except APIError as exc:
                logger.error("Error polling order %s: %s", order_id, exc)
                return False
            time.sleep(2)
        logger.warning("Order %s did not fill within %ds", order_id, timeout)
        return False
