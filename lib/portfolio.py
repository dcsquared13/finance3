#!/usr/bin/env python3
"""
lib/portfolio.py — Portfolio state management for the finance3 trading agent.

Persists the agent's view of its holdings and cash to data/portfolio.json.
Also syncs with Alpaca's live positions to stay accurate across restarts.

portfolio.json schema:
{
  "cash": 8450.00,
  "last_updated": "2026-03-01T09:35:42",
  "portfolio_value_at_open": 10120.00,
  "holdings": {
    "NVDA": {
      "shares": 3,
      "avg_price": 875.50,
      "date_bought": "2026-02-28",
      "cost_basis": 2626.50
    }
  }
}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages the agent's portfolio state, persisted to a JSON file."""

    def __init__(self, config) -> None:
        self.cfg = config
        self.filepath = config.PORTFOLIO_FILE
        self._state: dict = {}

    # ── Load / Save ───────────────────────────────────────────────────────

    def load(self) -> dict:
        """
        Load portfolio state from disk. Creates a fresh portfolio if file doesn't exist.

        Returns:
            The current portfolio state dict.
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self._state = json.load(f)
                logger.info(
                    "Portfolio loaded: cash=$%.2f, %d holdings",
                    self._state.get("cash", 0),
                    len(self._state.get("holdings", {})),
                )
            except (json.JSONDecodeError, IOError) as exc:
                logger.error("Failed to load portfolio (%s) — starting fresh.", exc)
                self._state = self._empty_state()
        else:
            logger.info("No portfolio file found — initialising with $%.2f capital.", self.cfg.INITIAL_CAPITAL)
            self._state = self._empty_state()
            self.save()

        return self._state

    def save(self) -> None:
        """Persist current state to disk."""
        self._state["last_updated"] = datetime.now().isoformat(timespec="seconds")
        try:
            with open(self.filepath, "w") as f:
                json.dump(self._state, f, indent=2)
        except IOError as exc:
            logger.error("Failed to save portfolio: %s", exc)

    def _empty_state(self) -> dict:
        return {
            "cash": self.cfg.INITIAL_CAPITAL,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "portfolio_value_at_open": self.cfg.INITIAL_CAPITAL,
            "holdings": {},
        }

    # ── State accessors ──────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return float(self._state.get("cash", 0.0))

    @property
    def holdings(self) -> dict:
        return self._state.get("holdings", {})

    @property
    def portfolio_value_at_open(self) -> float:
        return float(self._state.get("portfolio_value_at_open", self.cfg.INITIAL_CAPITAL))

    def holding_symbols(self) -> list[str]:
        return list(self._state.get("holdings", {}).keys())

    def n_holdings(self) -> int:
        return len(self._state.get("holdings", {}))

    # ── Updates ───────────────────────────────────────────────────────────

    def record_open_value(self, value: float) -> None:
        """Call once at session start to capture today's opening portfolio value."""
        self._state["portfolio_value_at_open"] = round(value, 2)
        logger.info("Portfolio value at open: $%.2f", value)
        self.save()

    def record_buy(self, ticker: str, shares: int, price: float) -> None:
        """
        Update state after a successful BUY order.
        Handles averaging down if ticker already held.
        """
        holdings = self._state.setdefault("holdings", {})
        cost = shares * price

        if ticker in holdings:
            existing = holdings[ticker]
            old_shares = existing["shares"]
            old_avg = existing["avg_price"]
            new_shares = old_shares + shares
            new_avg = (old_shares * old_avg + cost) / new_shares
            holdings[ticker] = {
                "shares": new_shares,
                "avg_price": round(new_avg, 4),
                "date_bought": existing.get("date_bought", datetime.now().strftime("%Y-%m-%d")),
                "cost_basis": round(new_shares * new_avg, 2),
            }
            logger.info(
                "BUY (add): %s +%d shares @ $%.2f (new total %d @ avg $%.2f)",
                ticker, shares, price, new_shares, new_avg,
            )
        else:
            holdings[ticker] = {
                "shares": shares,
                "avg_price": round(price, 4),
                "date_bought": datetime.now().strftime("%Y-%m-%d"),
                "cost_basis": round(cost, 2),
            }
            logger.info("BUY: %s %d shares @ $%.2f", ticker, shares, price)

        self._state["cash"] = round(self.cash - cost, 2)
        self.save()

    def record_sell(self, ticker: str, shares: int, price: float) -> Optional[float]:
        """
        Update state after a successful SELL order.

        Returns:
            Realized P&L for the sold position, or None if ticker not found.
        """
        holdings = self._state.get("holdings", {})
        if ticker not in holdings:
            logger.warning("SELL: %s not found in local holdings.", ticker)
            return None

        holding = holdings[ticker]
        avg_price = float(holding["avg_price"])
        proceeds = shares * price
        cost_sold = shares * avg_price
        realized_pnl = proceeds - cost_sold

        remaining = holding["shares"] - shares
        if remaining <= 0:
            del holdings[ticker]
            logger.info(
                "SELL (full): %s %d shares @ $%.2f | P&L: $%.2f",
                ticker, shares, price, realized_pnl,
            )
        else:
            holdings[ticker]["shares"] = remaining
            holdings[ticker]["cost_basis"] = round(remaining * avg_price, 2)
            logger.info(
                "SELL (partial): %s %d shares @ $%.2f | P&L: $%.2f | remaining: %d",
                ticker, shares, price, realized_pnl, remaining,
            )

        self._state["cash"] = round(self.cash + proceeds, 2)
        self.save()
        return round(realized_pnl, 2)

    def update_cash(self, new_cash: float) -> None:
        """Directly set cash (used when syncing with Alpaca account)."""
        self._state["cash"] = round(new_cash, 2)

    # ── Sync with Alpaca ──────────────────────────────────────────────────

    def sync_with_alpaca(self, broker) -> None:
        """
        Reconcile local portfolio.json against Alpaca's live positions.
        Alpaca is the source of truth for share counts and cash.
        Local records are updated to match.
        """
        logger.info("Syncing portfolio with Alpaca…")

        account = broker.get_account()
        live_positions = broker.get_positions()

        # Update cash from Alpaca
        self._state["cash"] = round(float(account["cash"]), 2)

        # Build a map of live positions
        live_map = {p["symbol"]: p for p in live_positions}

        # Remove any local holdings no longer in Alpaca
        local_holdings = self._state.setdefault("holdings", {})
        stale = [sym for sym in local_holdings if sym not in live_map]
        for sym in stale:
            logger.info("Sync: removing stale holding %s (no longer in Alpaca)", sym)
            del local_holdings[sym]

        # Add/update holdings from Alpaca
        for sym, pos in live_map.items():
            if sym in local_holdings:
                # Keep our avg_price and date_bought but update share count
                local_holdings[sym]["shares"] = pos["qty"]
            else:
                logger.info("Sync: adding external holding %s from Alpaca", sym)
                local_holdings[sym] = {
                    "shares": pos["qty"],
                    "avg_price": round(pos["avg_entry_price"], 4),
                    "date_bought": datetime.now().strftime("%Y-%m-%d"),
                    "cost_basis": round(pos["qty"] * pos["avg_entry_price"], 2),
                }

        self.save()
        logger.info(
            "Sync complete: cash=$%.2f, %d holdings", self.cash, self.n_holdings()
        )

    # ── Reporting ─────────────────────────────────────────────────────────

    def summary(self, current_prices: Optional[dict] = None) -> str:
        """Return a human-readable portfolio summary string."""
        lines = [
            f"  Cash:     ${self.cash:>10,.2f}",
            f"  Holdings: {self.n_holdings()}",
        ]
        for sym, h in self.holdings.items():
            shares = h["shares"]
            avg_p = h["avg_price"]
            cost = shares * avg_p
            if current_prices and sym in current_prices:
                curr_p = current_prices[sym]
                market_val = shares * curr_p
                pnl = market_val - cost
                pnl_pct = pnl / cost if cost > 0 else 0
                lines.append(
                    f"    {sym:<6} {shares:>4} shares @ avg ${avg_p:.2f} "
                    f"→ ${curr_p:.2f} | P&L ${pnl:+.2f} ({pnl_pct:+.1%})"
                )
            else:
                lines.append(f"    {sym:<6} {shares:>4} shares @ avg ${avg_p:.2f}")
        return "\n".join(lines)
