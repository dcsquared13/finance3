#!/usr/bin/env python3
"""
lib/risk.py — Risk management for the finance3 trading agent.

Provides stateless functions for:
  - Stop-loss evaluation (8% drop from avg entry price)
  - Daily loss limit check (portfolio down 3% from open)
  - Position sizing (equal-weight, integer shares)
  - Available slot calculation
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Stateless risk checks. All methods are pure functions given the config.
    No Alpaca API calls — those are handled in broker.py.
    """

    def __init__(self, config) -> None:
        self.cfg = config

    # ── Stop-loss ─────────────────────────────────────────────────────────

    def check_stop_loss(self, position: dict) -> tuple[bool, str]:
        """
        Evaluate whether a position should be closed due to stop-loss.

        Args:
            position: dict with keys:
                        symbol, qty, avg_entry_price, current_price,
                        unrealized_pl_pct (negative fraction, e.g. -0.083)

        Returns:
            (triggered: bool, reason: str)
        """
        sym = position.get("symbol", "?")
        pl_pct = float(position.get("unrealized_pl_pct", 0.0))
        avg_price = float(position.get("avg_entry_price", 0.0))
        current_price = float(position.get("current_price", 0.0))
        threshold = -self.cfg.STOP_LOSS_PCT

        if pl_pct <= threshold:
            reason = (
                f"Stop-loss triggered: {sym} down {pl_pct:.1%} "
                f"(avg entry ${avg_price:.2f} → current ${current_price:.2f}, "
                f"threshold {threshold:.0%})"
            )
            logger.warning(reason)
            return True, reason

        return False, ""

    def check_stop_losses(self, positions: list[dict]) -> list[tuple[dict, str]]:
        """
        Evaluate stop-loss for all positions.

        Returns:
            List of (position_dict, reason_str) for positions that triggered.
        """
        triggered = []
        for pos in positions:
            hit, reason = self.check_stop_loss(pos)
            if hit:
                triggered.append((pos, reason))
        return triggered

    # ── Daily loss limit ──────────────────────────────────────────────────

    def check_daily_loss_limit(
        self, portfolio_value_at_open: float, current_portfolio_value: float
    ) -> tuple[bool, str]:
        """
        Check if the portfolio has lost enough today to trigger a trading halt.

        Args:
            portfolio_value_at_open:   Portfolio value recorded at session start.
            current_portfolio_value:   Current live portfolio value.

        Returns:
            (triggered: bool, reason: str)
        """
        if portfolio_value_at_open <= 0:
            return False, ""

        loss_pct = (portfolio_value_at_open - current_portfolio_value) / portfolio_value_at_open
        threshold = self.cfg.DAILY_LOSS_LIMIT_PCT

        if loss_pct >= threshold:
            reason = (
                f"Daily loss limit triggered: portfolio down {loss_pct:.2%} "
                f"(${portfolio_value_at_open:,.2f} → ${current_portfolio_value:,.2f}, "
                f"threshold {threshold:.0%})"
            )
            logger.warning(reason)
            return True, reason

        logger.info(
            "Daily P&L: %.2f%% (limit %.0f%%) — OK",
            -loss_pct * 100,
            threshold * 100,
        )
        return False, ""

    # ── Position sizing ───────────────────────────────────────────────────

    def calculate_position_size(
        self, available_cash: float, price: float, n_slots: int
    ) -> int:
        """
        Calculate the number of shares to buy for a new position.

        Strategy: equal-weight across `n_slots` open positions.
        A cash reserve (CASH_RESERVE_PCT) is kept uninvested at all times.

        Args:
            available_cash: Current usable cash balance.
            price:          Current share price.
            n_slots:        Number of positions to fill (available slots).

        Returns:
            Number of whole shares to buy (0 if not enough cash for even 1 share).
        """
        if n_slots <= 0 or price <= 0:
            return 0

        investable_cash = available_cash * (1 - self.cfg.CASH_RESERVE_PCT)
        allocation = investable_cash / n_slots
        shares = math.floor(allocation / price)

        if shares < 1:
            logger.debug(
                "Position size = 0 for price $%.2f (allocation $%.2f, %d slots)",
                price, allocation, n_slots,
            )
            return 0

        cost = shares * price
        logger.info(
            "Position size: %d shares @ $%.2f = $%.2f (slot $%.2f of $%.2f investable)",
            shares, price, cost, allocation, investable_cash,
        )
        return shares

    # ── Slot availability ─────────────────────────────────────────────────

    def slots_available(self, current_position_count: int) -> int:
        """
        Return how many new positions can be opened.

        Args:
            current_position_count: Number of currently held positions.

        Returns:
            Non-negative integer (0 if at max capacity).
        """
        slots = max(0, self.cfg.MAX_POSITIONS - current_position_count)
        logger.info(
            "Position slots: %d/%d used, %d available",
            current_position_count,
            self.cfg.MAX_POSITIONS,
            slots,
        )
        return slots

    # ── Summary ───────────────────────────────────────────────────────────

    def risk_summary(
        self,
        positions: list[dict],
        portfolio_value_at_open: float,
        current_value: float,
    ) -> dict:
        """
        Return a snapshot of current risk state — useful for logging.
        """
        stop_loss_hits = self.check_stop_losses(positions)
        daily_limit_hit, daily_reason = self.check_daily_loss_limit(
            portfolio_value_at_open, current_value
        )
        daily_pnl_pct = (
            (current_value - portfolio_value_at_open) / portfolio_value_at_open
            if portfolio_value_at_open > 0
            else 0.0
        )
        return {
            "n_positions": len(positions),
            "stop_loss_triggers": len(stop_loss_hits),
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "daily_limit_triggered": daily_limit_hit,
            "daily_limit_reason": daily_reason,
        }
