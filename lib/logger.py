#!/usr/bin/env python3
"""
lib/logger.py — Trade logging for the finance3 trading agent.

Maintains a running CSV log of every trade decision (buys, sells, skips,
session starts, and daily-limit halts) with full indicator reasoning.

trade_log.csv columns:
  date, time, ticker, action, shares, price, trade_value,
  reason, score, rsi, macd_direction, momentum_pct, volume_ratio,
  rsi_score, macd_score, momentum_score, volume_score,
  portfolio_cash_after, portfolio_value_after
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# All CSV columns in order
COLUMNS = [
    "date",
    "time",
    "ticker",
    "action",            # BUY | SELL | SELL_STOP_LOSS | SELL_SIGNAL | SKIP | HALT | SESSION_START
    "shares",
    "price",
    "trade_value",       # shares * price (0 for non-trade actions)
    "reason",            # Full text reasoning
    "score",             # Composite indicator score (blank for non-scored actions)
    "rsi",
    "macd_direction",
    "momentum_pct",
    "volume_ratio",
    "rsi_score",
    "macd_score",
    "momentum_score",
    "volume_score",
    "portfolio_cash_after",
    "portfolio_value_after",
]


class TradeLogger:
    """Appends structured rows to trade_log.csv."""

    def __init__(self, config) -> None:
        self.cfg = config
        self.filepath = config.TRADE_LOG_FILE
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS)
                    writer.writeheader()
                logger.info("Created trade log: %s", self.filepath)
            except IOError as exc:
                logger.error("Could not create trade log: %s", exc)

    def _append(self, row: dict) -> None:
        """Append a single row to the CSV, filling missing columns with ''."""
        full_row = {col: row.get(col, "") for col in COLUMNS}
        try:
            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writerow(full_row)
        except IOError as exc:
            logger.error("Failed to append to trade log: %s", exc)

    def _now(self) -> tuple[str, str]:
        """Return (date_str, time_str) for now."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    # ── Public logging methods ────────────────────────────────────────────

    def log_session_start(
        self,
        portfolio_value: float,
        n_holdings: int,
        cash: float,
    ) -> None:
        """Log the start of a trading session."""
        date, time = self._now()
        reason = (
            f"Session start: portfolio=${portfolio_value:,.2f}, "
            f"cash=${cash:,.2f}, positions={n_holdings}"
        )
        self._append({
            "date": date,
            "time": time,
            "ticker": "—",
            "action": "SESSION_START",
            "reason": reason,
            "portfolio_cash_after": f"{cash:.2f}",
            "portfolio_value_after": f"{portfolio_value:.2f}",
        })
        logger.info(reason)

    def log_trade(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        reason: str,
        scores: Optional[dict] = None,
        portfolio_cash: Optional[float] = None,
        portfolio_value: Optional[float] = None,
    ) -> None:
        """
        Log a completed BUY or SELL trade.

        Args:
            ticker:           Stock symbol
            action:           "BUY", "SELL", "SELL_STOP_LOSS", "SELL_SIGNAL"
            shares:           Number of shares traded
            price:            Execution price per share
            reason:           Full reasoning string
            scores:           Signal dict from strategy.py (optional)
            portfolio_cash:   Cash remaining after trade
            portfolio_value:  Portfolio value after trade
        """
        date, time = self._now()
        trade_value = shares * price

        row: dict = {
            "date": date,
            "time": time,
            "ticker": ticker,
            "action": action,
            "shares": shares,
            "price": f"{price:.4f}",
            "trade_value": f"{trade_value:.2f}",
            "reason": reason,
        }

        if scores:
            row.update({
                "score":            scores.get("score", ""),
                "rsi":              scores.get("rsi", ""),
                "macd_direction":   scores.get("macd_direction", ""),
                "momentum_pct":     scores.get("momentum_pct", ""),
                "volume_ratio":     scores.get("volume_ratio", ""),
                "rsi_score":        scores.get("rsi_score", ""),
                "macd_score":       scores.get("macd_score", ""),
                "momentum_score":   scores.get("momentum_score", ""),
                "volume_score":     scores.get("volume_score", ""),
            })

        if portfolio_cash is not None:
            row["portfolio_cash_after"] = f"{portfolio_cash:.2f}"
        if portfolio_value is not None:
            row["portfolio_value_after"] = f"{portfolio_value:.2f}"

        self._append(row)
        logger.info(
            "LOGGED %s: %s %d shares @ $%.2f | %s",
            action, ticker, shares, price, reason[:80],
        )

    def log_skipped(
        self,
        ticker: str,
        reason: str,
        scores: Optional[dict] = None,
    ) -> None:
        """Log a stock that was considered but NOT traded."""
        date, time = self._now()
        row: dict = {
            "date": date,
            "time": time,
            "ticker": ticker,
            "action": "SKIP",
            "shares": 0,
            "price": "",
            "trade_value": "",
            "reason": reason,
        }
        if scores:
            row.update({
                "score":            scores.get("score", ""),
                "rsi":              scores.get("rsi", ""),
                "macd_direction":   scores.get("macd_direction", ""),
                "momentum_pct":     scores.get("momentum_pct", ""),
                "volume_ratio":     scores.get("volume_ratio", ""),
            })
        self._append(row)
        logger.debug("SKIP %s: %s", ticker, reason)

    def log_halt(self, reason: str, portfolio_value: float, cash: float) -> None:
        """Log a trading halt (daily loss limit or other circuit breaker)."""
        date, time = self._now()
        self._append({
            "date": date,
            "time": time,
            "ticker": "—",
            "action": "HALT",
            "reason": reason,
            "portfolio_cash_after": f"{cash:.2f}",
            "portfolio_value_after": f"{portfolio_value:.2f}",
        })
        logger.warning("HALT logged: %s", reason)

    def log_market_closed(self, next_open: str = "") -> None:
        """Log when the agent runs but market is closed."""
        date, time = self._now()
        reason = f"Market closed. Next open: {next_open}" if next_open else "Market closed."
        self._append({
            "date": date,
            "time": time,
            "ticker": "—",
            "action": "SKIP",
            "reason": reason,
        })
        logger.info(reason)
