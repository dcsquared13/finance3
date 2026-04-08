#!/usr/bin/env python3
"""
config.py — Central configuration for the finance3 trading agent.

All sensitive values (API keys) are loaded from environment variables.
Non-sensitive trading parameters have sensible defaults that can be
overridden via environment variables.

Usage:
    from config import Config
    cfg = Config()
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present


@dataclass
class Config:
    # ── Alpaca credentials ────────────────────────────────────────────────
    APCA_API_KEY_ID: str = field(
        default_factory=lambda: os.environ.get("APCA_API_KEY_ID", "")
    )
    APCA_API_SECRET_KEY: str = field(
        default_factory=lambda: os.environ.get("APCA_API_SECRET_KEY", "")
    )
    APCA_API_BASE_URL: str = field(
        default_factory=lambda: os.environ.get(
            "APCA_API_BASE_URL", "https://paper-api.alpaca.markets"
        )
    )

    # ── Capital ───────────────────────────────────────────────────────────
    INITIAL_CAPITAL: float = 10_000.0

    # ── Risk management ───────────────────────────────────────────────────
    STOP_LOSS_PCT: float = 0.08          # Sell if position drops 8% from avg price
    DAILY_LOSS_LIMIT_PCT: float = 0.03   # Halt trading if portfolio down 3% on day

    # ── Position sizing ───────────────────────────────────────────────────
    MAX_POSITIONS: int = 10              # Max concurrent holdings
    CASH_RESERVE_PCT: float = 0.05       # Keep 5% cash as buffer (not invested)

    # ── Signal thresholds ─────────────────────────────────────────────────
    MIN_SCORE_TO_BUY: float = 0.505       # Composite score must exceed this to buy
    SELL_SCORE_THRESHOLD: float = 0.40   # Exit position if score falls below this

    # ── Universe ──────────────────────────────────────────────────────────
    UNIVERSE_SIZE: int = 50              # Number of S&P 500 stocks to scan daily

    # ── Technical indicators ──────────────────────────────────────────────
    LOOKBACK_DAYS: int = 90              # Days of OHLCV history to fetch
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    MOMENTUM_PERIOD: int = 20            # Days for momentum calculation
    VOLUME_AVG_PERIOD: int = 20          # Days for volume moving average

    # ── Indicator weights (must sum to 1.0) ───────────────────────────────
    RSI_WEIGHT: float = 0.25
    MACD_WEIGHT: float = 0.30
    MOMENTUM_WEIGHT: float = 0.25
    VOLUME_WEIGHT: float = 0.20

    # ── Ticker exclusions ─────────────────────────────────────────────────
    # Tickers to skip entirely — never scored, never traded.
    # BRK-B and BF-B are excluded by default because Alpaca paper trading
    # does not list them as tradeable assets (neither hyphen nor slash format).
    EXCLUDED_TICKERS: frozenset = frozenset({"BRK-B", "BRK-A", "BF-B", "BF-A"})

    # ── Congressional trading signal ──────────────────────────────────────
    # When enabled, recent STOCK Act disclosures from House + Senate are
    # fetched and blended into the composite score as an extra signal.
    # Set CONGRESS_WEIGHT=0.0 or CONGRESS_SIGNAL_ENABLED=False to disable.
    CONGRESS_SIGNAL_ENABLED: bool = True
    CONGRESS_WEIGHT: float = 0.10          # Blend weight (0 = off, max ~0.20 recommended)
    CONGRESS_LOOKBACK_DAYS: int = 90       # How far back to look at disclosures

    # ── Paths ─────────────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    PORTFOLIO_FILE: str = "data/portfolio.json"
    TRADE_LOG_FILE: str = "data/trade_log.csv"
    CONGRESS_CACHE_FILE: str = "data/congress_cache.json"

    def validate(self) -> None:
        """Raise ValueError if required settings are missing or invalid."""
        if not self.APCA_API_KEY_ID:
            raise ValueError(
                "APCA_API_KEY_ID not set. Add it to .env or export as environment variable."
            )
        if not self.APCA_API_SECRET_KEY:
            raise ValueError(
                "APCA_API_SECRET_KEY not set. Add it to .env or export as environment variable."
            )
        weights = self.RSI_WEIGHT + self.MACD_WEIGHT + self.MOMENTUM_WEIGHT + self.VOLUME_WEIGHT
        if abs(weights - 1.0) > 0.001:
            raise ValueError(f"Indicator weights must sum to 1.0, got {weights:.3f}")

    def __post_init__(self):
        import os
        os.makedirs(self.DATA_DIR, exist_ok=True)
