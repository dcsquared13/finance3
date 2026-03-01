#!/usr/bin/env python3
"""
lib/strategy.py — Multi-indicator signal engine for the finance3 trading agent.

Fetches OHLCV data via yfinance and computes four technical indicators,
combining them into a composite score [0.0 – 1.0] per stock.

Indicators & weights (configurable in config.py):
  RSI (14d)           — 25%  Oversold = buy signal
  MACD (12/26/9)      — 30%  Bullish crossover = buy signal
  Momentum (20d)      — 25%  Positive price momentum vs peers
  Volume Breakout     — 20%  Unusual volume + price confirmation

Signal mapping:
  composite >= 0.60  →  "Buy"
  composite <= 0.40  j��→  "Sell"
  otherwise          →  "Hold"
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger(__name__)

# Static S&P 500 subset — high-liquidity, well-known names used as default universe.
# Agent will scan and score all of these, then trade the top signals.
STATIC_SP500_SUBSET = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO",
    "TSLA", "JPM", "V", "UNH", "XOM", "MA", "PG", "COST", "HD", "MRK", "ABBV",
    "CVX", "KO", "BAC", "PEP", "ADBE", "WMT", "CRM", "MCD", "TMO", "ACN",
    "NFLX", "AMD", "TXN", "ABT", "WFC", "QCOM", "PM", "INTU", "CAT", "DHR",
    "GE", "NEE", "SPGI", "RTX", "LOW", "AMGN", "UPS", "MS", "C", "GS",
]


class SignalEngine:
    """
    Scores a universe of stocks using technical indicators and returns
    ranked buy/sell signals with detailed reasoning strings.
    """

    def __init__(self, config) -> None:
        self.cfg = config

    # ── Universe ─────────────────────────────────────────────────────────

    def get_universe(self) -> list[str]:
        """
        Return the list of ticker symbols to scan.
        Tries to fetch live S&P 500 from Wikipedia; falls back to static list.
        """
        try:
            tickers = self._fetch_sp500_tickers()
            if tickers:
                return tickers[: self.cfg.UNIVERSE_SIZE]
        except Exception as exc:
            logger.warning("Live S&P 500 fetch failed (%s), using static list.", exc)

        return STATIC_SP500_SUBSET[: self.cfg.UNIVERSE_SIZE]

    def _fetch_sp500_tickers(self) -> list[str]:
        """Scrape S&P 500 constituent tickers from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        symbols = tables[0]["Symbol"].tolist()
        # Wikipedia uses dots (e.g. BRK.B); Alpaca/yfinance want dashes
        return [s.replace(".", "-") for s in symbols]

    # ── Data fetching ─────────────────────────────────────────────────────

    def _fetch_ohlcv(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV history for all tickers in one yfinance batch call.

        Returns:
            {symbol: DataFrame with columns [Open, High, Low, Close, Volume]}
        """
        period_map = {90: "3mo", 180: "6mo", 365: "1y"}
        period = period_map.get(self.cfg.LOOKBACK_DAYS, "3mo")

        logger.info("Fetching %dd OHLCV for %d tickers…", self.cfg.LOOKBACK_DAYS, len(tickers))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )

        result: dict[str, pd.DataFrame] = {}
        if len(tickers) == 1:
            sym = tickers[0]
            if not raw.empty:
                result[sym] = raw
        else:
            for sym in tickers:
                try:
                    df = raw[sym].dropna(how="all")
                    if len(df) >= self.cfg.MOMENTUM_PERIOD + 5:
                        result[sym] = df
                except (KeyError, TypeError):
                    pass

        logger.info("Successfully fetched data for %d/%d tickers.", len(result), len(tickers))
        return result

    # ── Indicators ────────────────────────────────────────────────────────

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        """Compute RSI and return the most recent value (0–100)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def _rsi_score(self, rsi: float) -> float:
        """Convert RSI (0–100) to a bullish score (0.0–1.0)."""
        if rsi <= 30:
            return 1.0   # oversold — strong buy
        if rsi >= 70:
            return 0.0   # overbought — strong sell
        # Linear interpolation between 30 and 70
        return (70 - rsi) / 40.0

    def _macd(self, close: pd.Series) -> tuple[float, float, str]:
        """
        Compute MACD line, signal line, and a text interpretation.

        Returns: (macd_value, signal_value, direction_str)
        """
        fast = close.ewm(span=self.cfg.MACD_FAST, adjust=False).mean()
        slow = close.ewm(span=self.cfg.MACD_SLOW, adjust=False).mean()
        macd_line = fast - slow
        signal_line = macd_line.ewm(span=self.cfg.MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_val = float(macd_line.iloc[-1])
        sig_val = float(signal_line.iloc[-1])
        hist_val = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2]) if len(histogram) >= 2 else hist_val

        if hist_val > 0 and hist_val > prev_hist:
            direction = "bullish_rising"
        elif hist_val > 0:
            direction = "bullish_flat"
        elif hist_val < 0 and hist_val < prev_hist:
            direction = "bearish_falling"
        else:
            direction = "bearish_flat"

        return macd_val, sig_val, direction

    def _macd_score(self, direction: str) -> float:
        """Convert MACD direction to a bullish score."""
        return {
            "bullish_rising": 1.0,
            "bullish_flat":   0.65,
            "bearish_flat":   0.35,
            "bearish_falling": 0.0,
        }.get(direction, 0.5)

    def _momentum(self, close: pd.Series) -> float:
        """
        Compute 20-day price momentum as a percentage change.
        Returns the raw percentage (e.g. 0.12 = +12%).
        """
        if len(close) < self.cfg.MOMENTUM_PERIOD + 1:
            return 0.0
        base = float(close.iloc[-(self.cfg.MOMENTUM_PERIOD + 1)])
        current = float(close.iloc[-1])
        return (current - base) / base if base != 0 else 0.0

    def _momentum_score(self, momentums: dict[str, float], symbol: str) -> float:
        """
        Normalize momentum across all universe stocks.
        Rank within the peer group → score [0, 1].
        """
        values = list(momentums.values())
        if not values:
            return 0.5
        sorted_vals = sorted(values)
        my_val = momentums.get(symbol, 0.0)
        rank = sorted_vals.index(my_val) if my_val in sorted_vals else len(sorted_vals) // 2
        return rank / max(len(sorted_vals) - 1, 1)

    def _volume_breakout(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Compute volume ratio (today vs 20d avg) and a price direction confirmation.

        Returns: (volume_ratio, price_change_pct_today)
        """
        if len(df) < self.cfg.VOLUME_AVG_PERIOD + 1:
            return 1.0, 0.0

        avg_volume = float(df["Volume"].iloc[-self.cfg.VOLUME_AVG_PERIOD - 1 : -1].mean())
        today_volume = float(df["Volume"].iloc[-1])
        volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0

        today_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        price_chg = (today_close - prev_close) / prev_close if prev_close > 0 else 0.0

        return volume_ratio, price_chg

    def _volume_score(self, volume_ratio: float, price_chg: float) -> float:
        """
        Score volume breakout signal.
        High volume + positive price → bullish. High volume + negative → bearish.
        """
        if volume_ratio >= 1.5:
            # Big volume day — direction matters a lot
            if price_chg > 0.01:
                return min(1.0, 0.7 + price_chg * 5)   # breakout up
            elif price_chg < -0.01:
                return max(0.0, 0.3 + price_chg * 5)   # breakout down
            else:
                return 0.5   # big volume, flat price
        else:
            # Normal volume — slight tilt based on price direction
            return 0.5 + min(0.15, price_chg * 3)

    # ── Scoring ───────────────────────────────────────────────────────────

    def score_universe(self, tickers: list[str]) -> dict[str, dict]:
        """
        Score all tickers and return a dict of signal data.

        Returns:
            {
              "AAPL": {
                "score": 0.72,
                "signal": "Buy",
                "rsi": 34.5,
                "macd_direction": "bullish_rising",
                "momentum_pct": 0.08,
                "volume_ratio": 1.6,
                "reasoning": "Score 0.72: RSI oversold(34.5) + MACD bullish_rising + ..."
              },
              ...
            }
        """
        ohlcv_data = self._fetch_ohlcv(tickers)
        if not ohlcv_data:
            logger.error("No OHLCV data returned — cannot score universe.")
            return {}

        # First pass: gather raw momentums for normalization
        raw_momentums: dict[str, float] = {}
        for sym, df in ohlcv_data.items():
            raw_momentums[sym] = self._momentum(df["Close"])

        # Second pass: compute all indicator scores
        scores: dict[str, dict] = {}
        for sym, df in ohlcv_data.items():
            try:
                rsi_val = self._rsi(df["Close"], self.cfg.RSI_PERIOD)
                macd_val, sig_val, macd_dir = self._macd(df["Close"])
                mom_pct = raw_momentums[sym]
                vol_ratio, price_chg = self._volume_breakout(df)

                rsi_s  = self._rsi_score(rsi_val)
                macd_s = self._macd_score(macd_dir)
                mom_s  = self._momentum_score(raw_momentums, sym)
                vol_s  = self._volume_score(vol_ratio, price_chg)

                composite = (
                    self.cfg.RSI_WEIGHT      * rsi_s
                    + self.cfg.MACD_WEIGHT   * macd_s
                    + self.cfg.MOMENTUM_WEIGHT * mom_s
                    + self.cfg.VOLUME_WEIGHT * vol_s
                )
                composite = round(float(np.clip(composite, 0.0, 1.0)), 4)

                if composite >= self.cfg.MIN_SCORE_TO_BUY:
                    signal = "Buy"
                elif composite <= self.cfg.SELL_SCORE_THRESHOLD:
                    signal = "Sell"
                else:
                    signal = "Hold"

                reasoning = (
                    f"Score {composite:.2f}: "
                    f"RSI={rsi_val:.1f}(score={rsi_s:.2f}) | "
                    f"MACD={macd_dir}(score={macd_s:.2f}) | "
                    f"Momentum={mom_pct:+.1%}(score={mom_s:.2f}) | "
                    f"VolRatio={vol_ratio:.2f}(score={vol_s:.2f})"
                )

                scores[sym] = {
                    "score": composite,
                    "signal": signal,
                    "rsi": round(rsi_val, 2),
                    "macd_direction": macd_dir,
                    "momentum_pct": round(mom_pct, 4),
                    "volume_ratio": round(vol_ratio, 2),
                    "price_change_today": round(price_chg, 4),
                    "reasoning": reasoning,
                    # Individual sub-scores for logging
                    "rsi_score": round(rsi_s, 3),
                    "macd_score": round(macd_s, 3),
                    "momentum_score": round(mom_s, 3),
                    "volume_score": round(vol_s, 3),
                }

            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", sym, exc)

        logger.info(
            "Scored %d stocks | Buy: %d | Hold: %d | Sell: %d",
            len(scores),
            sum(1 for v in scores.values() if v["signal"] == "Buy"),
            sum(1 for v in scores.values() if v["signal"] == "Hold"),
            sum(1 for v in scores.values() if v["signal"] == "Sell"),
        )
        return scores

    def rank_buys(self, scores: dict[str, dict]) -> list[str]:
        """
        Return tickers with Buy signal, sorted by composite score descending.
        """
        buys = [
            (sym, data["score"])
            for sym, data in scores.items()
            if data["signal"] == "Buy"
        ]
        buys.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in buys]

    def should_sell(self, ticker: str, scores: dict[str, dict]) -> bool:
        """
        Return True if signal has degraded to Sell for a currently held ticker.
        Returns False if ticker is not in scores (data unavailable — hold).
        """
        if ticker not in scores:
            logger.warning("%s not in scores — holding position (no data).", ticker)
            return False
        return scores[ticker]["signal"] == "Sell"
