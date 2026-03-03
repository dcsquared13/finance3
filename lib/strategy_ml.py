"""
lib/strategy_ml.py — Drop-in replacement for strategy.py that uses learned weights.

Differences from strategy.py:
  - SignalEngine accepts an optional `learner` (LinearSignalLearner).
    If provided, composite scores use learned weights instead of config constants.
  - Features are computed the same way as before, but each indicator's 0/1 score
    is stored in a feature dict before being combined — so the learner can
    consume them directly.
  - score_universe() returns (symbol, score, features_dict) triples so the
    trading agent can log features for later learning.

Everything else (get_universe, rank_buys, should_sell) is identical to
strategy.py, so this file is a drop-in replacement.
"""

import logging
import numpy as np
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    LOOKBACK_DAYS, UNIVERSE_SIZE,
    MIN_SCORE_TO_BUY, SELL_SCORE_THRESHOLD,
)

log = logging.getLogger(__name__)

# Static fallback universe (same as strategy.py)
FALLBACK_UNIVERSE = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','BRK-B','TSLA','JPM','V',
    'UNH','XOM','LLY','JNJ','WMT','MA','PG','HD','MRK','COST',
    'ABBV','CVX','BAC','NFLX','AMD','KO','PEP','TMO','ADBE','CRM',
    'ACN','MCD','LIN','ABT','CSCO','DHR','TXN','ORCL','WFC','NKE',
    'INTC','PM','AMGN','RTX','IBM','HON','LOW','UNP','SPGI','GS',
]


class SignalEngine:
    """
    Multi-indicator signal engine with optional ML-learned weights.

    Parameters
    ----------
    learner : LinearSignalLearner | None
        If supplied, composite score = learner.score(features_dict).
        If None, falls back to the static config weights (original behaviour).
    """

    def __init__(self, learner=None):
        self.learner = learner

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self) -> list[str]:
        """Return top N S&P 500 tickers. Falls back to static list."""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            tickers = [row.find('td').text.strip().replace('.', '-')
                       for row in table.find_all('tr')[1:]]
            return tickers[:UNIVERSE_SIZE]
        except Exception:
            log.warning("Could not fetch S&P 500 list, using fallback universe")
            return FALLBACK_UNIVERSE[:UNIVERSE_SIZE]

    # ------------------------------------------------------------------
    # Indicators → feature dict (all normalised to [0, 1])
    # ------------------------------------------------------------------

    def _compute_features(self, hist: pd.DataFrame) -> dict | None:
        """
        Given a DataFrame with Close + Volume columns, return a feature dict.
        Returns None if history is too short.
        """
        close = hist['Close'].dropna()
        volume = hist['Volume'].dropna() if 'Volume' in hist.columns else None

        if len(close) < 30:
            return None

        # RSI
        rsi_score = self._rsi_score(close)

        # MACD
        macd_score = self._macd_score(close)

        # 20-day momentum
        momentum_score = self._momentum_score(close)

        # Volume breakout
        volume_score = self._volume_score(volume, close) if volume is not None else 0.5

        return {
            'rsi':      rsi_score,
            'macd':     macd_score,
            'momentum': momentum_score,
            'volume':   volume_score,
        }

    def _rsi_score(self, close: pd.Series) -> float:
        """RSI(14). Score=1 → oversold, Score=0 → overbought."""
        delta = close.diff().dropna()
        gain  = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss  = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
        if loss == 0:
            rsi = 100.0
        else:
            rsi = 100 - 100 / (1 + gain / loss)
        return float(np.clip((70 - rsi) / 40, 0.0, 1.0))

    def _macd_score(self, close: pd.Series) -> float:
        """MACD(12,26,9). Bullish crossover + rising histogram → 1.0."""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig
        cross   = 1.0 if macd.iloc[-1] > sig.iloc[-1] else 0.0
        rising  = 1.0 if (len(hist) >= 2 and hist.iloc[-1] > hist.iloc[-2]) else 0.0
        return float(0.6 * cross + 0.4 * rising)

    def _momentum_score(self, close: pd.Series) -> float:
        """20-day price change, normalised to [0,1] with ±20% as extremes."""
        if len(close) < 21:
            return 0.5
        pct = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
        return float(np.clip((pct + 0.20) / 0.40, 0.0, 1.0))

    def _volume_score(self, volume: pd.Series, close: pd.Series) -> float:
        """Volume vs 20-day avg; penalise high volume on down days."""
        if len(volume) < 21:
            return 0.5
        ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
        score = float(np.clip((ratio - 0.5) / 2.0, 0.0, 1.0))
        # Penalise if price is falling
        if len(close) >= 2 and close.iloc[-1] < close.iloc[-2]:
            score *= 0.3
        return score

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def _composite_score(self, features: dict) -> float:
        """
        If a learner is available, use its weights.
        Otherwise fall back to static config weights.
        """
        if self.learner is not None:
            return self.learner.score(features)

        # Static fallback (mirrors original strategy.py weights)
        return (
            0.25 * features['rsi'] +
            0.30 * features['macd'] +
            0.25 * features['momentum'] +
            0.20 * features['volume']
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_universe(self, universe: list[str]) -> list[tuple]:
        """
        Fetch data for each symbol and score it.

        Returns
        -------
        List of (symbol, score, features_dict, current_price) sorted by score desc.
        features_dict is included so the trading agent can log it for later learning.
        """
        results = []

        prices = yf.download(
            universe,
            period=f'{LOOKBACK_DAYS}d',
            auto_adjust=True,
            progress=False,
        )

        for symbol in universe:
            try:
                if isinstance(prices.columns, pd.MultiIndex):
                    hist = prices.xs(symbol, axis=1, level=1).dropna()
                else:
                    hist = prices.rename(columns={'Close': 'Close', 'Volume': 'Volume'})

                features = self._compute_features(hist)
                if features is None:
                    continue

                score = self._composite_score(features)
                current_price = float(hist['Close'].iloc[-1])

                results.append((symbol, score, features, current_price))

            except Exception as e:
                log.debug(f"Skipping {symbol}: {e}")
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def rank_buys(self, scored: list[tuple], current_holdings: list[str]) -> list[tuple]:
        """Return scored stocks that pass MIN_SCORE_TO_BUY and aren't already held."""
        return [
            (sym, score, feats, price)
            for sym, score, feats, price in scored
            if score >= MIN_SCORE_TO_BUY and sym not in current_holdings
        ]

    def should_sell(self, symbol: str, scored: list[tuple]) -> tuple[bool, float | None]:
        """
        Return (True, score) if the symbol's signal has degraded below SELL_SCORE_THRESHOLD.
        Returns (False, score) if the position should be held.
        """
        for sym, score, _feats, _price in scored:
            if sym == symbol:
                return (score <= SELL_SCORE_THRESHOLD, score)
        return (False, None)   # symbol not in universe — hold
