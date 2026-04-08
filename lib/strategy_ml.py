"""
lib/strategy_ml.py — Drop-in replacement for strategy.py that uses learned weights.

Differences from strategy.py:
  - SignalEngine accepts an optional `learner` (LinearSignalLearner).
    If provided, composite scores use learned weights instead of config constants.
  - Features are computed the same way as before, but each indicator's 0/1 score
    is stored in a feature dict before being combined — so the learner can
    consume them directly.
  - score_universe() accepts an optional `sentiment_scores` dict (keyed by
    ticker symbol) produced by lib/sentiment.analyze_sentiment().  When
    provided, each ticker's sentiment composite is added to its feature dict
    as 'sentiment' (already normalised to [0, 1]).  When absent the feature
    defaults to 0.5 (neutral) so no stock is penalised for missing news data.
  - score_universe() returns (symbol, score, features_dict, current_price)
    4-tuples so the trading agent can log features for later learning.

Everything else (get_universe, rank_buys, should_sell) is identical to
strategy.py, so this file is a drop-in replacement.
"""

import logging
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from lib.yahoo_direct import download as yahoo_download

from config import Config

_cfg = Config()

log = logging.getLogger(__name__)

# Static fallback universe (same as strategy.py)
FALLBACK_UNIVERSE = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','JPM','V','C',
    'UNH','XOM','LLY','JNJ','WMT','MA','PG','HD','MRK','COST',
    'ABBV','CVX','BAC','NFLX','AMD','KO','PEP','TMO','ADBE','CRM',
    'ACN','MCD','LIN','ABT','CSCO','DHR','TXN','ORCL','WFC','NKE',
    'INTC','PM','AMGN','RTX','IBM','HON','LOW','UNP','SPGI','GS',
]


class SignalEngine:
    """
    Multi-indicator signal engine with optional ML-learned weights and
    optional AI sentiment scoring.

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
        """Return top N S&P 500 tickers, excluding Alpaca-unsupported assets.
        Falls back to static list on network error."""
        excluded = getattr(_cfg, 'EXCLUDED_TICKERS', frozenset())
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            tickers = [row.find('td').text.strip().replace('.', '-')
                       for row in table.find_all('tr')[1:]]
            filtered = [t for t in tickers if t not in excluded]
            return filtered[:_cfg.UNIVERSE_SIZE]
        except Exception:
            log.warning("Could not fetch S&P 500 list, using fallback universe")
            return [t for t in FALLBACK_UNIVERSE if t not in excluded][:_cfg.UNIVERSE_SIZE]

    # ------------------------------------------------------------------
    # Indicators → feature dict (all normalised to [0, 1])
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        hist: pd.DataFrame,
        sentiment_score: float = 0.5,
    ) -> dict | None:
        """
        Given a price/volume DataFrame, return a feature dict.
        Returns None if history is too short.

        sentiment_score : pre-computed [0, 1] from lib/sentiment.py.
                          Defaults to 0.5 (neutral) when unavailable.
        """
        close  = hist['Close'].dropna()
        volume = hist['Volume'].dropna() if 'Volume' in hist.columns else None

        if len(close) < 30:
            return None

        rsi_score      = self._rsi_score(close)
        macd_score     = self._macd_score(close)
        momentum_score = self._momentum_score(close)
        volume_score   = (
            self._volume_score(volume, close) if volume is not None else 0.5
        )

        return {
            'rsi':       rsi_score,
            'macd':      macd_score,
            'momentum':  momentum_score,
            'volume':    volume_score,
            'sentiment': float(np.clip(sentiment_score, 0.0, 1.0)),
        }

    def _rsi_score(self, close: pd.Series) -> float:
        """RSI(14).  Score=1 → oversold (buy signal), Score=0 → overbought."""
        delta = close.diff().dropna()
        gain  = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss  = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
        rsi   = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
        return float(np.clip((70 - rsi) / 40, 0.0, 1.0))

    def _macd_score(self, close: pd.Series) -> float:
        """MACD(12,26,9).  Bullish crossover + rising histogram → 1.0."""
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        sig    = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - sig
        cross  = 1.0 if macd.iloc[-1] > sig.iloc[-1] else 0.0
        rising = 1.0 if (len(hist) >= 2 and hist.iloc[-1] > hist.iloc[-2]) else 0.0
        return float(0.6 * cross + 0.4 * rising)

    def _momentum_score(self, close: pd.Series) -> float:
        """20-day price change, normalised to [0,1] with ±20 % as extremes."""
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
        if len(close) >= 2 and close.iloc[-1] < close.iloc[-2]:
            score *= 0.3
        return score

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def _composite_score(self, features: dict) -> float:
        """
        If a learner is available, use its learned weights (includes sentiment).
        Otherwise fall back to static weights — sentiment is included at a
        fixed 0.10 weight with the original four scaled to sum to 0.90.
        """
        if self.learner is not None:
            return self.learner.score(features)

        # Static fallback (mirrors DEFAULT_WEIGHTS in learner.py)
        return float(np.clip(
            0.225 * features.get('rsi',       0.5) +
            0.270 * features.get('macd',      0.5) +
            0.225 * features.get('momentum',  0.5) +
            0.180 * features.get('volume',    0.5) +
            0.100 * features.get('sentiment', 0.5),
            0.0, 1.0,
        ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_universe(
        self,
        universe: list[str],
        sentiment_scores: dict | None = None,
    ) -> list[tuple]:
        """
        Fetch price data for each symbol, attach sentiment if available,
        and return scored results.

        Parameters
        ----------
        universe         : list of ticker symbols
        sentiment_scores : optional dict returned by
                           lib/sentiment.analyze_sentiment().
                           Keys are ticker symbols; values contain at least
                           {'composite': float}.  If None or a ticker is
                           missing, sentiment defaults to 0.5.

        Returns
        -------
        List of (symbol, score, features_dict, current_price) sorted by
        score descending.  features_dict always contains all five keys:
        rsi, macd, momentum, volume, sentiment.
        """
        results = []
        sentiment_scores = sentiment_scores or {}

        prices = yahoo_download(
            universe,
            period=f'{_cfg.LOOKBACK_DAYS}d',
        )

        for symbol in universe:
            try:
                if isinstance(prices.columns, pd.MultiIndex):
                    hist = prices.xs(symbol, axis=1, level=1).dropna()
                else:
                    hist = prices.rename(
                        columns={'Close': 'Close', 'Volume': 'Volume'}
                    )

                # Pull sentiment composite; default neutral (0.5)
                sent_result = sentiment_scores.get(symbol)
                sent_value  = (
                    float(sent_result.get('composite', 0.5))
                    if isinstance(sent_result, dict) else 0.5
                )

                features = self._compute_features(hist, sentiment_score=sent_value)
                if features is None:
                    continue

                score         = self._composite_score(features)
                current_price = float(hist['Close'].iloc[-1])

                results.append((symbol, score, features, current_price))

            except Exception as exc:
                log.debug(f"Skipping {symbol}: {exc}")
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def rank_buys(
        self,
        scored: list[tuple],
        current_holdings: list[str],
    ) -> list[tuple]:
        """Return top-scoring stocks not already held, using an adaptive threshold.

        Effective threshold = max(MIN_SCORE_TO_BUY, universe median) so buys
        still happen even when all scores compress around 0.5 after training.
        """
        import numpy as np
        candidates = [
            (sym, score, feats, price)
            for sym, score, feats, price in scored
            if sym not in current_holdings
        ]
        if not candidates:
            return []
        median_score = float(np.median([s for _, s, _, _ in candidates]))
        threshold = max(_cfg.MIN_SCORE_TO_BUY, median_score)
        log.debug(f"rank_buys: median={median_score:.4f}  threshold={threshold:.4f}")
        return [
            (sym, score, feats, price)
            for sym, score, feats, price in candidates
            if score >= threshold
        ]

    def should_sell(
        self,
        symbol: str,
        scored: list[tuple],
    ) -> tuple[bool, float | None]:
        """
        Return (True, score) if the symbol's signal has degraded enough to sell.

        Uses an adaptive threshold that mirrors rank_buys():
          sell_threshold = max(SELL_SCORE_THRESHOLD, median_score - gap)
        where gap = MIN_SCORE_TO_BUY - SELL_SCORE_THRESHOLD.

        This prevents score compression from silencing sell signals — if all
        scores cluster around 0.5, the adaptive threshold rises with the median
        so underperformers are still exited.

        If the symbol is missing from scored (data failure), default to sell
        so we don't hold positions we can't evaluate.
        """
        entry = next(
            ((score) for sym, score, _, _ in scored if sym == symbol),
            None,
        )
        if entry is None:
            log.warning(
                f"should_sell: {symbol} not in scored universe — defaulting to sell"
            )
            return (True, None)

        score = entry
        all_scores = [s for _, s, _, _ in scored]
        median_score = float(np.median(all_scores))
        gap = _cfg.MIN_SCORE_TO_BUY - _cfg.SELL_SCORE_THRESHOLD  # e.g. 0.105
        adaptive_threshold = max(_cfg.SELL_SCORE_THRESHOLD, median_score - gap)

        log.debug(
            f"should_sell {symbol}: score={score:.4f}  "
            f"median={median_score:.4f}  adaptive_threshold={adaptive_threshold:.4f}"
        )
        return (score <= adaptive_threshold, score)
