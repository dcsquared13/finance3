"""
lib/learner.py — LinearSignalLearner

The model is a linear combination of five indicators:

    score = w0*rsi + w1*macd + w2*momentum + w3*volume + w4*sentiment + bias

Weights start at the hand-tuned values below and drift toward whatever
actually predicted returns in practice.

Learning algorithm: stochastic gradient descent on mean-squared error.
  loss      = (predicted_score - actual_return_normalized)²
  w_new     = w_old - lr * d_loss/d_w
  d_loss/dw = 2 * (pred - actual) * feature_value

After each live session the agent:
  1. Looks up how its picks from HOLD_PERIOD days ago actually performed
  2. Calls learner.update(features, actual_return) for each resolved pick
  3. Saves the new weights to data/learned_weights.json

The weights are normalized to sum to 1.0 after each update so they stay
interpretable as indicator "importance" percentages.

Sentiment feature
-----------------
Sourced from lib/sentiment.py (FinBERT / VADER).  During backtesting where
historical news is unavailable, pass sentiment=0.5 (neutral) in the feature
dict.  The learner will gradually discover the true importance of sentiment
once live updates start flowing.
"""

import json
import os
import logging
import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'learned_weights.json')

# Feature order is fixed — matches the feature vector built in strategy_ml.py
FEATURE_NAMES = ['rsi', 'macd', 'momentum', 'volume', 'sentiment']

# Starting weights: original four scaled to sum=0.90, sentiment seeded at 0.10
DEFAULT_WEIGHTS = {
    'rsi':       0.225,   # was 0.25  → scaled by 0.90
    'macd':      0.270,   # was 0.30  → scaled by 0.90
    'momentum':  0.225,   # was 0.25  → scaled by 0.90
    'volume':    0.180,   # was 0.20  → scaled by 0.90
    'sentiment': 0.100,   # new feature, seeded at 0.10
    'bias':      0.0,
}

log = logging.getLogger(__name__)


class LinearSignalLearner:
    """
    Online-learning linear model for indicator weighting.

    Usage:
        learner = LinearSignalLearner()

        # Score a stock (replaces fixed-weight scoring in strategy.py)
        score = learner.score({
            'rsi': 0.8, 'macd': 0.6,
            'momentum': 0.5, 'volume': 0.4,
            'sentiment': 0.7,
        })

        # After HOLD_PERIOD days, update with what actually happened
        learner.update(
            features={
                'rsi': 0.8, 'macd': 0.6,
                'momentum': 0.5, 'volume': 0.4,
                'sentiment': 0.7,
            },
            actual_return=0.032   # +3.2% over the hold period
        )
    """

    def __init__(self, learning_rate: float = 0.01, normalize: bool = True):
        self.lr = learning_rate
        self.normalize = normalize
        self.weights = self._load_weights()
        self.update_count = 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, features: dict) -> float:
        """
        Return a composite score in [0, 1] given a feature dict.

        Each feature should be normalized to [0, 1] by the caller.
        Missing features default to 0.5 (neutral) rather than 0.0 to
        avoid penalising stocks when a data source is temporarily unavailable.
        """
        w = self.weights
        raw = (
            w['rsi']       * features.get('rsi',       0.5) +
            w['macd']      * features.get('macd',      0.5) +
            w['momentum']  * features.get('momentum',  0.5) +
            w['volume']    * features.get('volume',    0.5) +
            w['sentiment'] * features.get('sentiment', 0.5) +
            w['bias']
        )
        # Clip to [0, 1] — keeps thresholds (BUY_THRESHOLD, SELL_THRESHOLD)
        # interpretable even after learning shifts weights around.
        return float(np.clip(raw, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, features: dict, actual_return: float) -> dict:
        """
        One SGD step.  actual_return is the realised % return (e.g. 0.032 = +3.2%).

        The target is the normalised return: we map actual_return to [0,1] using
        a soft sigmoid so large gains/losses don't dominate the gradient.
        """
        target    = self._normalize_return(actual_return)
        predicted = self.score(features)
        error     = predicted - target      # positive → overconfident buy signal

        # Gradient step for each weight
        grad = {}
        for feat in FEATURE_NAMES:
            feat_val   = features.get(feat, 0.5)
            grad[feat] = 2 * error * feat_val
            self.weights[feat] -= self.lr * grad[feat]

        # Bias term (no feature to multiply)
        self.weights['bias'] -= self.lr * 2 * error

        # Keep indicator weights ≥ 0 (negative weights would invert an
        # indicator's meaning, which we want to prevent)
        for feat in FEATURE_NAMES:
            self.weights[feat] = max(0.0, self.weights[feat])

        if self.normalize:
            self._normalize_weights()

        self.update_count += 1
        self._save_weights()

        log.info(
            f"Learner update #{self.update_count}: "
            f"pred={predicted:.3f} target={target:.3f} error={error:.3f} | "
            f"weights={self._weights_summary()}"
        )
        return {
            'predicted': predicted,
            'target':    target,
            'error':     error,
            'weights':   dict(self.weights),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_return(self, ret: float) -> float:
        """
        Map a raw % return to [0, 1].
        Uses a sigmoid centred at 0 so that:
          +10% → ~0.73   (good buy signal, score should be high)
            0% → 0.50   (neutral)
          -10% → ~0.27   (bad buy signal, score should be low)
        The scaling factor (10) controls sensitivity.
        """
        return float(1.0 / (1.0 + np.exp(-ret * 10)))

    def _normalize_weights(self):
        """Rescale positive indicator weights to sum to 1.0.  Bias is untouched."""
        total = sum(self.weights[f] for f in FEATURE_NAMES)
        if total > 0:
            for feat in FEATURE_NAMES:
                self.weights[feat] /= total

    def _weights_summary(self) -> str:
        return ' '.join(f"{f}={self.weights[f]:.3f}" for f in FEATURE_NAMES)

    def _load_weights(self) -> dict:
        if os.path.exists(WEIGHTS_PATH):
            try:
                with open(WEIGHTS_PATH) as f:
                    data = json.load(f)
                # Migrate old 4-feature weights files by injecting a
                # neutral sentiment weight
                if 'sentiment' not in data:
                    log.info(
                        "Migrating learned_weights.json: adding 'sentiment' "
                        "feature at default weight 0.10"
                    )
                    total = sum(data.get(f, 0.0) for f in
                                ['rsi', 'macd', 'momentum', 'volume'])
                    if total > 0:
                        scale = 0.90 / total
                        for f in ['rsi', 'macd', 'momentum', 'volume']:
                            data[f] = data.get(f, 0.0) * scale
                    data['sentiment'] = 0.10
                log.info(f"Loaded learned weights from {WEIGHTS_PATH}")
                return data
            except Exception as exc:
                log.warning(f"Could not load weights ({exc}), using defaults")
        return dict(DEFAULT_WEIGHTS)

    def _save_weights(self):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        with open(WEIGHTS_PATH, 'w') as f:
            json.dump(self.weights, f, indent=2)

    def summary(self) -> dict:
        return {
            'update_count': self.update_count,
            'weights':      dict(self.weights),
            'weights_file': WEIGHTS_PATH,
        }
