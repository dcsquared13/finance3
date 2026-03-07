"""
backtest.py — Historical training loop for the ML signal learner

HOW IT WORKS
============
1. Fetch N years of daily OHLCV data for the stock universe via Yahoo Finance v8 API.
2. Walk forward day by day (no lookahead — on day T we only use data up to day T).
3. On each day:
   a. Compute indicator features for every stock.
   b. Score all stocks with the current learner weights.
   c. Simulate "buying" the top-K scorers.
4. After HOLD_PERIOD trading days, look up the actual return for each simulated buy.
5. Call learner.update(features, actual_return) → one gradient step per resolved pick.
6. Record the evolving weights and portfolio equity curve.
7. Print a summary and (optionally) plot the learning curve.

Sentiment note
--------------
Historical news data is not available via yfinance, so the 'sentiment' feature
is set to 0.5 (neutral) for all historical picks.  The learner will start
weighting sentiment more accurately once live trading begins and real sentiment
values flow in via trading_agent_ml.py.

Usage:
    python backtest.py                        # default: 2 years, hold=5 days
    python backtest.py --years 3 --hold 10   # longer backtest, 10-day holds
    python backtest.py --plot                 # show matplotlib chart at end
    python backtest.py --reset               # wipe learned_weights.json first

The learned weights produced here are saved to data/learned_weights.json and
will be picked up automatically by trading_agent_ml.py on the next live run.
"""

import argparse
import os
import sys
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))
from lib.learner import LinearSignalLearner, WEIGHTS_PATH
from lib.yahoo_direct import download as yahoo_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe — same static fallback as strategy.py
# ---------------------------------------------------------------------------
UNIVERSE = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','BRK-B','TSLA','JPM','V',
    'UNH','XOM','LLY','JNJ','WMT','MA','PG','HD','MRK','COST',
    'ABBV','CVX','BAC','NFLX','AMD','KO','PEP','TMO','ADBE','CRM',
    'ACN','MCD','LIN','ABT','CSCO','DHR','TXN','ORCL','WFC','NKE',
    'INTC','PM','AMGN','RTX','IBM','HON','LOW','UNP','SPGI','GS',
]

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(prices: pd.DataFrame, symbol: str, today_idx: int) -> dict | None:
    """
    Compute the five normalised indicator scores for `symbol` as of `today_idx`.
    Returns None if there is insufficient history.
    All features are in [0, 1].

    'sentiment' is set to 0.5 (neutral) because historical news is not available.
    The learner will calibrate the sentiment weight once live data flows in.
    """
    try:
        col = (symbol, 'Close') if isinstance(prices.columns, pd.MultiIndex) else symbol
        close = prices[col].dropna()

        if len(close) < 30:
            return None

        # Slice history up to today (no lookahead)
        hist = close.iloc[:today_idx + 1]
        if len(hist) < 30:
            return None

        # --- RSI (14-day) ---
        delta    = hist.diff().dropna()
        gains    = delta.clip(lower=0)
        losses   = (-delta).clip(lower=0)
        avg_gain = gains.rolling(14).mean().iloc[-1]
        avg_loss = losses.rolling(14).mean().iloc[-1]
        rsi      = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
        # Score: oversold (RSI<30) → 1.0, overbought (RSI>70) → 0.0
        rsi_score = float(np.clip((70 - rsi) / 40, 0.0, 1.0))

        # --- MACD (12/26/9) ---
        ema12       = hist.ewm(span=12, adjust=False).mean()
        ema26       = hist.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram   = macd_line - signal_line
        macd_cross  = 1.0 if (macd_line.iloc[-1] > signal_line.iloc[-1]) else 0.0
        hist_rising = 1.0 if (
            len(histogram) >= 2 and histogram.iloc[-1] > histogram.iloc[-2]
        ) else 0.0
        macd_score = 0.6 * macd_cross + 0.4 * hist_rising

        # --- 20-day momentum ---
        if len(hist) >= 21:
            momentum_pct = (hist.iloc[-1] - hist.iloc[-21]) / hist.iloc[-21]
        else:
            momentum_pct = 0.0
        momentum_score = float(np.clip((momentum_pct + 0.20) / 0.40, 0.0, 1.0))

        # --- Volume breakout ---
        try:
            vcol = (
                (symbol, 'Volume')
                if isinstance(prices.columns, pd.MultiIndex)
                else symbol + '_Volume'
            )
            vol = prices[vcol].dropna().iloc[:today_idx + 1]
            if len(vol) >= 21:
                vol_ratio    = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
                volume_score = float(np.clip((vol_ratio - 0.5) / 2.0, 0.0, 1.0))
                if momentum_pct < 0:
                    volume_score *= 0.3   # penalise high volume on down days
            else:
                volume_score = 0.5
        except Exception:
            volume_score = 0.5

        return {
            'rsi':       rsi_score,
            'macd':      macd_score,
            'momentum':  momentum_score,
            'volume':    volume_score,
            'sentiment': 0.5,   # neutral — no historical news available
        }
    except Exception as exc:
        log.debug(f"Feature error {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    years: int = 2,
    hold_period: int = 5,
    top_k: int = 5,
    learning_rate: float = 0.01,
    plot: bool = False,
    reset: bool = False,
):
    if reset and os.path.exists(WEIGHTS_PATH):
        os.remove(WEIGHTS_PATH)
        log.info("Cleared learned_weights.json — starting from defaults")

    learner = LinearSignalLearner(learning_rate=learning_rate)
    log.info(f"Starting weights: {learner.weights}")

    # --- Fetch data ---
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=years * 365 + 60)   # extra buffer for indicators
    log.info(
        f"Fetching {len(UNIVERSE)} tickers from "
        f"{start_date.date()} to {end_date.date()}..."
    )

    raw = yahoo_download(
        UNIVERSE,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
    )
    log.info(f"Downloaded price data: {raw.shape}")

    if isinstance(raw.columns, pd.MultiIndex):
        close_matrix = raw['Close']
    else:
        close_matrix = raw[['Close']]

    close_matrix = close_matrix.dropna(how='all')
    dates  = close_matrix.index.tolist()
    n_days = len(dates)

    log.info(f"Trading days available: {n_days}")

    # --- State ---
    # pending_picks[settle_idx] = list of {symbol, features, entry_price}
    pending_picks  = defaultdict(list)
    equity_curve   = []   # (date, portfolio_value)
    weight_history = []   # (date, weights_dict)
    trade_outcomes = []   # (symbol, date_bought, return_pct, error)

    portfolio_value = 10_000.0
    update_count    = 0

    # Skip first 60 days (need history for indicators)
    warmup = 60

    for i in range(warmup, n_days):
        today = dates[i]

        # 1. Resolve any picks that matured today
        if i in pending_picks:
            for pick in pending_picks[i]:
                symbol      = pick['symbol']
                entry_price = pick['entry_price']
                try:
                    exit_price    = float(close_matrix[symbol].iloc[i])
                    actual_return = (exit_price - entry_price) / entry_price
                except Exception:
                    actual_return = 0.0

                result       = learner.update(pick['features'], actual_return)
                update_count += 1

                trade_outcomes.append({
                    'symbol':        symbol,
                    'date_bought':   pick['date_bought'].date(),
                    'actual_return': round(actual_return, 4),
                    'predicted':     round(result['predicted'], 4),
                    'error':         round(result['error'], 4),
                })

                # Simple P&L tracking
                position_size    = portfolio_value / top_k
                portfolio_value += position_size * actual_return

        # 2. Score today's universe and pick top-K
        scores         = {}
        features_cache = {}
        for symbol in UNIVERSE:
            try:
                feats = compute_features(raw, symbol, i)
                if feats is None:
                    continue
                score            = learner.score(feats)
                scores[symbol]   = score
                features_cache[symbol] = feats
            except Exception:
                continue

        if not scores:
            continue

        ranked    = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_picks = ranked[:top_k]

        # 3. Record picks for settlement after hold_period
        settle_idx = min(i + hold_period, n_days - 1)
        for symbol, score in top_picks:
            try:
                entry_price = float(close_matrix[symbol].iloc[i])
            except Exception:
                continue
            pending_picks[settle_idx].append({
                'symbol':      symbol,
                'date_bought': today,
                'entry_price': entry_price,
                'features':    features_cache[symbol],
                'score':       score,
            })

        equity_curve.append((today, portfolio_value))
        weight_history.append((today, dict(learner.weights)))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    print(f"  Period:           {dates[warmup].date()} → {dates[-1].date()}")
    print(f"  Trading days:     {n_days - warmup}")
    print(f"  Gradient updates: {update_count}")
    print(
        f"  Final portfolio:  ${portfolio_value:,.2f}  "
        f"({'+'if portfolio_value > 10000 else ''}"
        f"{(portfolio_value/10000-1)*100:.1f}%)"
    )
    print(f"\n  Final learned weights:")
    for feat, w in learner.weights.items():
        arrow = '↑' if w > 0.25 else ('↓' if w < 0.10 else ' ')
        print(f"    {feat:12s}: {w:.4f}  {arrow}")
    print(
        f"\n  Note: 'sentiment' weight was trained on neutral (0.5) values.\n"
        f"        It will become meaningful after live trading sessions."
    )

    if trade_outcomes:
        returns = [t['actual_return'] for t in trade_outcomes]
        wins    = sum(1 for r in returns if r > 0)
        print(f"\n  Resolved trades:  {len(returns)}")
        print(f"  Win rate:         {wins/len(returns)*100:.1f}%")
        print(f"  Avg return/hold:  {np.mean(returns)*100:.2f}%")
        print(
            f"  Avg |error|:      "
            f"{np.mean([abs(t['error']) for t in trade_outcomes]):.4f}"
        )

    print(f"\n  Weights saved → {WEIGHTS_PATH}")
    print("=" * 60)

    # --- Save outcomes CSV ---
    outcomes_path = os.path.join(os.path.dirname(WEIGHTS_PATH), 'backtest_outcomes.csv')
    pd.DataFrame(trade_outcomes).to_csv(outcomes_path, index=False)
    print(f"  Trade outcomes  → {outcomes_path}")

    # --- Plot ---
    if plot:
        _plot_results(equity_curve, weight_history)

    return learner


def _plot_results(equity_curve, weight_history):
    try:
        import matplotlib.pyplot as plt

        dates_eq = [d for d, _ in equity_curve]
        values   = [v for _, v in equity_curve]

        dates_wh  = [d for d, _ in weight_history]
        feats     = ['rsi', 'macd', 'momentum', 'volume', 'sentiment']
        wt_series = {f: [w.get(f, 0.0) for _, w in weight_history] for f in feats}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        ax1.plot(dates_eq, values, color='steelblue', linewidth=1.5)
        ax1.axhline(10_000, color='grey', linestyle='--', linewidth=0.8)
        ax1.set_title('Simulated Portfolio Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(alpha=0.3)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for feat, color in zip(feats, colors):
            ax2.plot(dates_wh, wt_series[feat], linewidth=1.5, label=feat, color=color)
        ax2.set_title('Learned Indicator Weights Over Time (incl. sentiment)')
        ax2.set_ylabel('Weight')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        print("  Chart saved     → backtest_results.png")
        plt.show()
    except ImportError:
        print("  (install matplotlib to see charts: pip install matplotlib)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train signal learner on historical data'
    )
    parser.add_argument('--years',  type=int,   default=2,    help='Years of history (default: 2)')
    parser.add_argument('--hold',   type=int,   default=5,    help='Hold period in trading days (default: 5)')
    parser.add_argument('--topk',   type=int,   default=5,    help='Picks per day (default: 5)')
    parser.add_argument('--lr',     type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--plot',   action='store_true',       help='Show equity + weight charts')
    parser.add_argument('--reset',  action='store_true',       help='Wipe learned weights before training')
    args = parser.parse_args()

    run_backtest(
        years=args.years,
        hold_period=args.hold,
        top_k=args.topk,
        learning_rate=args.lr,
        plot=args.plot,
        reset=args.reset,
    )
