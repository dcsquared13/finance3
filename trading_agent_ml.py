"""
trading_agent_ml.py — Main entry point for the ML-enhanced trading agent.

Differences from trading_agent.py
===================================
Two new steps are added around the existing 8-step loop:

  Step 0 (before market open): RESOLVE PENDING PICKS
    - Load picks logged from HOLD_PERIOD trading days ago (stored in
      data/pending_picks.json).
    - Fetch their current prices to compute actual returns.
    - Call learner.update(features, actual_return) for each one.
    - This updates data/learned_weights.json in place.

  Step 9 (after session summary): LOG TODAY'S PICKS FOR FUTURE LEARNING
    - Save today's buys (symbol, entry_price, features) to pending_picks.json
      tagged with their target settlement date (today + HOLD_PERIOD trading days).

Step 5 now includes AI sentiment analysis:
    - lib.sentiment.analyze_sentiment() fetches news headlines for every stock
      in the universe and scores them with FinBERT (or VADER as fallback).
    - The per-ticker composite sentiment score [0, 1] is passed into
      SignalEngine.score_universe() and included in each stock's feature dict.
    - The learner treats 'sentiment' as a 5th learnable indicator weight.
    - Set SENTIMENT_MODEL = 'finbert' for higher accuracy (requires torch +
      transformers) or 'vader' for a lightweight, zero-dependency fallback.

All other steps (market check, portfolio sync, stop-loss, daily limit,
signal-sell, new buys, session summary) are identical to trading_agent.py.

Run modes
---------
    python trading_agent_ml.py               # live paper trading with learning
    python trading_agent_ml.py --dry-run     # full analysis, no orders placed
    python trading_agent_ml.py --force-open  # skip market-hours check (for testing)
    python trading_agent_ml.py --no-sentiment  # disable sentiment (faster, no deps)

First-time setup
----------------
Run `python backtest.py --years 2 --reset` before the first live session to
pre-train the model on two years of historical data. The learned weights will
be picked up automatically here.
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/agent_ml.log'),
    ]
)
log = logging.getLogger(__name__)

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from lib.broker      import AlpacaBroker
from lib.portfolio   import PortfolioManager
from lib.risk        import RiskManager
from lib.logger      import TradeLogger
from lib.learner     import LinearSignalLearner
from lib.strategy_ml import SignalEngine

HOLD_PERIOD     = 5          # Trading days before we evaluate a pick's outcome
PENDING_FILE    = 'data/pending_picks.json'
LEARNING_RATE   = 0.01

# Sentiment settings — mirrors finance2 defaults
SENTIMENT_MODEL    = 'auto'   # 'auto' | 'finbert' | 'vader'
SENTIMENT_ARTICLES = 15       # max headlines per ticker


# ---------------------------------------------------------------------------
# Pending picks persistence
# ---------------------------------------------------------------------------

def load_pending() -> list:
    if os.path.exists(PENDING_FILE):
        with open(PENDING_FILE) as f:
            return json.load(f)
    return []


def save_pending(picks: list):
    os.makedirs('data', exist_ok=True)
    with open(PENDING_FILE, 'w') as f:
        json.dump(picks, f, indent=2, default=str)


def add_pending(picks: list, new_entries: list):
    """Append new entries and prune entries older than 30 days."""
    today   = str(date.today())
    cutoff  = str(date.today() - timedelta(days=30))
    picks.extend(new_entries)
    return [p for p in picks if p.get('settle_date', today) >= cutoff]


# ---------------------------------------------------------------------------
# Step 0: Resolve mature picks and update the learner
# ---------------------------------------------------------------------------

def resolve_pending_picks(
    broker: AlpacaBroker,
    learner: LinearSignalLearner,
    trade_logger: TradeLogger,
) -> int:
    """
    Look through pending_picks.json for entries whose settle_date ≤ today.
    Fetch exit prices, compute returns, and feed them to the learner.
    Returns number of updates made.
    """
    today_str = str(date.today())
    pending   = load_pending()
    remaining = []
    updates   = 0

    for pick in pending:
        if pick.get('settle_date', '9999-12-31') > today_str:
            remaining.append(pick)
            continue

        symbol      = pick['symbol']
        entry_price = pick['entry_price']
        features    = pick['features']

        try:
            exit_price    = broker.get_current_price(symbol)
            actual_return = (exit_price - entry_price) / entry_price

            result  = learner.update(features, actual_return)
            updates += 1

            log.info(
                f"[LEARN] {symbol}: {actual_return*100:+.2f}% over {HOLD_PERIOD} days | "
                f"pred={result['predicted']:.3f} err={result['error']:.3f}"
            )

            trade_logger.log({
                'action':            'LEARN_UPDATE',
                'symbol':            symbol,
                'actual_return_pct': round(actual_return * 100, 2),
                'predicted_score':   round(result['predicted'], 4),
                'learn_error':       round(result['error'], 4),
                'new_weights':       json.dumps(
                    {k: round(v, 4) for k, v in result['weights'].items()}
                ),
                'reason': f"Resolved pick from {pick.get('date_bought')}",
            })

        except Exception as exc:
            log.warning(f"Could not resolve pick for {symbol}: {exc}")
            remaining.append(pick)   # keep it — try again next session

    save_pending(remaining)

    if updates:
        log.info(
            f"[LEARN] {updates} weight update(s) applied. "
            f"New weights: {learner.summary()['weights']}"
        )
    return updates


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run(
    dry_run: bool = False,
    force_open: bool = False,
    use_sentiment: bool = True,
):
    log.info("=" * 60)
    log.info(
        f"finance3 ML agent — {'DRY RUN' if dry_run else 'LIVE'} — "
        f"{'+sentiment' if use_sentiment else 'no-sentiment'} — "
        f"{datetime.now()}"
    )
    log.info("=" * 60)

    cfg = Config()
    cfg.validate()

    broker    = AlpacaBroker(cfg.APCA_API_KEY_ID, cfg.APCA_API_SECRET_KEY, cfg.APCA_API_BASE_URL)
    portfolio = PortfolioManager(cfg)
    risk      = RiskManager(cfg)
    logger    = TradeLogger(cfg)
    learner   = LinearSignalLearner(learning_rate=LEARNING_RATE)
    engine    = SignalEngine(learner=learner)

    log.info(f"Current weights: {learner.summary()['weights']}")

    # -----------------------------------------------------------------------
    # Step 0: Resolve any matured picks from HOLD_PERIOD days ago
    # -----------------------------------------------------------------------
    updates = resolve_pending_picks(broker, learner, logger)
    log.info(f"Step 0: {updates} learning update(s) applied from past picks")

    # -----------------------------------------------------------------------
    # Step 1: Market hours check
    # -----------------------------------------------------------------------
    if not force_open and not broker.is_market_open():
        log.info("Market is closed. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Sync portfolio with Alpaca
    # -----------------------------------------------------------------------
    portfolio.sync_with_alpaca(broker.get_positions())
    account     = broker.get_account()
    total_value = float(account.portfolio_value)
    portfolio.portfolio_value_at_open = total_value
    log.info(
        f"Step 2: Portfolio synced. "
        f"Value=${total_value:,.2f}  Cash=${portfolio.cash:,.2f}"
    )

    # -----------------------------------------------------------------------
    # Step 3: Stop-loss evaluation
    # -----------------------------------------------------------------------
    current_prices = broker.get_current_prices(list(portfolio.holdings.keys()))
    for symbol, holding in list(portfolio.holdings.items()):
        price    = current_prices.get(symbol)
        pct_down = risk.pct_down_from_entry(holding['avg_price'], price)
        if risk.is_stop_loss(pct_down):
            log.info(f"STOP-LOSS: {symbol} down {pct_down*100:.1f}%")
            if not dry_run:
                order = broker.submit_market_order(symbol, holding['shares'], 'sell')
                broker.wait_for_order_fill(order.id)
                portfolio.record_sell(symbol, price, holding['shares'])
            logger.log({
                'action': 'SELL_STOP_LOSS', 'symbol': symbol,
                'reason': f"Down {pct_down*100:.1f}% from entry",
            })

    # -----------------------------------------------------------------------
    # Step 4: Daily loss circuit breaker
    # -----------------------------------------------------------------------
    current_prices = broker.get_current_prices(list(portfolio.holdings.keys()))
    current_value  = portfolio.current_value(current_prices)
    if risk.is_daily_loss_limit(portfolio.portfolio_value_at_open, current_value):
        pct = (
            (portfolio.portfolio_value_at_open - current_value)
            / portfolio.portfolio_value_at_open
        )
        log.warning(f"DAILY LOSS LIMIT: down {pct*100:.1f}%. Halting new trades.")
        logger.log({'action': 'HALT', 'reason': f"Daily loss {pct*100:.1f}%"})
        logger.flush_session_summary(portfolio, current_value)
        return

    # -----------------------------------------------------------------------
    # Step 5a: Fetch AI sentiment for the universe
    # -----------------------------------------------------------------------
    universe = engine.get_universe()
    sentiment_scores: dict = {}

    if use_sentiment:
        log.info(
            f"Step 5a: Running sentiment analysis on {len(universe)} tickers "
            f"(model={SENTIMENT_MODEL}, max_articles={SENTIMENT_ARTICLES})..."
        )
        try:
            from lib.sentiment import analyze_sentiment
            sentiment_scores = analyze_sentiment(
                universe,
                model=SENTIMENT_MODEL,
                max_articles=SENTIMENT_ARTICLES,
            )
            method = next(
                (v['method'] for v in sentiment_scores.values() if v.get('method')),
                'unknown',
            )
            pos_count = sum(
                1 for v in sentiment_scores.values()
                if v.get('signal') in ('Positive', 'Very Positive')
            )
            log.info(
                f"Step 5a: Sentiment done (method={method}). "
                f"{pos_count}/{len(universe)} stocks have positive sentiment."
            )
        except Exception as exc:
            log.warning(
                f"Step 5a: Sentiment analysis failed ({exc}). "
                f"Proceeding with neutral sentiment (0.5) for all tickers."
            )
            sentiment_scores = {}

    # -----------------------------------------------------------------------
    # Step 5b: Score the universe with ML weights + sentiment
    # -----------------------------------------------------------------------
    log.info(f"Step 5b: Scoring {len(universe)} stocks with learned weights...")
    scored = engine.score_universe(universe, sentiment_scores=sentiment_scores)
    log.info(
        f"Step 5b: Top picks: "
        f"{[(s, round(sc, 3)) for s, sc, _, _ in scored[:5]]}"
    )

    # -----------------------------------------------------------------------
    # Step 6: Sell positions whose signal has degraded
    # -----------------------------------------------------------------------
    for symbol in list(portfolio.holdings.keys()):
        sell, score = engine.should_sell(symbol, scored)
        if sell:
            holding = portfolio.holdings[symbol]
            price   = current_prices.get(symbol)
            log.info(f"SELL signal: {symbol} score={score:.3f}")
            if not dry_run:
                order = broker.submit_market_order(symbol, holding['shares'], 'sell')
                broker.wait_for_order_fill(order.id)
                portfolio.record_sell(symbol, price, holding['shares'])
            logger.log({
                'action': 'SELL_SIGNAL', 'symbol': symbol,
                'score':  round(score, 4),
                'reason': 'Signal degraded below threshold',
            })

    # -----------------------------------------------------------------------
    # Step 7: Buy new positions
    # -----------------------------------------------------------------------
    slots_available = cfg.MAX_POSITIONS - len(portfolio.holdings)
    usable_cash     = portfolio.cash * (1 - cfg.CASH_RESERVE_PCT)
    buy_candidates  = engine.rank_buys(scored, list(portfolio.holdings.keys()))

    today_buys = []   # collected for Step 9

    for symbol, score, features, price in buy_candidates[:slots_available]:
        if usable_cash < 1.0:
            break

        position_size = risk.position_size(
            total_value,
            len(portfolio.holdings) + 1,
            cfg.MAX_POSITIONS,
            cfg.CASH_RESERVE_PCT,
        )
        shares = int(position_size / price)
        if shares < 1:
            continue

        cost = shares * price
        log.info(
            f"BUY: {symbol} × {shares} @ ${price:.2f}  score={score:.3f}  "
            f"sentiment={features.get('sentiment', 0.5):.2f} "
            f"({sentiment_scores.get(symbol, {}).get('signal', 'N/A')})"
        )

        if not dry_run:
            order = broker.submit_market_order(symbol, shares, 'buy')
            broker.wait_for_order_fill(order.id)
            portfolio.record_buy(symbol, price, shares)

        logger.log({
            'action':    'BUY',
            'symbol':    symbol,
            'shares':    shares,
            'price':     price,
            'score':     round(score, 4),
            'rsi':       round(features.get('rsi',       0.5), 4),
            'macd':      round(features.get('macd',      0.5), 4),
            'momentum':  round(features.get('momentum',  0.5), 4),
            'volume':    round(features.get('volume',    0.5), 4),
            'sentiment': round(features.get('sentiment', 0.5), 4),
            'sent_signal': sentiment_scores.get(symbol, {}).get('signal', 'N/A'),
            'reason':    f"ML score {score:.3f} ≥ {cfg.MIN_SCORE_TO_BUY}",
            'weights':   json.dumps(
                {k: round(v, 4) for k, v in learner.weights.items()}
            ),
        })

        usable_cash -= cost

        # Stash this pick for resolution in HOLD_PERIOD trading days
        settle_date = str(
            date.today() + timedelta(days=HOLD_PERIOD + 2)  # +2 buffer for weekends
        )
        today_buys.append({
            'symbol':      symbol,
            'date_bought': str(date.today()),
            'settle_date': settle_date,
            'entry_price': price,
            'features':    features,
        })

    # -----------------------------------------------------------------------
    # Step 8: Session summary
    # -----------------------------------------------------------------------
    current_prices = broker.get_current_prices(list(portfolio.holdings.keys()))
    current_value  = portfolio.current_value(current_prices)
    logger.flush_session_summary(portfolio, current_value)
    log.info(f"Session complete. Portfolio value=${current_value:,.2f}")
    log.info(f"Learner has {learner.update_count} lifetime updates.")

    # -----------------------------------------------------------------------
    # Step 9: Save today's buys for future learning
    # -----------------------------------------------------------------------
    if today_buys:
        pending = load_pending()
        pending = add_pending(pending, today_buys)
        save_pending(pending)
        log.info(
            f"Step 9: {len(today_buys)} new pick(s) queued for "
            f"learning in {HOLD_PERIOD} days."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='finance3 ML trading agent with AI sentiment'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Analyse only — no orders placed',
    )
    parser.add_argument(
        '--force-open', action='store_true',
        help='Skip the market-hours check (useful for testing)',
    )
    parser.add_argument(
        '--no-sentiment', action='store_true',
        help='Disable sentiment analysis (faster, no NLP dependencies required)',
    )
    args = parser.parse_args()

    run(
        dry_run=args.dry_run,
        force_open=args.force_open,
        use_sentiment=not args.no_sentiment,
    )
