#!/usr/bin/env python3
"""
trading_agent.py — Main entry point for the finance3 stock trading agent.

Run once daily at market open (e.g. via cron at 9:30 AM ET Mon-Fri).
The agent will:
  1. Check market is open (exits gracefully if not)
  2. Sync portfolio with Alpaca
  3. Apply stop-losses to existing positions
  4. Check daily loss limit (halts if breached)
  5. Score the S&P 500 universe for new signals
  6. Sell positions whose signal has degraded
  7. Buy new top-ranked positions with available capital
  8. Log all decisions and save portfolio state

Usage:
  python trading_agent.py              # Live run
  python trading_agent.py --dry-run    # Simulate without placing orders
  python trading_agent.py --force-open # Skip market-open check (for testing)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

from config import Config
from lib.broker import AlpacaBroker
from lib.strategy import SignalEngine
from lib.risk import RiskManager
from lib.portfolio import PortfolioManager
from lib.logger import TradeLogger

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="finance3 trading agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full analysis but do NOT submit any orders.",
    )
    parser.add_argument(
        "--force-open",
        action="store_true",
        help="Skip market-open check (useful for weekend testing).",
    )
    return parser.parse_args()


def print_banner(dry_run: bool) -> None:
    mode = "DRY RUN — NO ORDERS WILL BE PLACED" if dry_run else "LIVE PAPER TRADING"
    print("\n" + "=" * 70)
    print(f"  finance3 TRADING AGENT  |  {mode}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def estimate_portfolio_value(portfolio: PortfolioManager, broker: AlpacaBroker) -> float:
    """Estimate total portfolio value = cash + sum of market values."""
    account = broker.get_account()
    return float(account["portfolio_value"])


# ── Main trading loop ─────────────────────────────────────────────────────────

def run(dry_run: bool = False, force_open: bool = False) -> None:
    print_banner(dry_run)

    # ── 0. Setup ─────────────────────────────────────────────────────────

    cfg = Config()
    cfg.validate()

    broker    = AlpacaBroker(cfg.APCA_API_KEY_ID, cfg.APCA_API_SECRET_KEY, cfg.APCA_API_BASE_URL)
    strategy  = SignalEngine(cfg)
    risk      = RiskManager(cfg)
    portfolio = PortfolioManager(cfg)
    trade_log = TradeLogger(cfg)

    # ── 1. Market check ───────────────────────────────────────────────────

    if not force_open:
        if not broker.is_market_open():
            try:
                next_open = str(broker.get_next_market_open())
            except Exception:
                next_open = "unknown"
            log.info("Market is closed. Next open: %s", next_open)
            trade_log.log_market_closed(next_open)
            print(f"\n  Market is closed. Next open: {next_open}\n")
            return
    else:
        log.info("--force-open flag set: skipping market-open check.")

    # ── 2. Load + sync portfolio ──────────────────────────────────────────

    portfolio.load()
    portfolio.sync_with_alpaca(broker)

    current_value = estimate_portfolio_value(portfolio, broker)
    portfolio.record_open_value(current_value)

    trade_log.log_session_start(
        portfolio_value=current_value,
        n_holdings=portfolio.n_holdings(),
        cash=portfolio.cash,
    )

    print(f"\n{'─'*60}")
    print(portfolio.summary())
    print(f"  Portfolio value: ${current_value:,.2f}")
    print(f"{'─'*60}\n")

    # ── 3. Stop-loss evaluation ───────────────────────────────────────────

    log.info("─── STEP 1: Stop-loss checks ───")
    live_positions = broker.get_positions()
    stop_loss_hits = risk.check_stop_losses(live_positions)

    for pos, sl_reason in stop_loss_hits:
        sym = pos["symbol"]
        qty = pos["qty"]
        curr_price = pos["current_price"]

        log.warning("Stop-loss: selling %d shares of %s @ $%.2f", qty, sym, curr_price)

        if not dry_run:
            order = broker.submit_market_order(sym, qty, "sell")
            if order:
                broker.wait_for_order_fill(order["id"])
                pnl = portfolio.record_sell(sym, qty, curr_price)
                new_value = estimate_portfolio_value(portfolio, broker)
                trade_log.log_trade(
                    ticker=sym,
                    action="SELL_STOP_LOSS",
                    shares=qty,
                    price=curr_price,
                    reason=sl_reason,
                    portfolio_cash=portfolio.cash,
                    portfolio_value=new_value,
                )
                print(f"  ✂  STOP-LOSS SELL: {sym} {qty} shares @ ${curr_price:.2f} | P&L: ${pnl:+.2f}")
        else:
            print(f"  [DRY RUN] Would STOP-LOSS SELL: {sym} {qty} shares @ ${curr_price:.2f}")

    # ── 4. Daily loss limit check ─────────────────────────────────────────

    log.info("─── STEP 2: Daily loss limit check ───")
    current_value = estimate_portfolio_value(portfolio, broker)
    halt, halt_reason = risk.check_daily_loss_limit(
        portfolio.portfolio_value_at_open, current_value
    )

    if halt:
        trade_log.log_halt(halt_reason, current_value, portfolio.cash)
        print(f"\n  ⛔  TRADING HALTED: {halt_reason}\n")
        log.warning("Trading halted for today. Exiting.")
        return

    # ── 5. Signal analysis ────────────────────────────────────────────────

    log.info("─── STEP 3: Scoring universe ───")
    universe = strategy.get_universe()
    log.info("Universe: %d stocks", len(universe))

    print(f"  Analysing {len(universe)} stocks…")
    t0 = time.time()
    scores = strategy.score_universe(universe)
    elapsed = time.time() - t0
    print(f"  Scored {len(scores)} stocks in {elapsed:.1f}s\n")

    if not scores:
        log.error("No scores returned — aborting session.")
        return

    # ── 6. Sell degraded positions ────────────────────────────────────────

    log.info("─── STEP 4: Signal-degraded sells ───")
    # Refresh positions after stop-loss sells
    live_positions = broker.get_positions()
    held_symbols = {p["symbol"] for p in live_positions}

    degraded = []
    for sym in list(held_symbols):
        if strategy.should_sell(sym, scores):
            pos = next((p for p in live_positions if p["symbol"] == sym), None)
            if pos:
                degraded.append((sym, pos))

    for sym, pos in degraded:
        qty = pos["qty"]
        curr_price = pos["current_price"]
        sig_data = scores.get(sym, {})
        sell_reason = f"Signal degraded to Sell | {sig_data.get('reasoning', '')}"

        log.info("Signal sell: %s %d shares @ $%.2f", sym, qty, curr_price)

        if not dry_run:
            order = broker.submit_market_order(sym, qty, "sell")
            if order:
                broker.wait_for_order_fill(order["id"])
                pnl = portfolio.record_sell(sym, qty, curr_price)
                new_value = estimate_portfolio_value(portfolio, broker)
                trade_log.log_trade(
                    ticker=sym,
                    action="SELL_SIGNAL",
                    shares=qty,
                    price=curr_price,
                    reason=sell_reason,
                    scores=sig_data,
                    portfolio_cash=portfolio.cash,
                    portfolio_value=new_value,
                )
                print(f"  📉  SIGNAL SELL: {sym} {qty} shares @ ${curr_price:.2f} | P&L: ${pnl:+.2f}")
                held_symbols.discard(sym)
        else:
            print(f"  [DRY RUN] Would SIGNAL SELL: {sym} {qty} shares (score={sig_data.get('score', '?')})")

    # ── 7. Buy new positions ──────────────────────────────────────────────

    log.info("─── STEP 5: Buying new positions ───")

    # Refresh state after sells
    portfolio.sync_with_alpaca(broker)
    current_positions = portfolio.n_holdings()
    slots = risk.slots_available(current_positions)

    if slots <= 0:
        log.info("No open slots — skipping buys (max positions reached).")
        print("  No open slots for new positions.\n")
    else:
        ranked_buys = strategy.rank_buys(scores)
        # Skip tickers we already hold
        candidates = [t for t in ranked_buys if t not in portfolio.holding_symbols()]

        log.info("%d buy candidates, %d slots available", len(candidates), slots)
        print(f"  {len(candidates)} buy candidates | {slots} slots available\n")

        buys_made = 0
        for ticker in candidates:
            if buys_made >= slots:
                break

            sig_data = scores[ticker]

            # Get live price
            price = broker.get_current_price(ticker)
            if price is None or price <= 0:
                trade_log.log_skipped(ticker, "Could not fetch price", sig_data)
                continue

            # Calculate position size
            remaining_slots = slots - buys_made
            shares = risk.calculate_position_size(portfolio.cash, price, remaining_slots)
            if shares <= 0:
                trade_log.log_skipped(
                    ticker,
                    f"Not enough cash for even 1 share at ${price:.2f}",
                    sig_data,
                )
                log.warning("Skipping %s — insufficient cash at $%.2f/share", ticker, price)
                break  # No point checking further if cash is exhausted

            cost = shares * price
            buy_reason = sig_data.get("reasoning", "")

            log.info(
                "BUY: %s %d shares @ $%.2f = $%.2f | %s",
                ticker, shares, price, cost, buy_reason,
            )

            if not dry_run:
                order = broker.submit_market_order(ticker, shares, "buy")
                if order:
                    broker.wait_for_order_fill(order["id"])
                    portfolio.record_buy(ticker, shares, price)
                    new_value = estimate_portfolio_value(portfolio, broker)
                    trade_log.log_trade(
                        ticker=ticker,
                        action="BUY",
                        shares=shares,
                        price=price,
                        reason=buy_reason,
                        scores=sig_data,
                        portfolio_cash=portfolio.cash,
                        portfolio_value=new_value,
                    )
                    print(
                        f"  📈  BUY: {ticker} {shares} shares @ ${price:.2f} = ${cost:.2f}"
                        f" | score={sig_data['score']:.2f}"
                    )
                    buys_made += 1
                else:
                    log.warning("Order failed for %s — skipping.", ticker)
            else:
                print(
                    f"  [DRY RUN] Would BUY: {ticker} {shares} shares @ ${price:.2f} = ${cost:.2f}"
                    f" | score={sig_data['score']:.2f}"
                )
                buys_made += 1

        if buys_made == 0:
            print("  No buys executed (no qualifying candidates or insufficient cash).\n")

    # ── 8. Session summary ────────────────────────────────────────────────

    log.info("─── SESSION COMPLETE ───")
    final_value = estimate_portfolio_value(portfolio, broker)
    portfolio.sync_with_alpaca(broker)

    # Get prices for holdings summary
    held = portfolio.holding_symbols()
    prices = broker.get_current_prices(held) if held else {}

    session_pnl = final_value - portfolio.portfolio_value_at_open
    session_pnl_pct = session_pnl / portfolio.portfolio_value_at_open if portfolio.portfolio_value_at_open > 0 else 0

    print(f"\n{'='*60}")
    print("  SESSION SUMMARY")
    print(f"{'='*60}")
    print(portfolio.summary(prices))
    print(f"\n  Portfolio value : ${final_value:,.2f}")
    print(f"  Session P&L     : ${session_pnl:+,.2f} ({session_pnl_pct:+.2%})")
    if dry_run:
        print("\n  ⚠  DRY RUN — no orders were placed.")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    try:
        run(dry_run=args.dry_run, force_open=args.force_open)
    except KeyboardInterrupt:
        log.info("Agent interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        log.exception("Unhandled exception in trading agent: %s", exc)
        sys.exit(1)
