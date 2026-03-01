# finance3 — AI Stock Trading Agent

A Python-based automated stock trading agent that runs once daily at market open, analyzes a universe of S&P 500 stocks using a multi-indicator technical strategy, executes trades via the **Alpaca paper trading API**, and maintains a full log of every decision and its reasoning.

---

## Features

- **Multi-indicator strategy** — RSI, MACD, 20-day momentum, and volume breakout combined into a composite score (0–1) per stock
- **Risk management** — 8% stop-loss per position + 3% daily loss circuit breaker
- **Equal-weight position sizing** — capital split evenly across up to 10 simultaneous positions with a 5% cash reserve
- **Persistent state** — portfolio holdings and cash tracked in `data/portfolio.json`
- **Full trade log** — every trade (and every skip) recorded in `data/trade_log.csv` with indicator values and reasoning
- **Dry-run mode** — full analysis with no orders placed, for testing

---

## Project Structure

```
finance3/
├── trading_agent.py     # Main entry point
├── config.py            # All settings (loaded from env vars)
├── requirements.txt
├── .env.example         # Template for API keys
├── data/
│   ├── portfolio.json   # Current holdings + cash (auto-created)
│   └── trade_log.csv    # Running trade log (auto-created)
└── lib/
    ├── broker.py        # Alpaca API wrapper
    ├── strategy.py      # Signal engine (indicators + scoring)
    ├── risk.py          # Stop-loss, daily limit, position sizing
    ├── portfolio.py     # Portfolio state manager
    └── logger.py        # Trade log writer
```

---

## Setup

### 1. Get an Alpaca account

1. Sign up at [alpaca.markets](https://alpaca.markets) (free)
2. Navigate to **Paper Trading → API Keys**
3. Generate a new key pair and copy both values

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your APCA_API_KEY_ID and APCA_API_SECRET_KEY
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the agent (live paper trading)
```bash
python trading_agent.py
```
The agent will exit immediately if the market is closed.

### Dry run (no orders placed)
```bash
python trading_agent.py --dry-run
```
Runs the full analysis pipeline and shows what it *would* trade — nothing is submitted to Alpaca.

### Test on a weekend (skip market-open check)
```bash
python trading_agent.py --dry-run --force-open
```

---

## Scheduling (Daily at Market Open)

Add a cron job to run the agent automatically at 9:31 AM ET (Mon–Fri):

```bash
crontab -e
```

Add this line (adjust the path to your Python and project):
```
31 9 * * 1-5 /usr/bin/python3 /path/to/finance3/trading_agent.py >> /path/to/finance3/agent.log 2>&1
```

On macOS, use `launchd` or a tool like [whenever](https://github.com/javan/whenever) instead.

---

## Trading Strategy

The agent scores every stock in the universe (top 50 S&P 500 by default) on four indicators:

| Indicator | Logic | Weight |
|---|---|---|
| RSI (14d) | RSI < 30 = oversold (buy), RSI > 70 = overbought (sell) | 25% |
| MACD (12/26/9) | Bullish crossover + rising histogram = buy | 30% |
| Momentum (20d) | Price change vs 20 days ago, ranked vs peers | 25% |
| Volume Breakout | Today's volume vs 20d avg + price direction | 20% |

**Composite score ≥ 0.60** → Buy signal
**Composite score ≤ 0.40** → Sell signal
**Between** → Hold

---

## Risk Management

| Rule | Threshold |
|---|---|
| Stop-loss | Sell any position down ≥ 8% from avg entry price |
| Daily loss limit | Halt all new trading if portfolio down ≥ 3% from open |
| Max positions | 10 simultaneous holdings |
| Cash reserve | Always keep 5% of portfolio in cash |

---

## Data Files

### data/portfolio.json
Tracks the agent's current holdings and cash. Auto-synced with Alpaca positions at each session start.

```json
{
  "cash": 8450.00,
  "last_updated": "2026-03-01T09:35:42",
  "portfolio_value_at_open": 10120.00,
  "holdings": {
    "NVDA": {
      "shares": 3,
      "avg_price": 875.50,
      "date_bought": "2026-02-28",
      "cost_basis": 2626.50
    }
  }
}
```

### data/trade_log.csv
Running log of every trade and decision. Key columns:

| Column | Description |
|---|---|
| `action` | `BUY`, `SELL_STOP_LOSS`, `SELL_SIGNAL`, `SKIP`, `HALT`, `SESSION_START` |
| `reason` | Full text reasoning for every decision |
| `score` | Composite indicator score (0–1) |
| `rsi`, `macd_direction`, `momentum_pct`, `volume_ratio` | Raw indicator values |
| `portfolio_value_after` | Portfolio value immediately after the trade |

---

## Configuration

All settings live in `config.py` and can be overridden via environment variables:

| Setting | Default | Description |
|---|---|---|
| `STOP_LOSS_PCT` | 0.08 | Stop-loss threshold (8%) |
| `DAILY_LOSS_LIMIT_PCT` | 0.03 | Daily loss halt threshold (3%) |
| `MAX_POSITIONS` | 10 | Max simultaneous holdings |
| `CASH_RESERVE_PCT` | 0.05 | Cash to keep uninvested (5%) |
| `MIN_SCORE_TO_BUY` | 0.60 | Minimum composite score to open position |
| `SELL_SCORE_THRESHOLD` | 0.40 | Score below which position is exited |
| `UNIVERSE_SIZE` | 50 | Number of S&P 500 stocks to scan |
| `LOOKBACK_DAYS` | 90 | Days of price history to fetch |

---

## Disclaimer

This project is for **educational and research purposes only**. It uses paper trading (simulated money) and does not constitute financial advice. Past performance of any strategy does not guarantee future results. Always consult a licensed financial advisor before trading with real money.
