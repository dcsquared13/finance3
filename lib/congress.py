"""
lib/congress.py — Congressional trading signal for finance3

Fetches STOCK Act disclosures from two free public APIs and converts
recent congressional buy/sell activity into a per-ticker sentiment
score [0.0 – 1.0].

Data sources (no API key required):
  House : https://housestockwatcher.com/api/issuances/json
  Senate: https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/
          aggregate/all_transactions.json

Score semantics
---------------
  > 0.60  — Net congressional buying   (bullish signal)
  0.45–0.60 — Mixed / minimal activity (neutral)
  < 0.45  — Net congressional selling  (bearish signal)

  No recent activity → 0.5 (neutral)

Scoring methodology
-------------------
Each trade is weighted by two factors:

  1. Recency  — how many days ago the trade was reported
       0 – 30  days  →  1.00
       31 – 60 days  →  0.60
       61 – 90 days  →  0.30
       > 90 days     →  excluded

  2. Amount   — reported dollar size of the transaction
       $1,001 – $15,000     →  1.0
       $15,001 – $50,000    →  1.5
       $50,001 – $100,000   →  2.0
       $100,001 – $250,000  →  2.5
       $250,001 – $500,000  →  3.0
       $500,001 – $1M       →  3.5
       > $1,000,000         →  4.0

  score = weighted_buys / (weighted_buys + weighted_sells)

  A minimum combined weight of 0.5 is required before the score departs
  from neutral — avoids noise from a single old micro-trade.

Caching
-------
  Fetched data is cached in data/congress_cache.json for
  CACHE_TTL_HOURS (default 24h) to avoid hammering the public APIs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── Public API endpoints ──────────────────────────────────────────────────────
# Note: these are community-maintained endpoints and can go down without notice.
# All fetch failures are handled gracefully — the agent continues with whatever
# data was retrieved (or neutral 0.5 scores if nothing could be fetched).

HOUSE_TRADES_URL = "https://housestockwatcher.com/api/issuances/json"

# The original S3 bucket (senate-stock-watcher-data.s3-us-west-2.amazonaws.com)
# was made private. The underlying data lives in a public GitHub repo;
# the raw content URL below is the current reliable source.
SENATE_TRADES_URL = (
    "https://raw.githubusercontent.com/timothycarambat"
    "/senate-stock-watcher-data/main/data/all_transactions.json"
)
# Fallback if the GitHub raw URL also changes
SENATE_TRADES_URL_FALLBACK = "https://senatestockwatcher.com/api/trades.json"

CACHE_FILE = "data/congress_cache.json"
CACHE_TTL_HOURS = 24
REQUEST_TIMEOUT = 20  # seconds

# ── Amount-size multipliers ───────────────────────────────────────────────────

_AMOUNT_TIERS: list[tuple[float, float]] = [
    (1_000_000, 4.0),
    (  500_000, 3.5),
    (  250_000, 3.0),
    (  100_000, 2.5),
    (   50_000, 2.0),
    (   15_000, 1.5),
    (        0, 1.0),
]

_AMOUNT_KEYWORDS: dict[str, float] = {
    "over $1,000,000":      4.0,
    "$1,000,001":           4.0,
    "$500,001":             3.5,
    "$250,001":             3.0,
    "$100,001":             2.5,
    "$50,001":              2.0,
    "$15,001":              1.5,
    "$1,001":               1.0,
}


def _amount_multiplier(amount_str: str) -> float:
    """
    Parse a free-text amount range like '$50,001 - $100,000' and return
    a size multiplier.  Falls back to 1.0 on any parse error.
    """
    if not amount_str:
        return 1.0

    s = amount_str.lower().replace(",", "").replace("$", "").strip()

    # Check for keyword shortcuts first (avoids regex)
    for kw, mult in _AMOUNT_KEYWORDS.items():
        if kw.lower().replace(",", "") in s:
            return mult

    # Try to extract the lower bound of a range "X - Y" or "X+"
    try:
        # Take the first number in the string
        import re
        nums = re.findall(r"\d+", s)
        if nums:
            lower = float(nums[0])
            for threshold, mult in _AMOUNT_TIERS:
                if lower > threshold:
                    return mult
    except Exception:
        pass

    return 1.0


# ── Date parsing ──────────────────────────────────────────────────────────────

_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"]


def _parse_date(date_str: str) -> Optional[date]:
    """Try several common date formats; return None on failure."""
    if not date_str:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> Optional[dict]:
    """Return cached data if it exists and is within TTL, else None."""
    try:
        if not os.path.exists(CACHE_FILE):
            return None
        with open(CACHE_FILE, "r") as fh:
            data = json.load(fh)
        cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600
        if age_hours < CACHE_TTL_HOURS:
            log.debug("Congress cache hit (%.1f h old)", age_hours)
            return data.get("trades", [])
        log.debug("Congress cache stale (%.1f h old) — will refresh", age_hours)
    except Exception as exc:
        log.debug("Congress cache read error: %s", exc)
    return None


def _save_cache(trades: list[dict]) -> None:
    """Persist fetched trades to the cache file."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as fh:
            json.dump({"cached_at": datetime.utcnow().isoformat(), "trades": trades}, fh)
        log.debug("Congress cache saved (%d trades)", len(trades))
    except Exception as exc:
        log.warning("Could not write congress cache: %s", exc)


# ── Fetchers ──────────────────────────────────────────────────────────────────

def _fetch_house_trades() -> list[dict]:
    """
    Fetch House STOCK Act disclosures from housestockwatcher.com.
    Returns a list of normalised trade dicts.
    """
    log.info("Fetching House congressional trades…")
    try:
        resp = requests.get(HOUSE_TRADES_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        log.warning("House trades fetch failed: %s", exc)
        return []

    trades = []
    for item in raw:
        ticker = (item.get("ticker") or "").strip().upper()
        if not ticker or ticker in ("N/A", "--", ""):
            continue

        tx_type = (item.get("type") or "").lower()
        if "purchase" in tx_type:
            direction = "buy"
        elif "sale" in tx_type or "sold" in tx_type:
            direction = "sell"
        else:
            continue  # exchange, options etc. — skip

        trade_date = _parse_date(item.get("transaction_date") or item.get("disclosure_date", ""))
        if trade_date is None:
            continue

        trades.append({
            "ticker":    ticker,
            "direction": direction,
            "date":      trade_date,
            "amount":    item.get("amount", ""),
            "member":    item.get("representative", "Unknown"),
            "chamber":   "house",
        })

    log.info("House trades fetched: %d qualifying records", len(trades))
    return trades


def _fetch_senate_trades() -> list[dict]:
    """
    Fetch Senate STOCK Act disclosures from the GitHub raw data repo,
    with a fallback to senatestockwatcher.com. Returns a list of
    normalised trade dicts, or an empty list if both sources fail.
    """
    log.info("Fetching Senate congressional trades…")
    raw = None
    for url in (SENATE_TRADES_URL, SENATE_TRADES_URL_FALLBACK):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            log.info("Senate trades fetched from %s", url)
            break
        except Exception as exc:
            log.info("Senate trades unavailable at %s (%s) — trying fallback.", url, exc)

    if raw is None:
        log.info(
            "Senate trades could not be fetched from any source — "
            "proceeding with House data only."
        )
        return []

    trades = []
    for item in raw:
        ticker = (item.get("ticker") or "").strip().upper()
        if not ticker or ticker in ("N/A", "--", ""):
            continue

        tx_type = (item.get("type") or "").lower()
        if "purchase" in tx_type:
            direction = "buy"
        elif "sale" in tx_type or "sold" in tx_type:
            direction = "sell"
        else:
            continue

        trade_date = _parse_date(item.get("transaction_date") or item.get("disclosure_date", ""))
        if trade_date is None:
            continue

        trades.append({
            "ticker":    ticker,
            "direction": direction,
            "date":      trade_date,
            "amount":    item.get("amount", ""),
            "member":    item.get("senator", item.get("first_name", "") + " " + item.get("last_name", "")),
            "chamber":   "senate",
        })

    log.info("Senate trades fetched: %d qualifying records", len(trades))
    return trades


# ── Main fetch orchestrator ───────────────────────────────────────────────────

def fetch_all_trades(lookback_days: int = 90) -> list[dict]:
    """
    Return all qualifying congressional trades from the past `lookback_days`.

    Checks the on-disk cache first.  If stale or absent, re-fetches both
    chambers and writes a fresh cache.

    Each trade dict has keys:
        ticker, direction ('buy'|'sell'), date (date), amount (str),
        member (str), chamber ('house'|'senate'), days_ago (int)
    """
    # Try cache first
    cached = _load_cache()
    if cached is not None:
        all_trades = [
            dict(t, date=_parse_date(t["date"]) if isinstance(t["date"], str) else t["date"])
            for t in cached
            if t.get("ticker")
        ]
    else:
        house   = _fetch_house_trades()
        senate  = _fetch_senate_trades()
        all_trades = house + senate

        # Serialise dates to strings for JSON storage
        serialisable = [
            dict(t, date=t["date"].isoformat() if isinstance(t["date"], date) else str(t["date"]))
            for t in all_trades
        ]
        _save_cache(serialisable)

    # Filter to lookback window and annotate days_ago
    cutoff = date.today() - timedelta(days=lookback_days)
    filtered = []
    for t in all_trades:
        d = t.get("date")
        if isinstance(d, str):
            d = _parse_date(d)
        if d is None or d < cutoff:
            continue
        filtered.append(dict(t, date=d, days_ago=(date.today() - d).days))

    log.info(
        "Congressional trades in last %d days: %d records (%d tickers)",
        lookback_days,
        len(filtered),
        len({t["ticker"] for t in filtered}),
    )
    return filtered


# ── Scoring ───────────────────────────────────────────────────────────────────

def _recency_weight(days_ago: int) -> float:
    """Return a [0, 1] recency multiplier based on how old the trade is."""
    if days_ago <= 30:
        return 1.0
    if days_ago <= 60:
        return 0.6
    if days_ago <= 90:
        return 0.3
    return 0.0


_MIN_WEIGHT_THRESHOLD = 0.5   # combined weight below this → stay neutral


def congressional_signal(
    tickers: list[str],
    lookback_days: int = 90,
) -> dict[str, dict]:
    """
    Score each ticker based on recent congressional trading activity.

    Parameters
    ----------
    tickers      : list of ticker symbols to check
    lookback_days: window to consider (matches CONGRESS_LOOKBACK_DAYS in config)

    Returns
    -------
    Dict keyed by ticker, each value:
        score              float  [0, 1]   — bullish > 0.5, bearish < 0.5
        n_buys             int             — raw buy transaction count
        n_sells            int             — raw sell transaction count
        n_members_buying   int             — distinct members who bought
        n_members_selling  int             — distinct members who sold
        weighted_buys      float           — recency+amount-weighted buy score
        weighted_sells     float           — recency+amount-weighted sell score
        net_direction      str             — 'bullish' | 'bearish' | 'neutral'
        latest_trade_days_ago int|None
        total_activity     int             — total qualifying trades
    """
    ticker_set = {t.upper() for t in tickers}
    all_trades = fetch_all_trades(lookback_days)

    # Group trades by ticker
    by_ticker: dict[str, list[dict]] = {t: [] for t in ticker_set}
    for trade in all_trades:
        sym = trade["ticker"]
        if sym in by_ticker:
            by_ticker[sym].append(trade)

    results: dict[str, dict] = {}

    for ticker in ticker_set:
        trades = by_ticker.get(ticker, [])

        if not trades:
            results[ticker] = {
                "score":              0.5,
                "n_buys":             0,
                "n_sells":            0,
                "n_members_buying":   0,
                "n_members_selling":  0,
                "weighted_buys":      0.0,
                "weighted_sells":     0.0,
                "net_direction":      "neutral",
                "latest_trade_days_ago": None,
                "total_activity":     0,
            }
            continue

        buys  = [t for t in trades if t["direction"] == "buy"]
        sells = [t for t in trades if t["direction"] == "sell"]

        weighted_buys = sum(
            _recency_weight(t["days_ago"]) * _amount_multiplier(t["amount"])
            for t in buys
        )
        weighted_sells = sum(
            _recency_weight(t["days_ago"]) * _amount_multiplier(t["amount"])
            for t in sells
        )

        total_weight = weighted_buys + weighted_sells

        if total_weight < _MIN_WEIGHT_THRESHOLD:
            # Too little signal — return neutral
            score = 0.5
        else:
            score = round(weighted_buys / total_weight, 4)

        if score > 0.60:
            net_dir = "bullish"
        elif score < 0.45:
            net_dir = "bearish"
        else:
            net_dir = "neutral"

        latest = min(t["days_ago"] for t in trades) if trades else None

        results[ticker] = {
            "score":              score,
            "n_buys":             len(buys),
            "n_sells":            len(sells),
            "n_members_buying":   len({t["member"] for t in buys}),
            "n_members_selling":  len({t["member"] for t in sells}),
            "weighted_buys":      round(weighted_buys, 3),
            "weighted_sells":     round(weighted_sells, 3),
            "net_direction":      net_dir,
            "latest_trade_days_ago": latest,
            "total_activity":     len(trades),
        }

    return results


def congress_feature(ticker_result: Optional[dict]) -> float:
    """
    Extract a single [0, 1] feature value from one ticker's congressional
    signal result.  Returns 0.5 (neutral) when no result is available.

    Mirrors the sentiment_feature() interface for compatibility with
    LinearSignalLearner.
    """
    if ticker_result is None:
        return 0.5
    return float(ticker_result.get("score", 0.5))
