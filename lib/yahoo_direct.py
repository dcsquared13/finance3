"""
lib/yahoo_direct.py — Direct Yahoo Finance fetcher via the v8 chart API.

Drop-in replacement for yf.download() that bypasses yfinance's cookie/crumb
handshake with fc.yahoo.com.  Uses only the `requests` library.

Usage
-----
    from lib.yahoo_direct import download

    # Same signature as the subset of yf.download() used by this project:
    df = download(['AAPL', 'MSFT'], start='2023-01-01', end='2025-01-01')
    df = download(['AAPL'], period='90d')
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
})

# Retry / back-off settings
_MAX_RETRIES = 3
_RETRY_DELAY = 2          # seconds between retries
_REQUEST_DELAY = 0.25     # courtesy delay between tickers


def _period_to_dates(period: str) -> tuple[str, str]:
    """Convert a yfinance-style period string (e.g. '90d', '2y') to start/end dates."""
    today = datetime.today()
    unit = period[-1]
    n = int(period[:-1])
    if unit == 'd':
        start = today - timedelta(days=n)
    elif unit == 'y':
        start = today - timedelta(days=n * 365)
    elif unit == 'm':
        start = today - timedelta(days=n * 30)
    else:
        start = today - timedelta(days=n)
    return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')


def _to_unix(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to Unix timestamp."""
    return int(datetime.strptime(date_str, '%Y-%m-%d').timestamp())


def _fetch_one(symbol: str, period1: int, period2: int) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a single ticker from the Yahoo v8 chart API.
    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    and a DatetimeIndex, or None on failure.
    """
    # Try both Yahoo Finance query hosts for resilience
    hosts = ['query1.finance.yahoo.com', 'query2.finance.yahoo.com']

    for attempt in range(1, _MAX_RETRIES + 1):
        host = hosts[(attempt - 1) % len(hosts)]
        url = (
            f'https://{host}/v8/finance/chart/{symbol}'
            f'?period1={period1}&period2={period2}'
            f'&interval=1d&includeAdjustedClose=true'
        )
        try:
            resp = _SESSION.get(url, timeout=15)

            if resp.status_code == 404:
                log.warning(f"[yahoo_direct] {symbol}: 404 not found")
                return None
            if resp.status_code == 429:
                wait = _RETRY_DELAY * attempt * 2
                log.warning(f"[yahoo_direct] {symbol}: rate-limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                log.warning(f"[yahoo_direct] {symbol}: HTTP {resp.status_code}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)
                    continue
                return None

            data = resp.json()
            result = data['chart']['result'][0]

            timestamps = result.get('timestamp')
            if not timestamps:
                log.debug(f"[yahoo_direct] {symbol}: no timestamp data")
                return None

            quote = result['indicators']['quote'][0]

            # Use adjusted close if available, otherwise regular close
            adj = result['indicators'].get('adjclose', [{}])
            adjclose = adj[0].get('adjclose') if adj else None

            df = pd.DataFrame({
                'Open':   quote.get('open'),
                'High':   quote.get('high'),
                'Low':    quote.get('low'),
                'Close':  adjclose if adjclose else quote.get('close'),
                'Volume': quote.get('volume'),
            }, index=pd.to_datetime(timestamps, unit='s', utc=True))

            df.index = df.index.tz_localize(None)   # drop timezone for consistency
            df.index.name = 'Date'
            df = df.dropna(subset=['Close'])

            return df

        except requests.exceptions.ConnectionError as e:
            log.warning(f"[yahoo_direct] {symbol} attempt {attempt}: connection error — {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)
        except Exception as e:
            log.warning(f"[yahoo_direct] {symbol} attempt {attempt}: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)

    log.error(f"[yahoo_direct] {symbol}: failed after {_MAX_RETRIES} attempts")
    return None


def download(
    tickers: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    auto_adjust: bool = True,      # ignored — we always use adjusted close
    progress: bool = False,        # ignored — we log instead
) -> pd.DataFrame:
    """
    Drop-in replacement for yf.download() that uses the v8 chart API.

    Returns a DataFrame with a MultiIndex on columns: (field, symbol)
    matching what yf.download() returns for multiple tickers.
    For a single ticker, still uses MultiIndex for consistency.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # Resolve date range
    if period and not start:
        start, end = _period_to_dates(period)
    if not end:
        end = datetime.today().strftime('%Y-%m-%d')
    if not start:
        start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    period1 = _to_unix(start)
    period2 = _to_unix(end)

    frames = {}
    failed = []

    for i, symbol in enumerate(tickers):
        df = _fetch_one(symbol, period1, period2)
        if df is not None and not df.empty:
            frames[symbol] = df
        else:
            failed.append(symbol)

        # Courtesy delay to avoid rate-limiting (skip after last ticker)
        if i < len(tickers) - 1:
            time.sleep(_REQUEST_DELAY)

    if failed:
        log.warning(f"[yahoo_direct] Failed tickers ({len(failed)}): {failed}")

    if not frames:
        log.error("[yahoo_direct] No data fetched for any ticker")
        return pd.DataFrame()

    log.info(f"[yahoo_direct] Fetched {len(frames)}/{len(tickers)} tickers")

    # Build MultiIndex DataFrame matching yf.download() format:
    # columns = MultiIndex(field, symbol)
    fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    pieces = {}
    for field in fields:
        field_frames = {}
        for sym, df in frames.items():
            if field in df.columns:
                field_frames[sym] = df[field]
        if field_frames:
            pieces[field] = pd.DataFrame(field_frames)

    if not pieces:
        return pd.DataFrame()

    combined = pd.concat(pieces, axis=1)
    # combined now has MultiIndex columns: (field, symbol)
    # yf.download() uses the same format

    return combined
