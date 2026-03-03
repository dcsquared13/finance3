"""
lib/sentiment.py — AI sentiment analysis for finance3

Ported from finance2/lib/sentiment.py.

Primary:  ProsusAI/FinBERT  (~400 MB download on first run, then cached in
          ~/.cache/huggingface/).  Requires: torch, transformers.
Fallback: NLTK VADER  (lightweight, no large downloads).
          Requires: nltk.

Mode "auto": tries FinBERT, silently falls back to VADER if torch /
             transformers is not installed.

Per-ticker output dict keys
---------------------------
  composite   float [0, 1]  — 1 = very positive, 0 = very negative
  signal      str           — one of the five labels below
  n_articles  int           — number of headlines scored
  headlines   [str]         — headlines that were used
  raw_scores  [float]       — per-headline composite values
  method      str           — 'finbert' or 'vader'

Signal thresholds
-----------------
  Very Positive  >= 0.70
  Positive       >= 0.55
  Neutral        >= 0.45
  Negative       >= 0.30
  Very Negative  <  0.30

Install (optional — VADER works out of the box if you have nltk):
  pip install torch transformers   # for FinBERT
  pip install nltk                 # for VADER fallback
"""

import logging
import yfinance as yf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _sentiment_signal(composite: float) -> str:
    if composite >= 0.70:
        return 'Very Positive'
    elif composite >= 0.55:
        return 'Positive'
    elif composite >= 0.45:
        return 'Neutral'
    elif composite >= 0.30:
        return 'Negative'
    else:
        return 'Very Negative'


# ---------------------------------------------------------------------------
# FinBERT scorer
# ---------------------------------------------------------------------------

def _build_finbert():
    """Lazily load the FinBERT pipeline and return a scoring callable."""
    from transformers import pipeline  # noqa: delayed import

    _pipe = pipeline(
        'text-classification',
        model='ProsusAI/finbert',
        return_all_scores=True,
        device=-1,       # CPU; change to 0 to use first GPU
    )

    # FinBERT label → sentiment value mapping
    LABEL_MAP = {'positive': 1.0, 'neutral': 0.5, 'negative': 0.0}

    def _score_texts(texts: list[str]) -> list[float]:
        results = _pipe(texts, truncation=True, max_length=512, batch_size=8)
        scores = []
        for result in results:
            label_scores = {r['label'].lower(): r['score'] for r in result}
            # Expected-value across the three FinBERT labels
            weighted = sum(LABEL_MAP.get(lbl, 0.5) * prob
                          for lbl, prob in label_scores.items())
            scores.append(float(weighted))
        return scores

    return _score_texts


# ---------------------------------------------------------------------------
# VADER scorer
# ---------------------------------------------------------------------------

def _build_vader():
    """Lazily load NLTK VADER and return a scoring callable."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except LookupError:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

    _sia = SentimentIntensityAnalyzer()

    def _score_texts(texts: list[str]) -> list[float]:
        scores = []
        for text in texts:
            compound = _sia.polarity_scores(text)['compound']
            # VADER compound is in [-1, 1]; map to [0, 1]
            scores.append((compound + 1.0) / 2.0)
        return scores

    return _score_texts


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_scorer = None
_scorer_method: str | None = None


def _get_scorer(model: str = 'auto'):
    """Return (scoring_fn, method_name), loading the model once."""
    global _scorer, _scorer_method

    if _scorer is not None:
        return _scorer, _scorer_method

    if model in ('finbert', 'auto'):
        try:
            fn = _build_finbert()
            _scorer = fn
            _scorer_method = 'finbert'
            log.info("FinBERT sentiment model loaded (ProsusAI/finbert)")
            return _scorer, _scorer_method
        except Exception as exc:
            if model == 'finbert':
                raise RuntimeError(
                    "FinBERT requested but could not be loaded. "
                    "Install: pip install torch transformers"
                ) from exc
            log.warning(
                f"FinBERT unavailable ({exc}), falling back to VADER"
            )

    # VADER path
    try:
        fn = _build_vader()
        _scorer = fn
        _scorer_method = 'vader'
        log.info("VADER sentiment model loaded")
        return _scorer, _scorer_method
    except Exception as exc:
        raise RuntimeError(
            "Neither FinBERT nor VADER could be loaded. "
            "Install: pip install nltk  (then python -m nltk.downloader vader_lexicon)"
        ) from exc


# ---------------------------------------------------------------------------
# News headline fetching helpers
# ---------------------------------------------------------------------------

def _extract_headline(item: dict) -> str | None:
    """
    Handle both old yfinance news format (flat dict) and new nested format
    where the title lives under item['content']['title'].
    """
    # Old format: item['title']
    title = item.get('title')
    if title:
        return str(title).strip()

    # New format: item['content']['title']
    content = item.get('content')
    if isinstance(content, dict):
        title = content.get('title')
        if title:
            return str(title).strip()

    return None


def _fetch_headlines(ticker: str, max_articles: int) -> list[str]:
    """Fetch up to `max_articles` news headlines for `ticker` via yfinance."""
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception as exc:
        log.debug(f"yfinance news fetch failed for {ticker}: {exc}")
        return []

    headlines = []
    for item in news_items[:max_articles]:
        headline = _extract_headline(item)
        if headline:
            headlines.append(headline)

    return headlines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sentiment(
    tickers: list[str],
    model: str = 'auto',
    max_articles: int = 15,
) -> dict:
    """
    Fetch news headlines and score the sentiment for each ticker.

    Parameters
    ----------
    tickers      : list of ticker symbols to analyse
    model        : 'auto' | 'finbert' | 'vader'
    max_articles : maximum number of headlines to score per ticker

    Returns
    -------
    dict keyed by ticker, each value a dict with:
        composite   float  [0, 1]
        signal      str
        n_articles  int
        headlines   list[str]
        raw_scores  list[float]
        method      str
    """
    scorer, method = _get_scorer(model)
    results: dict = {}

    for ticker in tickers:
        headlines = _fetch_headlines(ticker, max_articles)

        if not headlines:
            results[ticker] = {
                'composite':  0.5,
                'signal':     'Neutral',
                'n_articles': 0,
                'headlines':  [],
                'raw_scores': [],
                'method':     method,
            }
            continue

        try:
            raw_scores = scorer(headlines)
            composite = float(sum(raw_scores) / len(raw_scores))
        except Exception as exc:
            log.warning(f"Scoring failed for {ticker}: {exc}")
            composite = 0.5
            raw_scores = []

        results[ticker] = {
            'composite':  composite,
            'signal':     _sentiment_signal(composite),
            'n_articles': len(headlines),
            'headlines':  headlines,
            'raw_scores': raw_scores,
            'method':     method,
        }

    return results


def sentiment_feature(ticker_result: dict | None) -> float:
    """
    Extract a single [0, 1] feature value from one ticker's sentiment result.
    Returns 0.5 (neutral) when no result is available.

    Useful for converting the analyze_sentiment() output dict into a scalar
    that can be passed directly to LinearSignalLearner.score().
    """
    if ticker_result is None:
        return 0.5
    return float(ticker_result.get('composite', 0.5))
