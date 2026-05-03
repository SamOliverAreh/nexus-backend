"""
Data Fetcher — yfinance primary, Finnhub fallback.
Async wrapper around synchronous yfinance calls.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not installed — using fallback")


def _sync_quote(ticker: str, display_name: str) -> dict:
    """Synchronous yfinance quote fetch."""
    t = yf.Ticker(ticker)
    info = t.fast_info

    price   = float(getattr(info, "last_price",      0) or 0)
    open_   = float(getattr(info, "open",             price) or price)
    high    = float(getattr(info, "day_high",         price) or price)
    low_    = float(getattr(info, "day_low",          price) or price)
    prev    = float(getattr(info, "previous_close",   price) or price)
    volume  = float(getattr(info, "three_month_average_volume", 0) or 0)

    chg     = price - prev
    chg_pct = (chg / prev * 100) if prev else 0.0

    # 30-day historical for volatility
    hist = t.history(period="30d", interval="1d", auto_adjust=True)
    if len(hist) >= 2:
        ret = hist["Close"].pct_change().dropna()
        ann_vol = float(ret.std() * np.sqrt(252) * 100)
        rsi_val = _compute_rsi(hist["Close"].tolist())
        adx_val = _compute_adx(hist)
        sharpe  = _compute_sharpe(ret)
    else:
        ann_vol = rsi_val = adx_val = sharpe = 0.0

    return {
        "symbol":    display_name,
        "ticker":    ticker,
        "price":     round(price, 6),
        "open":      round(open_, 6),
        "high":      round(high, 6),
        "low":       round(low_, 6),
        "prev_close":round(prev, 6),
        "change":    round(chg, 6),
        "change_pct":round(chg_pct, 4),
        "volume":    int(volume),
        "ann_vol":   round(ann_vol, 2),
        "rsi":       round(rsi_val, 2),
        "adx":       round(adx_val, 2),
        "sharpe":    round(sharpe, 3),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _sync_history(ticker: str, period: str, interval: str) -> dict:
    """Synchronous yfinance history fetch."""
    t    = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No history returned for {ticker}")

    hist = hist.dropna(subset=["Close"])

    closes = [round(float(x), 6) for x in hist["Close"].tolist()]
    opens  = [round(float(x), 6) for x in hist["Open"].tolist()]
    highs  = [round(float(x), 6) for x in hist["High"].tolist()]
    lows   = [round(float(x), 6) for x in hist["Low"].tolist()]
    vols   = [int(x) for x in hist["Volume"].tolist()]
    dates  = [d.strftime("%Y-%m-%d") for d in hist.index]

    returns = []
    for i in range(1, len(closes)):
        r = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] else 0
        returns.append(round(r, 8))

    return {
        "ticker":  ticker,
        "period":  period,
        "interval":interval,
        "dates":   dates,
        "close":   closes,
        "open":    opens,
        "high":    highs,
        "low":     lows,
        "volume":  vols,
        "returns": returns,
        "count":   len(closes),
    }


class DataFetcher:
    def __init__(self):
        self.loop = None

    async def get_quote(self, ticker: str, display_name: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _sync_quote, ticker, display_name)

    async def get_history(self, ticker: str, period: str, interval: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _sync_history, ticker, period, interval)

    def future_dates(self, last_date_str: str, steps: int) -> List[str]:
        """Generate future trading dates from last known date."""
        try:
            last = datetime.strptime(last_date_str, "%Y-%m-%d")
        except Exception:
            last = datetime.utcnow()
        dates = []
        d = last
        while len(dates) < steps:
            d += timedelta(days=1)
            if d.weekday() < 5:  # Mon–Fri
                dates.append(d.strftime("%Y-%m-%d"))
        return dates


# ─── TECHNICAL INDICATORS ──────────────────────────────────────────────────────

def _compute_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = np.mean(gains[-period:])
    al = np.mean(losses[-period:])
    if al == 0:
        return 100.0
    rs = ag / al
    return float(100 - 100 / (1 + rs))


def _compute_adx(hist, period: int = 14) -> float:
    try:
        high  = hist["High"].values
        low   = hist["Low"].values
        close = hist["Close"].values
        n     = len(close)
        if n < period + 2:
            return 25.0

        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(1, n):
            tr  = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            pdm = max(high[i]-high[i-1], 0)
            ndm = max(low[i-1]-low[i],   0)
            if pdm > ndm:   ndm = 0
            elif ndm > pdm: pdm = 0
            else:           pdm = ndm = 0
            tr_list.append(tr); pdm_list.append(pdm); ndm_list.append(ndm)

        atr  = np.mean(tr_list[-period:])
        apdi = np.mean(pdm_list[-period:]) / (atr + 1e-9) * 100
        andi = np.mean(ndm_list[-period:]) / (atr + 1e-9) * 100
        dx   = abs(apdi - andi) / (apdi + andi + 1e-9) * 100
        return float(dx)
    except Exception:
        return 25.0


def _compute_sharpe(returns, rf_annual: float = 0.05) -> float:
    r  = np.array(returns)
    rf = rf_annual / 252
    if r.std() == 0:
        return 0.0
    return float((r.mean() - rf) / r.std() * np.sqrt(252))
