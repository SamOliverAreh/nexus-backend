"""
NEXUS Market Analysis Engine — FastAPI Backend
Endpoints: /quote, /history, /analyze, /health
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from datetime import datetime
from typing import Optional

from data.fetcher import DataFetcher
from models.arima_model import ARIMAModel
from models.garch_model import GARCHModel
from models.lstm_model import LSTMModel
from models.xgb_model import XGBModel
from models.prophet_model import ProphetModel
from models.hybrid_model import HybridModel
from models.ensemble import EnsembleResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NEXUS Market Analysis API",
    description="Live market data + statistical, ML, and hybrid forecasting",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fetcher = DataFetcher()

# ─── SYMBOL MAP ────────────────────────────────────────────────────────────────
SYMBOL_MAP = {
    # Forex
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X",    "EUR/GBP": "EURGBP=X",
    "USD/CHF": "CHF=X",    "NZD/USD": "NZDUSD=X",
    # Commodities
    "GOLD": "GC=F",        "SILVER": "SI=F",       "CRUDE OIL": "CL=F",
    "NAT GAS": "NG=F",     "COPPER": "HG=F",       "WHEAT": "ZW=F",
    "CORN": "ZC=F",        "PLATINUM": "PL=F",
    # Stocks
    "AAPL": "AAPL",  "MSFT": "MSFT",  "NVDA": "NVDA",  "AMZN": "AMZN",
    "TSLA": "TSLA",  "GOOGL": "GOOGL","META": "META",  "JPM": "JPM",
    # Indices
    "S&P 500": "^GSPC",  "NASDAQ": "^IXIC",   "DOW": "^DJI",
    "FTSE 100": "^FTSE", "NIKKEI": "^N225",   "DAX": "^GDAXI",
    "CAC 40": "^FCHI",   "ASX 200": "^AXJO",
}

HORIZON_DAYS = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 252}

# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/symbols")
async def get_symbols():
    """Return all available symbols grouped by market."""
    return {
        "forex":       [k for k in SYMBOL_MAP if "/" in k and k not in ["S&P 500"]],
        "commodities": ["GOLD","SILVER","CRUDE OIL","NAT GAS","COPPER","WHEAT","CORN","PLATINUM"],
        "stocks":      ["AAPL","MSFT","NVDA","AMZN","TSLA","GOOGL","META","JPM"],
        "index":       ["S&P 500","NASDAQ","DOW","FTSE 100","NIKKEI","DAX","CAC 40","ASX 200"],
    }


@app.get("/quote/{symbol:path}")
async def get_quote(symbol: str):
    """Get current quote + basic stats for a symbol."""
    ticker = SYMBOL_MAP.get(symbol.upper())
    if not ticker:
        raise HTTPException(404, f"Symbol '{symbol}' not found")
    try:
        data = await fetcher.get_quote(ticker, symbol)
        return data
    except Exception as e:
        logger.error(f"Quote error {symbol}: {e}")
        raise HTTPException(500, str(e))


@app.get("/history/{symbol:path}")
async def get_history(symbol: str, period: str = "6mo", interval: str = "1d"):
    """Get historical OHLCV data."""
    ticker = SYMBOL_MAP.get(symbol.upper())
    if not ticker:
        raise HTTPException(404, f"Symbol '{symbol}' not found")
    try:
        data = await fetcher.get_history(ticker, period, interval)
        return data
    except Exception as e:
        logger.error(f"History error {symbol}: {e}")
        raise HTTPException(500, str(e))


@app.get("/analyze/{symbol:path}")
async def analyze(
    symbol: str,
    horizon: str = "1M",
    models: str = "arima,garch,lstm,xgb,prophet,arima_lstm,kalman_xgb"
):
    """
    Full analysis: fetch data → fit all models → return forecasts + metrics.
    models param: comma-separated list of model ids.
    """
    ticker = SYMBOL_MAP.get(symbol.upper())
    if not ticker:
        raise HTTPException(404, f"Symbol '{symbol}' not found")

    steps = HORIZON_DAYS.get(horizon.upper(), 30)
    model_list = [m.strip() for m in models.split(",")]

    try:
        # 1. Fetch 2 years of daily history
        hist = await fetcher.get_history(ticker, "2y", "1d")
        prices = hist["close"]
        dates  = hist["dates"]

        if len(prices) < 60:
            raise HTTPException(400, "Not enough historical data (need ≥60 days)")

        # 2. Split train / validation (last 30 bars for validation)
        val_size = min(30, len(prices) // 5)
        train    = prices[:-val_size]
        val      = prices[-val_size:]

        results = {}

        # 3. Run each requested model
        if "arima" in model_list:
            try:
                m = ARIMAModel()
                m.fit(train)
                val_pred = m.predict(val_size)
                fc       = m.predict(steps, refit_on=prices)
                results["arima"] = build_result(m, "ARIMA", "stat", "#06b6d4",
                    "AutoRegressive Integrated Moving Average — statsmodels auto-order selection",
                    val, val_pred, fc, prices)
            except Exception as e:
                logger.warning(f"ARIMA failed: {e}")

        if "garch" in model_list:
            try:
                m = GARCHModel()
                m.fit(train)
                val_pred = m.predict_prices(val_size, train[-1])
                fc       = m.predict_prices(steps, prices[-1])
                results["garch"] = build_result(m, "GARCH(1,1)", "stat", "#22d3ee",
                    "Generalized ARCH — arch library, models conditional volatility",
                    val, val_pred, fc, prices)
            except Exception as e:
                logger.warning(f"GARCH failed: {e}")

        if "lstm" in model_list:
            try:
                m = LSTMModel()
                m.fit(train)
                val_pred = m.predict(val_size)
                fc       = m.predict(steps, refit_on=prices)
                results["lstm"] = build_result(m, "LSTM", "ml", "#8b5cf6",
                    "Long Short-Term Memory neural network — Keras, trained on price sequences",
                    val, val_pred, fc, prices)
            except Exception as e:
                logger.warning(f"LSTM failed: {e}")

        if "xgb" in model_list:
            try:
                m = XGBModel()
                m.fit(train)
                val_pred = m.predict(val_size)
                fc       = m.predict(steps, refit_on=prices)
                results["xgb"] = build_result(m, "XGBoost", "ml", "#a78bfa",
                    "Gradient Boosting — XGBoost with technical indicator features + SHAP",
                    val, val_pred, fc, prices)
            except Exception as e:
                logger.warning(f"XGBoost failed: {e}")

        if "prophet" in model_list:
            try:
                m = ProphetModel()
                m.fit(train, dates[:-val_size])
                val_pred = m.predict(val_size, dates[-val_size:])
                fc_dates = fetcher.future_dates(dates[-1], steps)
                fc       = m.predict(steps, fc_dates, refit_on=(prices, dates))
                results["prophet"] = build_result(m, "Prophet", "ml", "#c4b5fd",
                    "Facebook Prophet — piecewise linear trend + Fourier seasonality",
                    val, val_pred, fc, prices)
            except Exception as e:
                logger.warning(f"Prophet failed: {e}")

        if "arima_lstm" in model_list:
            try:
                m = HybridModel(mode="arima_lstm")
                m.fit(train)
                val_pred = m.predict(val_size)
                fc       = m.predict(steps, refit_on=prices)
                results["arima_lstm"] = build_result(m, "ARIMA+LSTM", "hybrid", "#f59e0b",
                    "ARIMA linear component + LSTM on residuals — true hybrid ensemble",
                    val, val_pred, fc, prices)
            except Exception as e:
                logger.warning(f"ARIMA+LSTM failed: {e}")

        if "kalman_xgb" in model_list:
            try:
                m = HybridModel(mode="kalman_xgb")
                m.fit(train)
                val_pred = m.predict(val_size)
                fc       = m.predict(steps, refit_on=prices)
                results["kalman_xgb"] = build_result(m, "Kalman+XGB", "hybrid", "#fbbf24",
                    "Kalman filter state-space denoising → XGBoost on cleaned signal",
                    val, val_pred, fc, prices)
            except Exception as e:
                logger.warning(f"Kalman+XGB failed: {e}")

        if not results:
            raise HTTPException(500, "All models failed — check logs")

        ensemble = EnsembleResult(results, prices, steps)

        return {
            "symbol":    symbol,
            "ticker":    ticker,
            "horizon":   horizon,
            "steps":     steps,
            "history":   {"prices": prices[-90:], "dates": dates[-90:]},
            "models":    results,
            "ensemble":  ensemble.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze error {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# ─── HELPER ────────────────────────────────────────────────────────────────────

def build_result(model_obj, name, mtype, color, desc, val, val_pred, forecast, full_prices):
    """Compute metrics and build standard result dict."""
    import numpy as np
    val      = list(val)
    val_pred = list(val_pred)
    forecast = list(forecast)
    n        = min(len(val), len(val_pred))

    if n == 0:
        rmse = mae = mape = r2 = 0.0
    else:
        v  = np.array(val[:n])
        p  = np.array(val_pred[:n])
        e  = v - p
        rmse = float(np.sqrt(np.mean(e**2)))
        mae  = float(np.mean(np.abs(e)))
        mape = float(np.mean(np.abs(e / (np.abs(v) + 1e-9))) * 100)

        ss_res = float(np.sum(e**2))
        ss_tot = float(np.sum((v - np.mean(v))**2))
        r2     = float(1 - ss_res / (ss_tot + 1e-9)) if ss_tot > 0 else 0.0
        r2     = max(0.0, min(1.0, r2))  # clamp [0,1]

    last = full_prices[-1]
    return {
        "name":     name,
        "type":     mtype,
        "color":    color,
        "desc":     desc,
        "metrics":  {"rmse": round(rmse,6), "mae": round(mae,6),
                     "mape": round(mape,4), "r2":  round(r2,4)},
        "forecast": [round(f, 6) for f in forecast],
        "val_pred": [round(v, 6) for v in val_pred],
        "direction": "bull" if (forecast[0] if forecast else last) > last else "bear",
    }
