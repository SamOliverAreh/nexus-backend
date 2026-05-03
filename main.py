"""
NEXUS Market Analysis Engine — FastAPI Backend
Endpoints: /quote, /history, /analyze, /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

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

app = FastAPI(title="NEXUS Market Analysis API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fetcher = DataFetcher()

# ─── SYMBOL MAP ─────────────────────────────────────────────────────────────
SYMBOL_MAP = {
    # Forex
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "EUR/GBP": "EURGBP=X",
    "USD/CHF": "USDCHF=X", "NZD/USD": "NZDUSD=X",
    # Commodities — continuous futures
    "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE OIL": "CL=F",
    "NAT GAS": "NG=F", "COPPER": "HG=F", "WHEAT": "ZW=F",
    "CORN": "ZC=F", "PLATINUM": "PL=F",
    # Stocks
    "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "AMZN": "AMZN",
    "TSLA": "TSLA", "GOOGL": "GOOGL", "META": "META", "JPM": "JPM",
    # Indices
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI",
    "FTSE 100": "^FTSE", "NIKKEI": "^N225", "DAX": "^GDAXI",
    "CAC 40": "^FCHI", "ASX 200": "^AXJO",
}

# ETF fallbacks when futures contract is temporarily unavailable
TICKER_FALLBACKS = {
    "ZC=F": "CORN",  # Teucrium Corn ETF
    "ZW=F": "WEAT",  # Teucrium Wheat ETF
    "SI=F": "SLV",   # iShares Silver ETF
    "PL=F": "PPLT",  # Aberdeen Platinum ETF
    "HG=F": "CPER",  # US Copper Index ETF
    "NG=F": "UNG",   # US Natural Gas ETF
}

HORIZON_DAYS = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 252}


# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/symbols")
async def get_symbols():
    return {
        "forex":       ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","EUR/GBP","USD/CHF","NZD/USD"],
        "commodities": ["GOLD","SILVER","CRUDE OIL","NAT GAS","COPPER","WHEAT","CORN","PLATINUM"],
        "stocks":      ["AAPL","MSFT","NVDA","AMZN","TSLA","GOOGL","META","JPM"],
        "index":       ["S&P 500","NASDAQ","DOW","FTSE 100","NIKKEI","DAX","CAC 40","ASX 200"],
    }


@app.get("/quote/{symbol:path}")
async def get_quote(symbol: str):
    ticker = SYMBOL_MAP.get(symbol.upper())
    if not ticker:
        raise HTTPException(404, f"Symbol '{symbol}' not found")
    try:
        return await fetcher.get_quote(ticker, symbol, TICKER_FALLBACKS)
    except Exception as e:
        logger.error(f"Quote error {symbol}: {e}")
        raise HTTPException(500, str(e))


@app.get("/history/{symbol:path}")
async def get_history(symbol: str, period: str = "2y", interval: str = "1d"):
    ticker = SYMBOL_MAP.get(symbol.upper())
    if not ticker:
        raise HTTPException(404, f"Symbol '{symbol}' not found")
    try:
        return await fetcher.get_history(ticker, period, interval, TICKER_FALLBACKS)
    except Exception as e:
        logger.error(f"History error {symbol}: {e}")
        raise HTTPException(500, str(e))


@app.get("/analyze/{symbol:path}")
async def analyze(
    symbol: str,
    horizon: str = "1M",
    models: str = "arima,garch,lstm,xgb,prophet,arima_lstm,kalman_xgb"
):
    ticker = SYMBOL_MAP.get(symbol.upper())
    if not ticker:
        raise HTTPException(404, f"Symbol '{symbol}' not found")

    steps      = HORIZON_DAYS.get(horizon.upper(), 30)
    model_list = [m.strip() for m in models.split(",")]

    try:
        hist   = await fetcher.get_history(ticker, "2y", "1d", TICKER_FALLBACKS)
        prices = hist["close"]
        dates  = hist["dates"]

        if len(prices) < 60:
            raise HTTPException(400, f"Not enough data for {symbol} ({len(prices)} bars). Try a different asset.")

        val_size = max(20, min(60, len(prices) // 5))
        train    = prices[:-val_size]
        val      = prices[-val_size:]

        results = {}

        if "arima" in model_list:
            try:
                m = ARIMAModel(); m.fit(train)
                results["arima"] = build_result(
                    "ARIMA", "stat", "#0891b2",
                    "Auto-order ARIMA via AIC grid search — statsmodels",
                    val, m.predict(val_size), m.predict(steps, refit_on=prices), prices)
            except Exception as e:
                logger.warning(f"ARIMA failed: {e}")

        if "garch" in model_list:
            try:
                m = GARCHModel(); m.fit(train)
                results["garch"] = build_result(
                    "GARCH(1,1)", "stat", "#06b6d4",
                    "Conditional volatility — arch library GARCH(1,1)",
                    val, m.predict_prices(val_size, train[-1]),
                    m.predict_prices(steps, prices[-1]), prices)
            except Exception as e:
                logger.warning(f"GARCH failed: {e}")

        if "lstm" in model_list:
            try:
                m = LSTMModel(); m.fit(train)
                results["lstm"] = build_result(
                    "LSTM", "ml", "#7c3aed",
                    "Keras LSTM 64→32 units trained on normalised price sequences",
                    val, m.predict(val_size), m.predict(steps, refit_on=prices), prices)
            except Exception as e:
                logger.warning(f"LSTM failed: {e}")

        if "xgb" in model_list:
            try:
                m = XGBModel(); m.fit(train)
                r = build_result(
                    "XGBoost", "ml", "#9333ea",
                    "XGBoost with 15 technical indicator features + SHAP attribution",
                    val, m.predict(val_size), m.predict(steps, refit_on=prices), prices)
                if hasattr(m, "feature_importance"):
                    r["feature_importance"] = m.feature_importance()
                results["xgb"] = r
            except Exception as e:
                logger.warning(f"XGBoost failed: {e}")

        if "prophet" in model_list:
            try:
                m = ProphetModel(); m.fit(train, dates[:-val_size])
                fc_dates = fetcher.future_dates(dates[-1], steps)
                results["prophet"] = build_result(
                    "Prophet", "ml", "#a855f7",
                    "Facebook Prophet — piecewise linear trend + Fourier seasonality",
                    val, m.predict(val_size, dates[-val_size:]),
                    m.predict(steps, fc_dates, refit_on=(prices, dates)), prices)
            except Exception as e:
                logger.warning(f"Prophet failed: {e}")

        if "arima_lstm" in model_list:
            try:
                m = HybridModel(mode="arima_lstm"); m.fit(train)
                results["arima_lstm"] = build_result(
                    "ARIMA+LSTM", "hybrid", "#d97706",
                    "ARIMA linear component + LSTM corrects residuals — true hybrid",
                    val, m.predict(val_size), m.predict(steps, refit_on=prices), prices)
            except Exception as e:
                logger.warning(f"ARIMA+LSTM failed: {e}")

        if "kalman_xgb" in model_list:
            try:
                m = HybridModel(mode="kalman_xgb"); m.fit(train)
                results["kalman_xgb"] = build_result(
                    "Kalman+XGB", "hybrid", "#f59e0b",
                    "Kalman state-space filter denoising → XGBoost on clean signal",
                    val, m.predict(val_size), m.predict(steps, refit_on=prices), prices)
            except Exception as e:
                logger.warning(f"Kalman+XGB failed: {e}")

        if not results:
            raise HTTPException(500, "All models failed — check server logs")

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


# ─── HELPER ─────────────────────────────────────────────────────────────────

def build_result(name, mtype, color, desc, val, val_pred, forecast, full_prices):
    import numpy as np
    val = list(val); val_pred = list(val_pred); forecast = list(forecast)
    n = min(len(val), len(val_pred))

    if n == 0:
        rmse = mae = mape = r2 = 0.0
    else:
        v = np.array(val[:n], dtype=float)
        p = np.array(val_pred[:n], dtype=float)
        e = v - p
        rmse   = float(np.sqrt(np.mean(e**2)))
        mae    = float(np.mean(np.abs(e)))
        mape   = float(np.mean(np.abs(e / (np.abs(v) + 1e-9))) * 100)
        ss_res = float(np.sum(e**2))
        ss_tot = float(np.sum((v - np.mean(v))**2))
        r2     = float(np.clip(1 - ss_res / (ss_tot + 1e-9), 0.0, 1.0))

    last = full_prices[-1]
    return {
        "name": name, "type": mtype, "color": color, "desc": desc,
        "metrics": {
            "rmse": round(rmse, 6), "mae": round(mae, 6),
            "mape": round(mape, 4), "r2":  round(r2, 4),
        },
        "forecast":  [round(float(f), 6) for f in forecast],
        "val_pred":  [round(float(v), 6) for v in val_pred],
        "direction": "bull" if (forecast[0] if forecast else last) > last else "bear",
    }
