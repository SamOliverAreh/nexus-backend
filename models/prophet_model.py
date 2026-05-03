"""
Prophet Model — Facebook Prophet.
Handles trend changepoints + weekly/annual Fourier seasonality.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

import pandas as pd
from datetime import datetime, timedelta


class ProphetModel:
    def __init__(self):
        self.model  = None
        self.fitted = False
        self.last_date = None

    def fit(self, prices: list, dates: list):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed")

        df = pd.DataFrame({
            "ds": pd.to_datetime(dates),
            "y":  prices
        })

        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=0.1,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80,
        )
        self.model.fit(df)
        self.last_date = dates[-1]
        self.fitted    = True

    def predict(self, steps: int, future_dates: list = None,
                refit_on=None) -> list:
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed")

        if refit_on is not None:
            prices, dates = refit_on
            self.fit(prices, dates)

        if future_dates:
            fd = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
        else:
            fd = self.model.make_future_dataframe(periods=steps, freq="B")
            fd = fd.tail(steps)

        fc = self.model.predict(fd)
        return [round(float(v), 6) for v in fc["yhat"].tolist()]

    def in_sample_r2(self, prices: list, dates: list) -> float:
        if not self.fitted:
            return 0.0
        df     = pd.DataFrame({"ds": pd.to_datetime(dates)})
        fc     = self.model.predict(df)
        y_pred = fc["yhat"].values
        y_true = np.array(prices)
        n      = min(len(y_true), len(y_pred))
        ss_res = np.sum((y_true[:n] - y_pred[:n]) ** 2)
        ss_tot = np.sum((y_true[:n] - np.mean(y_true[:n])) ** 2)
        return float(np.clip(1 - ss_res / (ss_tot + 1e-9), 0, 1))
