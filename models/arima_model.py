"""
ARIMA Model — statsmodels auto_arima order selection.
Fits on real price history, computes proper in-sample R².
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def _best_arima_order(series: np.ndarray):
    """Select (p,d,q) by AIC search over small grid."""
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    m = ARIMA(series, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order


class ARIMAModel:
    def __init__(self):
        self.model    = None
        self.result   = None
        self.order    = None
        self.train    = None
        self.fitted_values = None

    def fit(self, prices: list):
        self.train = np.array(prices, dtype=float)
        self.order = _best_arima_order(self.train)
        self.result = ARIMA(self.train, order=self.order).fit()
        self.fitted_values = self.result.fittedvalues.tolist()

    def predict(self, steps: int, refit_on: list = None) -> list:
        """Forecast `steps` ahead. Optionally refit on full series first."""
        if refit_on is not None:
            series = np.array(refit_on, dtype=float)
            res = ARIMA(series, order=self.order).fit()
        else:
            res = self.result
        fc = res.forecast(steps=steps)
        return list(np.array(fc).flatten())

    def in_sample_r2(self) -> float:
        """True in-sample R² from fitted model."""
        if self.result is None:
            return 0.0
        actual    = self.train
        predicted = np.array(self.fitted_values)
        n = min(len(actual), len(predicted))
        if n == 0:
            return 0.0
        a, p = actual[:n], predicted[:n]
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        return float(np.clip(1 - ss_res / (ss_tot + 1e-9), 0, 1))
