"""
XGBoost Model — xgboost library.
Features: lagged returns, MA, RSI, Bollinger, ATR, MACD.
Recursive multi-step forecasting.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class XGBModel:
    def __init__(self, n_lags: int = 10):
        self.n_lags  = n_lags
        self.model   = None
        self.scaler  = StandardScaler()
        self.train   = None
        self.fitted  = False

    # ── feature engineering ────────────────────────────────────────
    @staticmethod
    def _features(prices: np.ndarray, n_lags: int = 10) -> np.ndarray:
        feats = []
        n = len(prices)
        for i in range(n_lags, n):
            row = []
            window = prices[i - n_lags:i]

            # Lagged returns
            for lag in range(1, n_lags + 1):
                row.append((prices[i-lag] - prices[i-lag-1]) / (prices[i-lag-1] + 1e-9)
                           if i-lag-1 >= 0 else 0.0)

            # MAs
            for w in [5, 10, 20]:
                if i >= w:
                    row.append(prices[i-1] / np.mean(prices[i-w:i]) - 1)
                else:
                    row.append(0.0)

            # Volatility (std of returns)
            ret_w = np.diff(prices[max(0, i-20):i]) / (prices[max(0, i-20):i-1] + 1e-9)
            row.append(float(np.std(ret_w)) if len(ret_w) > 1 else 0.0)

            # RSI (14)
            if i >= 15:
                d = np.diff(prices[i-15:i])
                g = np.mean(d[d > 0]) if any(d > 0) else 0
                l = np.mean(-d[d < 0]) if any(d < 0) else 1e-9
                row.append(float(100 - 100 / (1 + g / l)))
            else:
                row.append(50.0)

            # Bollinger band position
            if i >= 20:
                mu  = np.mean(prices[i-20:i])
                sig = np.std(prices[i-20:i])
                row.append(float((prices[i-1] - mu) / (sig + 1e-9)))
            else:
                row.append(0.0)

            # MACD signal proxy
            if i >= 26:
                ema12 = np.mean(prices[i-12:i])
                ema26 = np.mean(prices[i-26:i])
                row.append(float((ema12 - ema26) / (ema26 + 1e-9)))
            else:
                row.append(0.0)

            feats.append(row)

        return np.array(feats, dtype=float)

    # ── fit ────────────────────────────────────────────────────────
    def fit(self, prices: list):
        self.train = np.array(prices, dtype=float)
        n = self.n_lags

        X = self._features(self.train, n)
        # Target = next-day return
        y = np.array([
            (self.train[i + 1] - self.train[i]) / (self.train[i] + 1e-9)
            for i in range(n, len(self.train) - 1)
        ])

        if len(X) > len(y):
            X = X[:len(y)]
        if len(y) > len(X):
            y = y[:len(X)]

        X_scaled = self.scaler.fit_transform(X)

        if XGB_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, random_state=42
            )

        self.model.fit(X_scaled, y)
        self.fitted = True

    # ── predict ────────────────────────────────────────────────────
    def predict(self, steps: int, refit_on: list = None) -> list:
        if refit_on is not None:
            self.fit(refit_on)

        buf   = list(self.train)
        preds = []

        for _ in range(steps):
            arr  = np.array(buf)
            X    = self._features(arr, self.n_lags)
            if len(X) == 0:
                preds.append(buf[-1]); continue
            xrow = self.scaler.transform(X[-1:])
            ret  = float(self.model.predict(xrow)[0])
            nxt  = buf[-1] * (1 + ret)
            preds.append(round(nxt, 6))
            buf.append(nxt)

        return preds

    def in_sample_r2(self) -> float:
        if not self.fitted:
            return 0.0
        n  = self.n_lags
        X  = self._features(self.train, n)
        y  = np.array([
            (self.train[i + 1] - self.train[i]) / (self.train[i] + 1e-9)
            for i in range(n, len(self.train) - 1)
        ])
        if len(X) > len(y): X = X[:len(y)]
        if len(y) > len(X): y = y[:len(X)]
        if len(X) == 0: return 0.0
        Xs   = self.scaler.transform(X)
        pred = self.model.predict(Xs)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(np.clip(1 - ss_res / (ss_tot + 1e-9), 0, 1))

    def feature_importance(self) -> dict:
        """Return named feature importances for SHAP-style display."""
        if not self.fitted or not hasattr(self.model, "feature_importances_"):
            return {}
        imp  = self.model.feature_importances_
        names = (
            [f"Lag_{i+1}_return" for i in range(self.n_lags)] +
            ["MA5_ratio", "MA10_ratio", "MA20_ratio",
             "Volatility", "RSI_14", "Bollinger_pos", "MACD_signal"]
        )
        names = names[:len(imp)]
        total = imp.sum() + 1e-9
        return {n: round(float(v / total * 100), 2) for n, v in zip(names, imp)}
