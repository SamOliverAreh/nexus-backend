"""
Hybrid Models:
  1. ARIMA+LSTM  — ARIMA fits linear component, LSTM corrects residuals
  2. Kalman+XGB  — Kalman filter denoises prices, XGBoost forecasts clean signal
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from models.arima_model import ARIMAModel
from models.lstm_model  import LSTMModel
from models.xgb_model   import XGBModel


# ─── KALMAN FILTER ─────────────────────────────────────────────────────────────

def kalman_smooth(prices: np.ndarray,
                  process_var: float = 1e-4,
                  obs_var:     float = 1e-2) -> np.ndarray:
    """1-D Kalman filter (constant-velocity model)."""
    n    = len(prices)
    x    = prices[0]
    P    = 1.0
    smoothed = np.empty(n)

    for i, z in enumerate(prices):
        # Predict
        x_pred = x
        P_pred = P + process_var
        # Update
        K = P_pred / (P_pred + obs_var)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        smoothed[i] = x

    return smoothed


# ─── HYBRID BASE ───────────────────────────────────────────────────────────────

class HybridModel:
    """
    mode = 'arima_lstm' | 'kalman_xgb'
    """
    def __init__(self, mode: str = "arima_lstm"):
        self.mode    = mode
        self.train   = None
        self.fitted  = False

        if mode == "arima_lstm":
            self.arima = ARIMAModel()
            self.lstm  = LSTMModel()
        elif mode == "kalman_xgb":
            self.xgb   = XGBModel()
        else:
            raise ValueError(f"Unknown hybrid mode: {mode}")

    # ── fit ────────────────────────────────────────────────────────
    def fit(self, prices: list):
        self.train = np.array(prices, dtype=float)

        if self.mode == "arima_lstm":
            # 1. Fit ARIMA on prices
            self.arima.fit(prices)

            # 2. Compute ARIMA residuals (actual - fitted)
            fitted      = np.array(self.arima.fitted_values)
            n           = min(len(self.train), len(fitted))
            residuals   = self.train[:n] - fitted[:n]

            # 3. Fit LSTM on residuals to model non-linear pattern
            if len(residuals) >= 30:
                self.lstm.fit(residuals.tolist())

        elif self.mode == "kalman_xgb":
            # 1. Smooth prices with Kalman filter
            smoothed = kalman_smooth(self.train)
            # 2. Fit XGBoost on smoothed series
            self.xgb.fit(smoothed.tolist())
            self.smoothed_train = smoothed

        self.fitted = True

    # ── predict ────────────────────────────────────────────────────
    def predict(self, steps: int, refit_on: list = None) -> list:
        if refit_on is not None:
            self.fit(refit_on)

        if self.mode == "arima_lstm":
            # ARIMA linear forecast
            arima_fc = np.array(self.arima.predict(steps))

            # LSTM residual correction
            try:
                resid_fc = np.array(self.lstm.predict(steps))
                # Blend: 60% ARIMA + 40% LSTM-corrected
                combined = arima_fc + 0.4 * resid_fc
            except Exception:
                combined = arima_fc

            return [round(float(v), 6) for v in combined]

        elif self.mode == "kalman_xgb":
            smoothed = kalman_smooth(self.train)
            fc       = np.array(self.xgb.predict(steps, refit_on=smoothed.tolist()))
            return [round(float(v), 6) for v in fc]

    # ── R² ─────────────────────────────────────────────────────────
    def in_sample_r2(self) -> float:
        if not self.fitted:
            return 0.0
        if self.mode == "arima_lstm":
            # Average of both components
            r2_a = self.arima.in_sample_r2()
            r2_l = self.lstm.in_sample_r2()
            return float(np.clip((r2_a + r2_l) / 2, 0, 1))
        elif self.mode == "kalman_xgb":
            return float(np.clip(self.xgb.in_sample_r2(), 0, 1))
        return 0.0
