"""
LSTM Model — Keras/TensorFlow.
Trains on sliding windows of normalised prices.
Retrains on full data before final forecast.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Try TF first, fall back to sklearn MLPRegressor if TF unavailable
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


LOOKBACK = 20      # sequence length
EPOCHS   = 40
BATCH    = 16


class LSTMModel:
    def __init__(self):
        self.model    = None
        self.scaler_min = None
        self.scaler_rng = None
        self.train    = None
        self.lookback = LOOKBACK
        self.fitted   = False

    # ── normalisation ──────────────────────────────────────────────
    def _scale(self, arr):
        return (arr - self.scaler_min) / (self.scaler_rng + 1e-9)

    def _unscale(self, arr):
        return arr * (self.scaler_rng + 1e-9) + self.scaler_min

    # ── build sequences ────────────────────────────────────────────
    def _make_sequences(self, prices_norm):
        X, y = [], []
        for i in range(self.lookback, len(prices_norm)):
            X.append(prices_norm[i - self.lookback:i])
            y.append(prices_norm[i])
        return np.array(X)[..., np.newaxis], np.array(y)

    # ── fit ────────────────────────────────────────────────────────
    def fit(self, prices: list):
        self.train      = np.array(prices, dtype=float)
        self.scaler_min = self.train.min()
        self.scaler_rng = self.train.max() - self.train.min()

        norm = self._scale(self.train)
        X, y = self._make_sequences(norm)

        if TF_AVAILABLE and len(X) >= 10:
            self.model = self._build_keras()
            es = EarlyStopping(monitor="loss", patience=5,
                               restore_best_weights=True, verbose=0)
            self.model.fit(X, y, epochs=EPOCHS, batch_size=BATCH,
                           callbacks=[es], verbose=0)
        else:
            # Fallback: simple linear regression on flattened sequences
            from sklearn.linear_model import Ridge
            Xf = X.reshape(len(X), -1)
            self.model = Ridge(alpha=1.0)
            self.model.fit(Xf, y)

        self.fitted = True

    def _build_keras(self):
        m = Sequential([
            LSTM(64, return_sequences=True,
                 input_shape=(self.lookback, 1)),
            Dropout(0.1),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
        return m

    # ── predict ────────────────────────────────────────────────────
    def predict(self, steps: int, refit_on: list = None) -> list:
        if refit_on is not None:
            self.fit(refit_on)

        prices_norm = self._scale(self.train)
        seq         = list(prices_norm[-self.lookback:])
        preds_norm  = []

        for _ in range(steps):
            x = np.array(seq[-self.lookback:])
            if TF_AVAILABLE and hasattr(self.model, "predict"):
                p = float(self.model.predict(
                    x[np.newaxis, :, np.newaxis], verbose=0)[0][0])
            else:
                p = float(self.model.predict(x.reshape(1, -1))[0])
            preds_norm.append(p)
            seq.append(p)

        preds = self._unscale(np.array(preds_norm))
        return [round(float(v), 6) for v in preds]

    def in_sample_r2(self) -> float:
        if not self.fitted:
            return 0.0
        norm       = self._scale(self.train)
        X, y_true  = self._make_sequences(norm)
        if len(X) == 0:
            return 0.0

        if TF_AVAILABLE and hasattr(self.model, "predict"):
            y_pred = self.model.predict(X, verbose=0).flatten()
        else:
            y_pred = self.model.predict(X.reshape(len(X), -1))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(np.clip(1 - ss_res / (ss_tot + 1e-9), 0, 1))
