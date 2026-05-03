"""
GARCH(1,1) Model — arch library.
Models conditional volatility + produces price path forecasts.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from arch import arch_model


class GARCHModel:
    def __init__(self):
        self.result   = None
        self.train    = None
        self.returns  = None

    def fit(self, prices: list):
        self.train   = np.array(prices, dtype=float)
        # Convert to percentage returns for numerical stability
        self.returns = np.diff(np.log(self.train)) * 100
        am = arch_model(self.returns, vol="Garch", p=1, q=1,
                        dist="normal", rescale=False)
        self.result = am.fit(disp="off", show_warning=False)

    def predict_prices(self, steps: int, last_price: float) -> list:
        """
        Simulate price paths using GARCH variance forecast.
        Returns mean simulated path.
        """
        if self.result is None:
            raise RuntimeError("Model not fitted")

        # Get GARCH variance forecasts
        fc      = self.result.forecast(horizon=steps, reindex=False)
        var_fc  = fc.variance.values[-1]          # shape (steps,)
        mu_ret  = float(self.result.params.get("mu", 0))

        # Build price path from expected return + vol
        prices = []
        p = float(last_price)
        for i in range(steps):
            sig   = float(np.sqrt(max(var_fc[i], 1e-8))) / 100   # back to decimal
            ret   = mu_ret / 100 + sig * np.random.randn() * 0.05  # damped noise
            p     = p * np.exp(ret)
            prices.append(round(p, 6))
        return prices

    def in_sample_r2(self) -> float:
        """Standardised residuals R² proxy."""
        if self.result is None:
            return 0.0
        resid     = self.result.std_resid
        ss_res    = float(np.sum(resid ** 2))
        ss_tot    = float(len(resid))   # standardised → variance ≈ 1
        return float(np.clip(1 - ss_res / (ss_tot + 1e-9), 0, 1))
