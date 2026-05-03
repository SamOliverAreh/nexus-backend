"""
Ensemble — aggregates all model outputs into consensus signal,
confidence intervals, and median price targets.
"""

import numpy as np
from typing import Dict


class EnsembleResult:
    def __init__(self, results: Dict, prices: list, steps: int):
        self.results = results
        self.prices  = np.array(prices)
        self.steps   = steps
        self.last    = float(prices[-1])

    def _forecasts_at(self, step: int) -> np.ndarray:
        vals = []
        for r in self.results.values():
            fc = r.get("forecast", [])
            if len(fc) > step:
                vals.append(fc[step])
        return np.array(vals)

    def median_forecast(self) -> list:
        meds = []
        for i in range(self.steps):
            v = self._forecasts_at(i)
            meds.append(float(np.median(v)) if len(v) else self.last)
        return [round(v, 6) for v in meds]

    def confidence_bands(self):
        uppers, lowers = [], []
        for i in range(self.steps):
            v = self._forecasts_at(i)
            if len(v) == 0:
                uppers.append(self.last); lowers.append(self.last)
            else:
                uppers.append(float(np.percentile(v, 84)))
                lowers.append(float(np.percentile(v, 16)))
        return (
            [round(v, 6) for v in uppers],
            [round(v, 6) for v in lowers],
        )

    def consensus(self) -> dict:
        vals     = list(self.results.values())
        total    = len(vals)
        if total == 0:
            return {"score": 50, "direction": "neutral", "bullish": 0, "bearish": 0}

        bullish  = sum(1 for r in vals if (r["forecast"][0] if r["forecast"] else self.last) > self.last)
        bearish  = total - bullish
        score    = round(bullish / total * 100)
        direction = "bullish" if score > 55 else "bearish" if score < 45 else "neutral"
        agreement = round(max(bullish, bearish) / total * 100)
        return {
            "score":     score,
            "direction": direction,
            "bullish":   bullish,
            "bearish":   bearish,
            "total":     total,
            "agreement": agreement,
        }

    def price_targets(self) -> dict:
        steps_map = {"d1": 0, "d7": 6, "d30": 29}
        targets   = {}
        for k, s in steps_map.items():
            v = self._forecasts_at(min(s, self.steps - 1))
            targets[k] = round(float(np.median(v)), 6) if len(v) else self.last
        return targets

    def best_model(self) -> str:
        best, best_r2 = "", -1.0
        for name, r in self.results.items():
            r2 = r.get("metrics", {}).get("r2", 0)
            if r2 > best_r2:
                best_r2 = r2
                best    = r.get("name", name)
        return best

    def to_dict(self) -> dict:
        upper, lower = self.confidence_bands()
        return {
            "median_forecast": self.median_forecast(),
            "upper_band":      upper,
            "lower_band":      lower,
            "consensus":       self.consensus(),
            "price_targets":   self.price_targets(),
            "best_model":      self.best_model(),
        }
