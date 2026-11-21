import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import redis

import config

rdb = redis.from_url(config.REDIS_URL, decode_responses=True)

_WINDOW_KEY = "corrector:window:{device_id}"
_PARAMS_KEY  = "corrector:params:{device_id}"


@dataclass
class CorrectionResult:
    corrected_depth: float
    baseline: float
    slope: float
    intercept: float
    r_squared: float
    window_size: int


def _linear_regression(tilts: list, depths: list):
    n = len(tilts)
    if n < 3:
        mean_d = float(np.mean(depths)) if depths else 12.0
        return 0.0, mean_d, 0.0

    t = np.array(tilts, dtype=float)
    d = np.array(depths, dtype=float)

    mean_t, mean_d = t.mean(), d.mean()
    num = ((t - mean_t) * (d - mean_d)).sum()
    den = ((t - mean_t) ** 2).sum()

    if den == 0:
        return 0.0, float(mean_d), 0.0

    slope     = float(num / den)
    intercept = float(mean_d - slope * mean_t)

    predicted = slope * t + intercept
    ss_res = float(((d - predicted) ** 2).sum())
    ss_tot = float(((d - mean_d) ** 2).sum())
    r_squared = max(0.0, 1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return slope, intercept, r_squared


def _compute_baseline(corr_depths: list) -> float:
    if not corr_depths:
        return 12.0
    sorted_d = sorted(corr_depths, reverse=True)
    top_n    = max(1, int(len(sorted_d) * 0.3))
    top      = sorted_d[:top_n]
    return sum(top) / len(top)


def correct(device_id: str, depth: float, tilt: float) -> CorrectionResult:
    window_key = _WINDOW_KEY.format(device_id=device_id)
    params_key = _PARAMS_KEY.format(device_id=device_id)

    rdb.rpush(window_key, json.dumps({"depth": depth, "tilt": tilt}))
    rdb.ltrim(window_key, -config.CORRECTION_WINDOW_SIZE, -1)

    raw_window = rdb.lrange(window_key, 0, -1)
    pairs      = [json.loads(p) for p in raw_window]

    tilts  = [p["tilt"]  for p in pairs]
    depths = [p["depth"] for p in pairs]

    slope, intercept, r_squared = _linear_regression(tilts, depths)

    predicted        = slope * tilt + intercept
    residual         = depth - predicted
    corrected_depth  = float(intercept + residual)

    corr_depths = [float(intercept + (p["depth"] - (slope * p["tilt"] + intercept))) for p in pairs]
    baseline    = _compute_baseline(corr_depths)

    rdb.hset(params_key, mapping={
        "slope":       slope,
        "intercept":   intercept,
        "r_squared":   r_squared,
        "baseline":    baseline,
        "window_size": len(pairs),
    })

    return CorrectionResult(
        corrected_depth=corrected_depth,
        baseline=baseline,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        window_size=len(pairs),
    )


def get_params(device_id: str) -> Optional[dict]:
    params_key = _PARAMS_KEY.format(device_id=device_id)
    raw = rdb.hgetall(params_key)
    if not raw:
        return None
    return {k: float(v) for k, v in raw.items()}


def get_window_size(device_id: str) -> int:
    window_key = _WINDOW_KEY.format(device_id=device_id)
    return rdb.llen(window_key)


class TiltCorrector:

    def __init__(self, window_size: int = 0):
        self._window: list = []
        self._max    = window_size or config.CORRECTION_WINDOW_SIZE

    def correct(self, depth: float, tilt: float) -> CorrectionResult:
        self._window.append({"depth": depth, "tilt": tilt})
        if len(self._window) > self._max:
            self._window = self._window[-self._max:]

        tilts  = [p["tilt"]  for p in self._window]
        depths = [p["depth"] for p in self._window]

        slope, intercept, r_squared = _linear_regression(tilts, depths)

        predicted       = slope * tilt + intercept
        corrected_depth = float(intercept + (depth - predicted))

        corr_depths = [
            float(intercept + (p["depth"] - (slope * p["tilt"] + intercept)))
            for p in self._window
        ]
        baseline = _compute_baseline(corr_depths)

        return CorrectionResult(
            corrected_depth=corrected_depth,
            baseline=baseline,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            window_size=len(self._window),
        )
