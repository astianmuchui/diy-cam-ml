from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class EnvironmentScore:
    overall: float
    factors: Dict[str, float]
    recommendation: str


_WEIGHTS = {
    "time":        0.30,
    "temperature": 0.25,
    "solar":       0.20,
    "wind":        0.15,
    "rain":        0.10,
}


def _time_score(ts: datetime) -> float:
    h = ts.hour + ts.minute / 60.0

    if 5.0 <= h < 7.0 or 17.5 <= h < 19.0:
        return 1.0
    if 7.0 <= h < 9.0 or 15.5 <= h < 17.5:
        return 0.7
    if 10.0 <= h < 14.0:
        return 0.3
    return 0.4


def _temp_score(temp_c: Optional[float]) -> float:
    if temp_c is None:
        return 0.5
    if 22.0 <= temp_c <= 28.0:
        return 1.0
    if (18.0 <= temp_c < 22.0) or (28.0 < temp_c <= 32.0):
        return 0.6
    return 0.2


def _solar_score(solar_w_m2: Optional[float]) -> float:
    if solar_w_m2 is None:
        return 0.5
    normalised = min(max((solar_w_m2 - 200.0) / 600.0, 0.0), 1.0)
    return 0.8 - normalised * 0.4


def _wind_score(wind_kmh: Optional[float]) -> float:
    if wind_kmh is None:
        return 0.5
    if 5.0 <= wind_kmh <= 15.0:
        return 0.8
    if wind_kmh < 5.0:
        return 0.5
    if wind_kmh > 25.0:
        return 0.3
    return 0.8 - ((wind_kmh - 15.0) / 10.0) * 0.5


def _rain_score(precip_mm: Optional[float]) -> float:
    if precip_mm is None:
        return 0.5
    if precip_mm == 0.0:
        return 0.5
    if precip_mm < 5.0:
        return 0.7
    return 0.3


def score_environment(weather: Optional[dict], timestamp: datetime) -> EnvironmentScore:
    w = weather or {}

    factors = {
        "time":        _time_score(timestamp),
        "temperature": _temp_score(w.get("temperature")),
        "solar":       _solar_score(w.get("solar_radiation")),
        "wind":        _wind_score(w.get("wind_speed")),
        "rain":        _rain_score(w.get("precipitation")),
    }

    overall = sum(factors[k] * _WEIGHTS[k] for k in factors)

    if overall >= 0.70:
        recommendation = "prime"
    elif overall >= 0.55:
        recommendation = "good"
    elif overall >= 0.40:
        recommendation = "moderate"
    else:
        recommendation = "poor"

    return EnvironmentScore(overall=overall, factors=factors, recommendation=recommendation)
