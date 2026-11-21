import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import config
from detection.environment import score_environment
from detection.species import (
    _GEO_KEY,
    _DATA_KEY,
    _SPECIES_PRIORS,
    predict_species,
    rdb,
)

_HORIZON_HOURS = config.INSIGHTS_HORIZON_HOURS
_TREND_DAYS    = config.INSIGHTS_TREND_DAYS


def _hourly_scores(weather: Optional[dict], start: datetime, hours: int) -> list:
    out = []
    for h in range(hours):
        ts  = start + timedelta(hours=h)
        env = score_environment(weather, ts)
        active = [
            sp for sp, prior in _SPECIES_PRIORS.items()
            if _in_peak(ts.hour + ts.minute / 60.0, prior.get("peak_hours", []))
        ]
        out.append({
            "ts":             ts,
            "score":          round(env.overall, 4),
            "recommendation": env.recommendation,
            "species_active": active,
        })
    return out


def _in_peak(hour: float, windows: list) -> bool:
    for (s, e) in windows:
        if (s <= e and s <= hour < e) or (s > e and (hour >= s or hour < e)):
            return True
    return False


def _best_time_windows(hourly: list, top_n: int = 3) -> list:
    windows = []
    run: list = []

    def flush():
        if not run:
            return
        scores = [r["score"] for r in run]
        species = sorted({sp for r in run for sp in r["species_active"]})
        windows.append({
            "start":           run[0]["ts"].isoformat(),
            "end":             (run[-1]["ts"] + timedelta(hours=1)).isoformat(),
            "avg_score":       round(sum(scores) / len(scores), 4),
            "peak_score":      round(max(scores), 4),
            "recommendation":  _label(max(scores)),
            "species_active":  species,
        })

    for row in hourly:
        if row["recommendation"] in ("prime", "good"):
            run.append(row)
        else:
            flush()
            run = []
    flush()

    windows.sort(key=lambda w: w["peak_score"], reverse=True)
    return windows[:top_n]


def _label(score: float) -> str:
    if score >= 0.70:
        return "prime"
    if score >= 0.55:
        return "good"
    if score >= 0.40:
        return "moderate"
    return "poor"


def _collect_catch_logs(lat: float, lon: float, radius_m: float) -> list:
    try:
        nearby = rdb.geosearch(
            _GEO_KEY, longitude=lon, latitude=lat,
            radius=radius_m, unit="m", sort="ASC", count=500,
            withdist=True, withcoord=True,
        )
    except Exception:
        return []

    logs = []
    for item in nearby:
        data = rdb.hgetall(_DATA_KEY.format(id=item[0]))
        if not data:
            continue
        try:
            logs.append({
                "fish_type": data.get("fish_type", "unknown"),
                "count":     float(data.get("count", 1)),
                "lat":       float(data.get("latitude", 0.0)),
                "lon":       float(data.get("longitude", 0.0)),
                "caught_at": data.get("caught_at", ""),
                "dist_m":    float(item[1]),
            })
        except (TypeError, ValueError):
            continue
    return logs


def _best_locations(logs: list, now: datetime, top_n: int = 5) -> list:
    cells: dict = defaultdict(lambda: {
        "lat_sum": 0.0, "lon_sum": 0.0, "n": 0,
        "weight": 0.0, "catches": 0.0, "species": defaultdict(float),
    })

    for lg in logs:
        try:
            caught = datetime.fromisoformat(lg["caught_at"].replace("Z", "+00:00"))
            days   = max(0.0, (now - caught).total_seconds() / 86400.0)
        except (ValueError, AttributeError):
            days = 30.0
        recency = math.exp(-days / 30.0)
        w       = lg["count"] * recency

        key = (round(lg["lat"], 3), round(lg["lon"], 3))
        c   = cells[key]
        c["lat_sum"] += lg["lat"]
        c["lon_sum"] += lg["lon"]
        c["n"]       += 1
        c["weight"]  += w
        c["catches"] += lg["count"]
        if lg["fish_type"] != "unknown":
            c["species"][lg["fish_type"]] += w

    ranked = []
    for c in cells.values():
        dominant = max(c["species"].items(), key=lambda x: x[1])[0] if c["species"] else "unknown"
        ranked.append({
            "lat":              round(c["lat_sum"] / c["n"], 6),
            "lon":              round(c["lon_sum"] / c["n"], 6),
            "total_catches":    int(c["catches"]),
            "recent_weight":    round(c["weight"], 4),
            "dominant_species": dominant,
        })

    ranked.sort(key=lambda x: x["recent_weight"], reverse=True)
    return ranked[:top_n]


def _catch_trend(logs: list, now: datetime) -> dict:
    cutoff = now - timedelta(days=_TREND_DAYS)
    mid    = now - timedelta(days=_TREND_DAYS / 2)
    recent = older = 0.0

    for lg in logs:
        try:
            caught = datetime.fromisoformat(lg["caught_at"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if caught < cutoff:
            continue
        if caught >= mid:
            recent += lg["count"]
        else:
            older += lg["count"]

    if recent + older == 0:
        return {"direction": "unknown", "recent": 0, "previous": 0, "change_pct": 0.0}

    change = 100.0 if older == 0 else round((recent - older) / older * 100.0, 1)
    if recent > older * 1.15:
        direction = "improving"
    elif recent < older * 0.85:
        direction = "declining"
    else:
        direction = "stable"
    return {
        "direction":  direction,
        "recent":     int(recent),
        "previous":   int(older),
        "change_pct": change,
    }


def build_insights(
    lat:           float,
    lon:           float,
    timestamp:     Optional[datetime] = None,
    weather:       Optional[dict] = None,
    radius_m:      float = 2000.0,
    horizon_hours: Optional[int] = None,
) -> dict:
    now    = timestamp or datetime.now(tz=timezone.utc)
    hours  = horizon_hours or _HORIZON_HOURS

    hourly        = _hourly_scores(weather, now, hours)
    time_windows  = _best_time_windows(hourly)
    logs          = _collect_catch_logs(lat, lon, radius_m)
    locations     = _best_locations(logs, now)
    trend         = _catch_trend(logs, now)

    species_now = predict_species(lat, lon, now, radius_m=radius_m)
    species_at_peak = None
    if time_windows:
        peak_dt = datetime.fromisoformat(time_windows[0]["start"])
        species_at_peak = predict_species(lat, lon, peak_dt, radius_m=radius_m)

    env_now = score_environment(weather, now)
    advisory = _advisory(env_now, trend, time_windows)

    return {
        "generated_at":   now.isoformat(),
        "location":       {"lat": lat, "lon": lon, "radius_m": radius_m},
        "current_environment": {
            "overall":        round(env_now.overall, 4),
            "recommendation": env_now.recommendation,
            "factors":        {k: round(v, 4) for k, v in env_now.factors.items()},
        },
        "best_time_windows": time_windows,
        "best_locations":    locations,
        "expected_species": {
            "now":             species_now,
            "at_peak_window":  species_at_peak,
        },
        "catch_trend":   trend,
        "advisory":      advisory,
        "sample_size":   len(logs),
    }


def _advisory(env, trend: dict, windows: list) -> str:
    parts = []

    rec = env.recommendation
    if rec == "prime":
        parts.append("Conditions are prime right now — get on the water.")
    elif rec == "good":
        parts.append("Good fishing conditions currently.")
    elif rec == "moderate":
        parts.append("Conditions are moderate; expect slower bites.")
    else:
        parts.append("Poor conditions now — consider waiting for a better window.")

    if windows:
        w = windows[0]
        parts.append(
            f"Best window ahead: {w['start'][11:16]}–{w['end'][11:16]} "
            f"({w['recommendation']}, score {w['peak_score']:.2f})."
        )

    d = trend["direction"]
    if d == "improving":
        parts.append(f"Catch rate is improving (+{trend['change_pct']:.0f}% vs prior period).")
    elif d == "declining":
        parts.append(f"Catch rate is declining ({trend['change_pct']:.0f}% vs prior period).")
    elif d == "stable":
        parts.append("Catch rate is stable versus the prior period.")

    return " ".join(parts)
