import json
import math
from datetime import datetime, timezone
from typing import Optional

import redis

import config

rdb = redis.from_url(config.REDIS_URL, decode_responses=True)

_GEO_KEY  = "catchlog:geo"
_DATA_KEY = "catchlog:data:{id}"

_SPECIES_PRIORS: dict = {
    "tilapia": {
        "peak_hours":    [(5, 7), (17, 19)],
        "depth_pref_m":  (2.0, 8.0),
        "base_prob":     0.70,
    },
    "catfish": {
        "peak_hours":    [(18, 24), (0, 6)],
        "depth_pref_m":  (8.0, 14.0),
        "base_prob":     0.25,
    },
    "bass": {
        "peak_hours":    [(6, 9), (16, 19)],
        "depth_pref_m":  (4.0, 10.0),
        "base_prob":     0.05,
    },
}

DEFAULT_RADIUS_M: float = 2000.0
DISTANCE_HALFLIFE_M: float = 500.0
RECENCY_HALFLIFE_DAYS: float = 30.0
DEPTH_MATCH_BOOST:   float = 1.2
DEPTH_MISMATCH_DAMP: float = 0.6


def ingest_catch_log(entry: dict) -> None:
    entry_id  = entry.get("id", "")
    lat       = float(entry.get("latitude", 0.0))
    lon       = float(entry.get("longitude", 0.0))
    fish_type = str(entry.get("fish_type", "")).lower().strip()
    count     = int(entry.get("count", 1))
    caught_at = entry.get("caught_at")

    if not entry_id or fish_type in ("", "unknown"):
        return

    if isinstance(caught_at, datetime):
        caught_at = caught_at.isoformat()
    elif not isinstance(caught_at, str):
        caught_at = datetime.now(tz=timezone.utc).isoformat()

    rdb.geoadd(_GEO_KEY, [lon, lat, entry_id])
    rdb.hset(_DATA_KEY.format(id=entry_id), mapping={
        "fish_type":  fish_type,
        "count":      count,
        "caught_at":  caught_at,
        "latitude":   lat,
        "longitude":  lon,
    })


def _time_affinity(caught_at_str: str, species: str) -> float:
    try:
        dt   = datetime.fromisoformat(caught_at_str.replace("Z", "+00:00"))
        hour = dt.hour + dt.minute / 60.0
    except ValueError:
        return 0.5

    for (start, end) in _SPECIES_PRIORS.get(species, {}).get("peak_hours", []):
        if start <= end:
            if start <= hour < end:
                return 1.0
        else:
            if hour >= start or hour < end:
                return 1.0
    return 0.4


def _depth_weight(depth_m: Optional[float], species: str) -> float:
    if depth_m is None:
        return 1.0
    d_min, d_max = _SPECIES_PRIORS.get(species, {}).get("depth_pref_m", (0.0, 20.0))
    return DEPTH_MATCH_BOOST if d_min <= depth_m <= d_max else DEPTH_MISMATCH_DAMP


def predict_species(
    lat:      float,
    lon:      float,
    timestamp: datetime,
    radius_m: float = DEFAULT_RADIUS_M,
    depth_m:  Optional[float] = None,
) -> dict:
    try:
        nearby = rdb.geosearch(
            _GEO_KEY,
            longitude=lon,
            latitude=lat,
            radius=radius_m,
            unit="m",
            sort="ASC",
            count=50,
            withdist=True,
            withcoord=True,
        )
    except Exception:
        nearby = []

    if not nearby:
        return _prior_prediction(timestamp, depth_m)

    now    = datetime.now(tz=timezone.utc)
    votes: dict = {}

    for item in nearby:
        entry_id = item[0]
        dist_m   = float(item[1])
        data     = rdb.hgetall(_DATA_KEY.format(id=entry_id))
        if not data:
            continue

        fish_type = data.get("fish_type", "unknown")
        if fish_type == "unknown":
            continue

        count     = float(data.get("count", 1))
        caught_at = data.get("caught_at", "")

        try:
            caught_dt = datetime.fromisoformat(caught_at.replace("Z", "+00:00"))
            days_ago  = max(0.0, (now - caught_dt).total_seconds() / 86400.0)
        except ValueError:
            days_ago = RECENCY_HALFLIFE_DAYS

        recency_w = math.exp(-days_ago  / RECENCY_HALFLIFE_DAYS)
        dist_w    = math.exp(-dist_m    / DISTANCE_HALFLIFE_M)
        time_w    = _time_affinity(caught_at, fish_type)
        depth_w   = _depth_weight(depth_m, fish_type)

        votes[fish_type] = votes.get(fish_type, 0.0) + count * recency_w * dist_w * time_w * depth_w

    if not votes:
        return _prior_prediction(timestamp, depth_m)

    total  = sum(votes.values())
    normed = {k: round(v / total, 4) for k, v in sorted(votes.items(), key=lambda x: -x[1])}
    best   = next(iter(normed))

    return {
        "species":     best,
        "confidence":  normed[best],
        "votes":       normed,
        "sample_size": len(nearby),
    }


def _prior_prediction(timestamp: datetime, depth_m: Optional[float]) -> dict:
    hour  = timestamp.hour + timestamp.minute / 60.0
    votes = {}

    for species, prior in _SPECIES_PRIORS.items():
        score = prior["base_prob"]

        for (start, end) in prior.get("peak_hours", []):
            if (start <= end and start <= hour < end) or \
               (start > end and (hour >= start or hour < end)):
                score *= 1.4
                break

        score *= _depth_weight(depth_m, species)
        votes[species] = score

    total  = sum(votes.values())
    normed = {k: round(v / total, 4) for k, v in sorted(votes.items(), key=lambda x: -x[1])}
    best   = next(iter(normed))

    return {
        "species":     best,
        "confidence":  round(normed[best], 4),
        "votes":       normed,
        "sample_size": 0,
    }


def spatial_fish_map(
    points:    list,
    timestamp: datetime,
    radius_m:  float = 1000.0,
) -> list:
    results = []
    for pt in points:
        pred = predict_species(pt["lat"], pt["lon"], timestamp, radius_m=radius_m)
        results.append({
            "lat":         pt["lat"],
            "lon":         pt["lon"],
            "species":     pred["species"],
            "confidence":  pred["confidence"],
            "sample_size": pred["sample_size"],
        })
    return results
