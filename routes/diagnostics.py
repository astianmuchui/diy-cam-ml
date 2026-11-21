import json
from datetime import datetime, timezone, timedelta

import redis
from flask import Blueprint, jsonify, request

import config
from detection.correction import get_params
from detection.environment import score_environment
from detection.vision import model_status as cnn_model_status

diagnostics_bp = Blueprint("diagnostics", __name__)

rdb = redis.from_url(config.REDIS_URL, decode_responses=True)

_CACHE_KEY     = "flask:diagnostics"
_HISTORY_KEY   = "detection:history:{device_id}"
_REJECTION_KEY = "gate:rejections:{device_id}"


def _list_devices() -> list:
    keys    = rdb.keys("corrector:params:*")
    prefix  = "corrector:params:"
    return [k[len(prefix):] for k in keys]


def _rejection_stats(device_id: str, window_seconds: int = 3600) -> dict:
    key     = _REJECTION_KEY.format(device_id=device_id)
    raw     = rdb.lrange(key, 0, -1)
    cutoff  = datetime.now(tz=timezone.utc) - timedelta(seconds=window_seconds)

    recent = []
    for entry in raw:
        try:
            data = json.loads(entry)
            ts   = datetime.fromisoformat(data["ts"])
            if ts >= cutoff:
                recent.append(data)
        except (ValueError, KeyError):
            continue

    total_in_window = len(recent)
    reason_counts: dict = {}
    for item in recent:
        for r in item.get("reasons", []):
            reason_counts[r.split("=")[0]] = reason_counts.get(r.split("=")[0], 0) + 1

    return {
        "count_last_hour": total_in_window,
        "reason_breakdown": reason_counts,
    }


def _recent_detections(device_id: str, n: int = 20) -> list:
    key = _HISTORY_KEY.format(device_id=device_id)
    raw = rdb.lrange(key, 0, n - 1)
    results = []
    for entry in raw:
        try:
            results.append(json.loads(entry))
        except ValueError:
            continue
    return results


@diagnostics_bp.route("/detection-diagnostics", methods=["GET"])
def detection_diagnostics():
    cached = rdb.get(_CACHE_KEY)
    if cached:
        return jsonify(json.loads(cached))

    devices = {}
    for device_id in _list_devices():
        params = get_params(device_id)
        if not params:
            continue
        devices[device_id] = {
            "corrector":          params,
            "recent_detections":  _recent_detections(device_id),
            "rejection_stats":    _rejection_stats(device_id),
        }

    env_weather = _latest_cached_weather()
    current_env = None
    if env_weather is not None:
        from detection.environment import score_environment
        env = score_environment(env_weather, datetime.now(tz=timezone.utc))
        current_env = {
            "overall":        round(env.overall, 4),
            "factors":        {k: round(v, 4) for k, v in env.factors.items()},
            "recommendation": env.recommendation,
        }

    result = {
        "devices":             devices,
        "current_environment": current_env,
        "cnn_models":          cnn_model_status(),
        "generated_at":        datetime.now(tz=timezone.utc).isoformat(),
    }

    rdb.setex(_CACHE_KEY, config.DIAGNOSTICS_TTL, json.dumps(result))
    return jsonify(result)


def _latest_cached_weather() -> dict | None:
    keys = rdb.keys("weather:*:*")
    if not keys:
        return None
    try:
        raw = rdb.get(keys[0])
        return json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return None
