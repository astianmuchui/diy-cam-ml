import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import redis

import config
from detection.gate import gate
from detection.correction import correct, TiltCorrector
from detection.environment import score_environment
from detection.scorer import score
from detection.species import predict_species

rdb = redis.from_url(config.REDIS_URL, decode_responses=True)

_HISTORY_KEY   = "detection:history:{device_id}"
_REJECTION_KEY = "gate:rejections:{device_id}"


def _parse_timestamp(recorded_at) -> datetime:
    if isinstance(recorded_at, datetime):
        return recorded_at
    if isinstance(recorded_at, str):
        try:
            return datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)


def process(
    reading:          dict,
    weather:          Optional[dict] = None,
    prev_corr_depth:  Optional[float] = None,
    next_corr_depth:  Optional[float] = None,
) -> dict:
    device_id = reading.get("device_id", "default")

    gate_result = gate(reading)
    if not gate_result.valid:
        _record_rejection(device_id, gate_result.reject_reasons)
        return {
            "fish_detected":   False,
            "confidence":      0.0,
            "classification":  "no_detection",
            "corrected_depth": reading.get("water_depth_m", 0.0),
            "baseline":        0.0,
            "fish_count":      0,
            "fish_type":       "unknown",
            "factors":         {},
            "environment":     {},
            "correction":      {},
            "gate_valid":      False,
            "reject_reasons":  gate_result.reject_reasons,
        }

    depth      = reading.get("water_depth_m", 0.0)
    tilt       = reading.get("tilt_angle", 0.0)
    corr       = correct(device_id, depth, tilt)

    ts         = _parse_timestamp(reading.get("recorded_at"))
    env_score  = score_environment(weather, ts)

    detection  = score(
        corrected_depth=corr.corrected_depth,
        baseline=corr.baseline,
        reading=reading,
        env_score=env_score,
        prev_corr_depth=prev_corr_depth,
        next_corr_depth=next_corr_depth,
    )

    result = asdict(detection)
    result["correction"] = {
        "slope":       round(corr.slope, 6),
        "intercept":   round(corr.intercept, 6),
        "r_squared":   round(corr.r_squared, 4),
        "baseline":    round(corr.baseline, 4),
        "window_size": corr.window_size,
    }

    _enrich_with_species(result, reading, ts)
    _record_detection(device_id, reading.get("id", ""), result)
    return result


def _record_detection(device_id: str, reading_id: str, result: dict) -> None:
    key     = _HISTORY_KEY.format(device_id=device_id)
    entry   = {
        "reading_id":     reading_id,
        "ts":             datetime.now(tz=timezone.utc).isoformat(),
        "fish_detected":  result["fish_detected"],
        "confidence":     result["confidence"],
        "classification": result["classification"],
        "factors":        result["factors"],
    }
    rdb.lpush(key, json.dumps(entry))
    rdb.ltrim(key, 0, config.DETECTION_HISTORY_SIZE - 1)


def _enrich_with_species(result: dict, reading: dict, ts: datetime) -> None:
    if not result.get("fish_detected"):
        return

    lat = reading.get("latitude") or reading.get("Latitude")
    lon = reading.get("longitude") or reading.get("Longitude")
    if lat is None or lon is None:
        return

    depth_m = result.get("corrected_depth") or reading.get("water_depth_m")

    pred = predict_species(float(lat), float(lon), ts, depth_m=depth_m)
    result["fish_type"]         = pred["species"]
    result["species_confidence"] = pred["confidence"]
    result["species_votes"]      = pred["votes"]
    result["species_sample_size"] = pred["sample_size"]


def _process_with_corrector(
    reading:         dict,
    corrector:       TiltCorrector,
    weather:         Optional[dict] = None,
    prev_corr_depth: Optional[float] = None,
    next_corr_depth: Optional[float] = None,
) -> dict:
    gate_result = gate(reading)
    if not gate_result.valid:
        return {
            "fish_detected":   False,
            "confidence":      0.0,
            "classification":  "no_detection",
            "corrected_depth": reading.get("water_depth_m", 0.0),
            "baseline":        0.0,
            "fish_count":      0,
            "fish_type":       "unknown",
            "factors":         {},
            "environment":     {},
            "correction":      {},
            "gate_valid":      False,
            "reject_reasons":  gate_result.reject_reasons,
        }

    depth = reading.get("water_depth_m", 0.0)
    tilt  = reading.get("tilt_angle", 0.0)
    corr  = corrector.correct(depth, tilt)

    ts        = _parse_timestamp(reading.get("recorded_at"))
    env_score = score_environment(weather, ts)

    detection = score(
        corrected_depth=corr.corrected_depth,
        baseline=corr.baseline,
        reading=reading,
        env_score=env_score,
        prev_corr_depth=prev_corr_depth,
        next_corr_depth=next_corr_depth,
    )

    result = asdict(detection)
    result["correction"] = {
        "slope":       round(corr.slope, 6),
        "intercept":   round(corr.intercept, 6),
        "r_squared":   round(corr.r_squared, 4),
        "baseline":    round(corr.baseline, 4),
        "window_size": corr.window_size,
    }

    ts = _parse_timestamp(reading.get("recorded_at"))
    _enrich_with_species(result, reading, ts)
    return result


def process_session(readings: list, weather: Optional[dict] = None) -> tuple:
    corrector1  = TiltCorrector()
    corr_depths = []

    for reading in readings:
        gate_result = gate(reading)
        if gate_result.valid:
            cr = corrector1.correct(
                reading.get("water_depth_m", 0.0),
                reading.get("tilt_angle", 0.0),
            )
            corr_depths.append(cr.corrected_depth)
        else:
            corr_depths.append(None)

    corrector2    = TiltCorrector()
    results       = []
    likely        = 0
    possible      = 0
    gate_rejected = 0

    for i, reading in enumerate(readings):
        prev_d = corr_depths[i - 1] if i > 0 else None
        next_d = corr_depths[i + 1] if i < len(readings) - 1 else None

        result = _process_with_corrector(
            reading=reading,
            corrector=corrector2,
            weather=weather,
            prev_corr_depth=prev_d,
            next_corr_depth=next_d,
        )
        results.append(result)

        if not result.get("gate_valid"):
            gate_rejected += 1
        else:
            cls = result.get("classification", "")
            if cls == "likely_fish":
                likely += 1
            elif cls == "possible":
                possible += 1

    first_valid = next((r for r in results if r.get("gate_valid")), {})
    corr_block  = first_valid.get("correction", {})

    summary = {
        "total_readings":          len(results),
        "gate_rejected":           gate_rejected,
        "likely_fish":             likely,
        "possible":                possible,
        "baseline_depth_m":        corr_block.get("baseline", 0),
        "tilt_slope_cm_per_10deg": round(corr_block.get("slope", 0) * 10, 3),
    }

    return results, summary


def _record_rejection(device_id: str, reasons: list) -> None:
    key   = _REJECTION_KEY.format(device_id=device_id)
    entry = {
        "ts":      datetime.now(tz=timezone.utc).isoformat(),
        "reasons": reasons,
    }
    rdb.lpush(key, json.dumps(entry))
    rdb.ltrim(key, 0, config.REJECTION_HISTORY_SIZE - 1)
