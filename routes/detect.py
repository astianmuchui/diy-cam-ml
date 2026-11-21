import json

import redis

import config
from flask import Blueprint, request, jsonify
from detection.engine import process, process_session
from detection.fusion import fuse
from detection.gate import gate
from detection.species import ingest_catch_log

detect_bp = Blueprint("detect", __name__)

_rdb = redis.from_url(config.REDIS_URL, decode_responses=True)
_SESSION_CACHE_KEY = "session:analysis:{session_id}"


@detect_bp.route("/detect", methods=["POST"])
def detect():
    body = request.get_json(silent=True)
    if not body or "reading" not in body:
        return jsonify({"error": "reading field required"}), 400

    result = process(
        reading=body["reading"],
        weather=body.get("weather"),
        prev_corr_depth=body.get("prev_corr_depth"),
        next_corr_depth=body.get("next_corr_depth"),
    )
    return jsonify({"data": result})


@detect_bp.route("/detect/batch", methods=["POST"])
def detect_batch():
    body = request.get_json(silent=True)
    if not body or "readings" not in body:
        return jsonify({"error": "readings field required"}), 400

    readings = body["readings"]
    if not isinstance(readings, list):
        return jsonify({"error": "readings must be an array"}), 400

    weather = body.get("weather")

    results          = []
    prev_corr_depth  = None

    for reading in readings:
        result = process(
            reading=reading,
            weather=weather,
            prev_corr_depth=prev_corr_depth,
            next_corr_depth=None,
        )
        results.append(result)
        if result.get("gate_valid"):
            prev_corr_depth = result.get("corrected_depth")

    return jsonify({"data": results, "count": len(results)})


@detect_bp.route("/sessions/<session_id>/analyze", methods=["POST"])
def session_analyze(session_id: str):
    cache_key = _SESSION_CACHE_KEY.format(session_id=session_id)
    cached = _rdb.get(cache_key)
    if cached:
        return cached, 200, {"Content-Type": "application/json"}

    body = request.get_json(silent=True)
    if not body or "readings" not in body:
        return jsonify({"error": "readings field required"}), 400

    readings = body["readings"]
    if not isinstance(readings, list):
        return jsonify({"error": "readings must be an array"}), 400

    for cl in body.get("catch_logs", []):
        ingest_catch_log(cl)

    results, summary = process_session(readings, weather=body.get("weather"))

    photo_results: dict = body.get("photo_results") or {}
    if photo_results:
        for det in results:
            rid = det.get("reading_id") or det.get("id")
            if rid and rid in photo_results:
                cnn_res = photo_results[rid]
                cnn_input = {
                    "scene_class":     cnn_res.get("scene_class", "inconclusive"),
                    "fish_confidence": cnn_res.get("confidence", 0.0),
                }
                det["fusion"] = fuse(sonar_result=det, cnn_result=cnn_input)

    response = {
        "session_id": session_id,
        "count":      len(results),
        "detections": results,
        "summary":    summary,
    }

    serialised = json.dumps(response)
    _rdb.set(cache_key, serialised)
    return serialised, 200, {"Content-Type": "application/json"}


@detect_bp.route("/gate", methods=["POST"])
def gate_check():
    body = request.get_json(silent=True)
    if not body or "reading" not in body:
        return jsonify({"error": "reading field required"}), 400

    result = gate(body["reading"])
    return jsonify({"data": {"valid": result.valid, "reject_reasons": result.reject_reasons}})
