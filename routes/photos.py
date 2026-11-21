import base64
import json
import logging

import redis
from flask import Blueprint, jsonify, request

import config
from detection import vision
from detection.fusion import fuse

photos_bp = Blueprint("photos", __name__)
logger    = logging.getLogger(__name__)

_rdb              = redis.from_url(config.REDIS_URL, decode_responses=True)
_HISTORY_KEY      = "detection:history:{device_id}"


def _read_image_bytes() -> tuple[bytes | None, str | None]:
    if request.files and "photo" in request.files:
        return request.files["photo"].read(), None

    raw = request.get_data()
    if raw:
        return raw, None

    return None, "no image data: send multipart 'photo' field or raw JPEG body"


def _lookup_sonar(reading_id: str) -> dict | None:
    if not reading_id:
        return None

    keys = _rdb.keys(_HISTORY_KEY.replace("{device_id}", "*"))
    for key in keys:
        entries = _rdb.lrange(key, 0, -1)
        for raw in entries:
            try:
                det = json.loads(raw)
                if det.get("reading_id") == reading_id:
                    return det
            except (ValueError, KeyError):
                continue
    return None


def _build_response(reading_id: str | None, cnn: dict, sonar: dict | None) -> dict:
    payload: dict = {
        "reading_id":        reading_id,
        "is_fish":           cnn.get("is_fish", False),
        "fish_confidence":   cnn.get("fish_confidence", 0.0),
        "scene_class":       cnn.get("scene_class", "inconclusive"),
        "species":           cnn.get("species"),
        "species_confidence": cnn.get("species_confidence"),
        "fish_count":        cnn.get("fish_count", 0),
        "species_breakdown": cnn.get("species_breakdown", {}),
        "boxes":             cnn.get("boxes", []),
    }

    if sonar is not None:
        fusion = fuse(sonar_result=sonar, cnn_result=cnn)
        payload["fusion"] = {
            "sonar_confidence": fusion["sonar_confidence"],
            "cnn_confidence":   fusion["cnn_confidence"],
            "fused_confidence": fusion["fused_confidence"],
            "method":           fusion["method"],
        }
        payload["is_fish"]         = fusion["fused_detected"]
        payload["fish_confidence"] = fusion["fused_confidence"]

    return payload


@photos_bp.route("/classify", methods=["POST"])
def classify_photo():
    image_bytes, err = _read_image_bytes()
    if err:
        return jsonify({"error": err}), 400

    reading_id = (
        request.form.get("reading_id")
        or request.headers.get("X-Reading-ID")
    )

    cnn    = vision.classify(image_bytes)
    sonar  = _lookup_sonar(reading_id) if reading_id else None

    if cnn.get("error"):
        logger.warning("CNN classify error for reading %s: %s", reading_id, cnn["error"])

    return jsonify({"data": _build_response(reading_id, cnn, sonar)})


@photos_bp.route("/classify/batch", methods=["POST"])
def classify_batch():
    body = request.get_json(silent=True)
    if not body or not isinstance(body, list):
        return jsonify({"error": "body must be a JSON array of {reading_id, photo_base64}"}), 400

    results = []
    for item in body:
        reading_id = item.get("reading_id")
        b64        = item.get("photo_base64", "")

        try:
            image_bytes = base64.b64decode(b64)
        except Exception:
            results.append({
                "reading_id": reading_id,
                "error":      "invalid base64",
            })
            continue

        cnn   = vision.classify(image_bytes)
        sonar = _lookup_sonar(reading_id) if reading_id else None
        results.append(_build_response(reading_id, cnn, sonar))

    return jsonify({"count": len(results), "data": results})
