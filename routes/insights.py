from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from detection.insights import build_insights

insights_bp = Blueprint("insights", __name__)


def _parse_ts(ts_str):
    if not ts_str:
        return datetime.now(tz=timezone.utc)
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None


@insights_bp.route("/insights", methods=["POST"])
def insights_post():
    body = request.get_json(silent=True) or {}
    lat  = body.get("latitude")
    lon  = body.get("longitude")
    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude required"}), 400

    ts = _parse_ts(body.get("timestamp"))
    if ts is None:
        return jsonify({"error": "invalid timestamp format, use ISO 8601"}), 400

    result = build_insights(
        lat=float(lat),
        lon=float(lon),
        timestamp=ts,
        weather=body.get("weather"),
        radius_m=float(body.get("radius_m", 2000)),
        horizon_hours=body.get("horizon_hours"),
    )
    return jsonify({"data": result})


@insights_bp.route("/insights", methods=["GET"])
def insights_get():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon query params required"}), 400

    result = build_insights(
        lat=lat,
        lon=lon,
        radius_m=request.args.get("radius_m", default=2000, type=float),
        horizon_hours=request.args.get("horizon_hours", type=int),
    )
    return jsonify({"data": result})
