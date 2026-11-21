from datetime import datetime, timezone

from flask import Blueprint, request, jsonify

from detection.species import ingest_catch_log, predict_species, spatial_fish_map

catchlogs_bp = Blueprint("catchlogs", __name__)


@catchlogs_bp.route("/catch-logs/ingest", methods=["POST"])
def ingest():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "body required"}), 400

    if isinstance(body, list):
        logs = body
    elif isinstance(body, dict) and "catch_logs" in body:
        logs = body["catch_logs"]
    else:
        logs = [body]

    ingested = 0
    skipped  = 0
    for entry in logs:
        fish_type = str(entry.get("fish_type", "")).strip().lower()
        if fish_type and fish_type != "unknown":
            ingest_catch_log(entry)
            ingested += 1
        else:
            skipped += 1

    return jsonify({"ingested": ingested, "skipped": skipped})


@catchlogs_bp.route("/species/predict", methods=["POST"])
def species_predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "body required"}), 400

    lat = body.get("latitude")
    lon = body.get("longitude")
    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude required"}), 400

    ts_str = body.get("timestamp")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "invalid timestamp format, use ISO 8601"}), 400
    else:
        ts = datetime.now(tz=timezone.utc)

    radius_m = float(body.get("radius_m", 2000))
    depth_m  = body.get("depth_m")
    if depth_m is not None:
        depth_m = float(depth_m)

    result = predict_species(float(lat), float(lon), ts, radius_m=radius_m, depth_m=depth_m)
    return jsonify({"data": result})


@catchlogs_bp.route("/spatial/fish-map", methods=["POST"])
def fish_map():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "body required"}), 400

    ts_str = body.get("timestamp")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "invalid timestamp format, use ISO 8601"}), 400
    else:
        ts = datetime.now(tz=timezone.utc)

    radius_m = float(body.get("radius_m", 1000))

    if "points" in body:
        points = body["points"]
        if not isinstance(points, list):
            return jsonify({"error": "points must be an array"}), 400
    elif "bbox" in body:
        bbox = body["bbox"]
        try:
            points = _generate_grid(bbox, int(body.get("grid_size", 10)))
        except KeyError as e:
            return jsonify({"error": f"bbox missing field: {e}"}), 400
    else:
        return jsonify({"error": "points or bbox required"}), 400

    results = spatial_fish_map(points, ts, radius_m=radius_m)
    return jsonify({"data": results, "count": len(results)})


def _generate_grid(bbox: dict, grid_size: int) -> list:
    min_lat  = float(bbox["min_lat"])
    max_lat  = float(bbox["max_lat"])
    min_lon  = float(bbox["min_lon"])
    max_lon  = float(bbox["max_lon"])
    lat_step = (max_lat - min_lat) / grid_size
    lon_step = (max_lon - min_lon) / grid_size

    return [
        {"lat": round(min_lat + i * lat_step, 6),
         "lon": round(min_lon + j * lon_step, 6)}
        for i in range(grid_size + 1)
        for j in range(grid_size + 1)
    ]
