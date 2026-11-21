import config

_SONAR_W = config.FUSION_SONAR_WEIGHT
_CNN_W   = config.FUSION_CNN_WEIGHT
_THRESH  = config.CNN_OVERRIDE_THRESHOLD

_LIKELY_THRESHOLD   = config.SCORE_THRESHOLD_LIKELY
_POSSIBLE_THRESHOLD = config.SCORE_THRESHOLD_POSSIBLE
_UNLIKELY_THRESHOLD = config.SCORE_THRESHOLD_UNLIKELY


def fuse(sonar_result: dict, cnn_result: dict) -> dict:
    sonar_conf  = float(sonar_result.get("confidence", 0.0))
    cnn_scene   = cnn_result.get("scene_class", "inconclusive")
    cnn_conf    = float(cnn_result.get("fish_confidence", 0.0))

    if cnn_scene == "inconclusive":
        fused_conf = sonar_conf
        method     = "sonar_only"

    elif cnn_scene == "no_fish" and cnn_conf > _THRESH:
        fused_conf = sonar_conf * 0.3
        method     = "cnn_override"

    elif cnn_scene == "fish_detected" and cnn_conf > _THRESH:
        fused_conf = min(1.0, sonar_conf + cnn_conf * _CNN_W)
        method     = "weighted_average"

    else:
        fused_conf = _SONAR_W * sonar_conf + _CNN_W * cnn_conf
        method     = "weighted_average"

    fused_conf = max(0.0, min(1.0, fused_conf))

    return {
        "fused_detected":       fused_conf >= _POSSIBLE_THRESHOLD,
        "fused_confidence":     round(fused_conf, 4),
        "fused_classification": _classify(fused_conf),
        "sonar_confidence":     round(sonar_conf, 4),
        "cnn_confidence":       round(cnn_conf, 4),
        "cnn_scene_class":      cnn_scene,
        "method":               method,
    }


def _classify(confidence: float) -> str:
    if confidence >= _LIKELY_THRESHOLD:
        return "likely_fish"
    if confidence >= _POSSIBLE_THRESHOLD:
        return "possible"
    if confidence >= _UNLIKELY_THRESHOLD:
        return "unlikely"
    return "no_detection"
