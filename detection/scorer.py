import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import config
from detection.environment import EnvironmentScore


@dataclass
class Detection:
    fish_detected:   bool
    confidence:      float
    classification:  str
    corrected_depth: float
    baseline:        float
    fish_count:      int
    fish_type:       str
    factors:         Dict[str, float]
    environment:     dict
    gate_valid:      bool
    reject_reasons:  List[str]


def score(
    corrected_depth:  float,
    baseline:         float,
    reading:          dict,
    env_score:        EnvironmentScore,
    prev_corr_depth:  Optional[float] = None,
    next_corr_depth:  Optional[float] = None,
) -> Detection:
    raw_score = 0.0
    factors: Dict[str, float] = {}

    depth_dev   = baseline - corrected_depth
    if depth_dev > 0 and baseline > 0:
        depth_score = min(depth_dev / (baseline * 0.015), 1.0)
    else:
        depth_score = 0.0
    contribution = depth_score * 0.35
    raw_score   += contribution
    factors["depth_deviation"] = round(contribution, 4)

    gx, gy, gz = reading.get("gyro_x", 0.0), reading.get("gyro_y", 0.0), reading.get("gyro_z", 0.0)
    gyro_mag    = math.sqrt(gx**2 + gy**2 + gz**2)
    gyro_penalty = min((gyro_mag - 30) / 60, 0.3) if gyro_mag > 30 else 0.0
    raw_score   -= gyro_penalty
    factors["gyro_penalty"] = round(-gyro_penalty, 4)

    ax, ay, az = reading.get("accel_x", 0.0), reading.get("accel_y", 0.0), reading.get("accel_z", 1.0)
    accel_mag   = math.sqrt(ax**2 + ay**2 + az**2)
    accel_bonus = 0.15 if abs(accel_mag - 1.0) < 0.15 and depth_score > 0.2 else 0.0
    raw_score  += accel_bonus
    factors["accel_bonus"] = round(accel_bonus, 4)

    transience = 0.0
    if depth_score > 0.2:
        neighbors = [d for d in [prev_corr_depth, next_corr_depth] if d is not None]
        if neighbors:
            neighbor_mean = sum(neighbors) / len(neighbors)
            neighbor_dev  = baseline - neighbor_mean
            if abs(neighbor_dev) < depth_dev * 0.5:
                transience = 0.15
            else:
                transience = -0.05
    raw_score += transience
    factors["transience"] = round(transience, 4)

    tilt       = reading.get("tilt_angle", 0.0)
    tilt_bonus = 0.10 if tilt < 5.0 and depth_score > 0.2 else 0.0
    raw_score += tilt_bonus
    factors["tilt_bonus"] = round(tilt_bonus, 4)

    env_weight = 0.6 + env_score.overall * 0.8
    factors["env_weight"] = round(env_weight, 4)

    confidence = max(0.0, min(raw_score * env_weight, 1.0))

    if confidence > config.SCORE_THRESHOLD_LIKELY:
        classification = "likely_fish"
    elif confidence > config.SCORE_THRESHOLD_POSSIBLE:
        classification = "possible"
    elif confidence > config.SCORE_THRESHOLD_UNLIKELY:
        classification = "unlikely"
    else:
        classification = "no_detection"

    fish_detected = classification in ("likely_fish", "possible")

    return Detection(
        fish_detected=fish_detected,
        confidence=round(confidence, 4),
        classification=classification,
        corrected_depth=round(corrected_depth, 4),
        baseline=round(baseline, 4),
        fish_count=1 if fish_detected else 0,
        fish_type="unknown",
        factors=factors,
        environment={
            "overall":         round(env_score.overall, 4),
            "factors":         {k: round(v, 4) for k, v in env_score.factors.items()},
            "recommendation":  env_score.recommendation,
        },
        gate_valid=True,
        reject_reasons=[],
    )
