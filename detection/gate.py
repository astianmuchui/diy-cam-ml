import math
from dataclasses import dataclass, field
from typing import List
import config


@dataclass
class GateResult:
    valid: bool
    reject_reasons: List[str] = field(default_factory=list)


def gate(reading: dict) -> GateResult:
    reasons = []

    tilt  = reading.get("tilt_angle", 0.0)
    depth = reading.get("water_depth_m", 0.0)
    dist  = reading.get("distance_cm", 0.0)
    ax    = reading.get("accel_x", 0.0)
    ay    = reading.get("accel_y", 0.0)
    az    = reading.get("accel_z", 1.0)

    if tilt > config.GATE_MAX_TILT:
        reasons.append(f"tilt={tilt:.1f}deg")

    if depth < config.GATE_MIN_DEPTH:
        reasons.append(f"depth={depth:.2f}m")

    if dist < config.GATE_MIN_DIST_CM:
        reasons.append(f"dist={dist:.0f}cm")

    accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
    if abs(accel_mag - 1.0) > config.GATE_MAX_ACCEL_DEV:
        reasons.append(f"accel={accel_mag:.2f}g")

    return GateResult(valid=len(reasons) == 0, reject_reasons=reasons)
