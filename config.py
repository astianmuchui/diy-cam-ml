import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

CORRECTION_WINDOW_SIZE = int(os.getenv("CORRECTION_WINDOW_SIZE", "50"))

GATE_MAX_TILT        = float(os.getenv("GATE_MAX_TILT", "25.0"))
GATE_MIN_DEPTH       = float(os.getenv("GATE_MIN_DEPTH", "1.0"))
GATE_MIN_DIST_CM     = float(os.getenv("GATE_MIN_DIST_CM", "100.0"))
GATE_MAX_ACCEL_DEV   = float(os.getenv("GATE_MAX_ACCEL_DEV", "0.4"))

SCORE_THRESHOLD_LIKELY   = float(os.getenv("SCORE_THRESHOLD_LIKELY", "0.45"))
SCORE_THRESHOLD_POSSIBLE = float(os.getenv("SCORE_THRESHOLD_POSSIBLE", "0.25"))
SCORE_THRESHOLD_UNLIKELY = float(os.getenv("SCORE_THRESHOLD_UNLIKELY", "0.10"))

DIAGNOSTICS_TTL = int(os.getenv("DIAGNOSTICS_TTL", "15"))

DETECTION_HISTORY_SIZE = int(os.getenv("DETECTION_HISTORY_SIZE", "100"))
REJECTION_HISTORY_SIZE = int(os.getenv("REJECTION_HISTORY_SIZE", "500"))

SCENE_MODEL_PATH   = os.getenv("SCENE_MODEL_PATH",   "models/scene_classifier.onnx")
SPECIES_MODEL_PATH = os.getenv("SPECIES_MODEL_PATH", "models/species_classifier.onnx")

FUSION_SONAR_WEIGHT = float(os.getenv("FUSION_SONAR_WEIGHT", "0.4"))
FUSION_CNN_WEIGHT   = float(os.getenv("FUSION_CNN_WEIGHT",   "0.6"))

CNN_OVERRIDE_THRESHOLD = float(os.getenv("CNN_OVERRIDE_THRESHOLD", "0.7"))

COUNT_GRID_N         = int(os.getenv("COUNT_GRID_N", "3"))
COUNT_TILE_OVERLAP   = float(os.getenv("COUNT_TILE_OVERLAP", "0.25"))
COUNT_TILE_THRESHOLD = float(os.getenv("COUNT_TILE_THRESHOLD", "0.55"))
COUNT_MERGE_IOU      = float(os.getenv("COUNT_MERGE_IOU", "0.2"))
COUNT_MAX_FISH       = int(os.getenv("COUNT_MAX_FISH", "20"))

INSIGHTS_HORIZON_HOURS = int(os.getenv("INSIGHTS_HORIZON_HOURS", "24"))
INSIGHTS_TREND_DAYS    = int(os.getenv("INSIGHTS_TREND_DAYS", "30"))
