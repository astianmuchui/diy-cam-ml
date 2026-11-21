import logging
import os
from io import BytesIO

import numpy as np

logger = logging.getLogger(__name__)

_scene_model = None
_species_model = None
_models_ready = False


_SCENE_CLASSES   = ["fish_detected", "inconclusive", "no_fish"]
_SPECIES_CLASSES = ["bass", "catfish", "tilapia"]

_INPUT_SHAPE     = (224, 224)

_FISH_DETECTED_THRESHOLD = 0.85
_NO_FISH_THRESHOLD       = 0.60
_MIN_SPECIES_CONF        = 0.55

_scene_model_path   = ""
_species_model_path = ""


def init_models(model_dir: str) -> None:
    global _scene_model, _species_model, _models_ready
    global _scene_model_path, _species_model_path

    scene_path   = os.path.join(model_dir, "scene_classifier.onnx")
    species_path = os.path.join(model_dir, "species_classifier.onnx")
    _scene_model_path   = scene_path
    _species_model_path = species_path

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — CNN classification disabled")
        return

    _scene_model   = _try_load(scene_path, "scene classifier")
    _species_model = _try_load(species_path, "species classifier")
    _models_ready  = _scene_model is not None

    if _models_ready:
        logger.info("CNN models loaded: scene=%s species=%s", scene_path, species_path)
    else:
        logger.warning(
            "Scene classifier not found at %s — classify() will return inconclusive",
            scene_path,
        )


def _try_load(path: str, label: str):
    if not os.path.exists(path):
        logger.warning("%s model file not found: %s", label, path)
        return None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        logger.info("Loaded %s from %s", label, path)
        return session
    except Exception as exc:
        logger.error("Failed to load %s from %s: %s", label, path, exc)
        return None


def _preprocess(image_bytes: bytes) -> "np.ndarray | None":
    try:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize(_INPUT_SHAPE, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = arr / 127.5 - 1.0
        return np.expand_dims(arr, axis=0)
    except Exception as exc:
        logger.error("Image preprocessing failed: %s", exc)
        return None


def _run(session, tensor: "np.ndarray") -> "np.ndarray":
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    return output[0]


def classify_scene(image_bytes: bytes) -> dict:
    if not _models_ready or _scene_model is None:
        return _inconclusive_scene()

    tensor = _preprocess(image_bytes)
    if tensor is None:
        return _inconclusive_scene(error="image_decode_failed")

    try:
        probs = _run(_scene_model, tensor)
        idx   = int(np.argmax(probs))
        raw_class  = _SCENE_CLASSES[idx]
        confidence = float(probs[idx])

        if raw_class == "fish_detected" and confidence < _FISH_DETECTED_THRESHOLD:
            gated_class = "inconclusive"
            logger.info(
                "Scene gate: fish_detected@%.3f below %.2f → inconclusive",
                confidence, _FISH_DETECTED_THRESHOLD,
            )
        elif raw_class == "no_fish" and confidence < _NO_FISH_THRESHOLD:
            gated_class = "inconclusive"
            logger.info(
                "Scene gate: no_fish@%.3f below %.2f → inconclusive",
                confidence, _NO_FISH_THRESHOLD,
            )
        else:
            gated_class = raw_class

        logger.info(
            "Scene inference: raw=%s gated=%s confidence=%.4f probs=%s",
            raw_class, gated_class, confidence,
            {c: round(float(p), 4) for c, p in zip(_SCENE_CLASSES, probs)},
        )

        return {
            "scene_class":   gated_class,
            "confidence":    confidence,
            "probabilities": {cls: float(p) for cls, p in zip(_SCENE_CLASSES, probs)},
            "raw_class":     raw_class,
        }
    except Exception as exc:
        logger.error("Scene inference failed: %s", exc)
        return _inconclusive_scene(error="inference_failed")


def classify_species(image_bytes: bytes) -> dict:
    if _species_model is None:
        return {"species": "unknown", "confidence": 0.0, "probabilities": {}}

    tensor = _preprocess(image_bytes)
    if tensor is None:
        return {"species": "unknown", "confidence": 0.0, "probabilities": {}, "error": "image_decode_failed"}

    try:
        probs = _run(_species_model, tensor)
        idx   = int(np.argmax(probs))
        return {
            "species":       _SPECIES_CLASSES[idx],
            "confidence":    float(probs[idx]),
            "probabilities": {cls: float(p) for cls, p in zip(_SPECIES_CLASSES, probs)},
        }
    except Exception as exc:
        logger.error("Species inference failed: %s", exc)
        return {"species": "unknown", "confidence": 0.0, "probabilities": {}, "error": "inference_failed"}


def classify(image_bytes: bytes) -> dict:
    scene = classify_scene(image_bytes)
    result: dict = {
        "is_fish":             scene["scene_class"] == "fish_detected",
        "fish_confidence":     scene["confidence"],
        "scene_class":         scene["scene_class"],
        "scene_probabilities": scene.get("probabilities", {}),
        "species":             None,
        "species_confidence":  None,
        "species_probabilities": None,
        "fish_count":          0,
        "species_breakdown":   {},
        "boxes":               [],
    }

    if scene.get("error"):
        result["error"] = scene["error"]

    if scene["scene_class"] == "fish_detected":
        sp = classify_species(image_bytes)
        sp_conf = float(sp.get("confidence", 0.0))
        if sp_conf < _MIN_SPECIES_CONF:
            logger.info(
                "Two-stage gate: scene=fish@%.3f but species=%s@%.3f below %.2f → inconclusive",
                scene["confidence"], sp.get("species"), sp_conf, _MIN_SPECIES_CONF,
            )
            result["scene_class"]           = "inconclusive"
            result["is_fish"]               = False
            result["fish_confidence"]       = scene["confidence"]
            result["species_probabilities"] = sp.get("probabilities")
            return result
        result["species"]              = sp.get("species")
        result["species_confidence"]   = sp.get("confidence")
        result["species_probabilities"] = sp.get("probabilities")

        from detection.counter import count_fish

        c = count_fish(image_bytes)
        result["boxes"]             = c["boxes"]
        result["species_breakdown"] = c["species_breakdown"]
        if c.get("note"):
            result["count_note"] = c["note"]

        if c["fish_count"] > 0:
            result["fish_count"] = c["fish_count"]
        else:
            result["fish_count"] = 1
            sp_name = sp.get("species")
            if sp_name and sp_name != "unknown":
                result["species_breakdown"] = {sp_name: 1}

    return result


def preprocess_pil(img) -> "np.ndarray":
    img = img.convert("RGB").resize(_INPUT_SHAPE, _bilinear())
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


def _bilinear():
    from PIL import Image
    return Image.BILINEAR


def scene_probs(tensor: "np.ndarray") -> "tuple[str, float, np.ndarray] | None":
    if not _models_ready or _scene_model is None:
        return None
    try:
        probs = _run(_scene_model, tensor)
        idx   = int(np.argmax(probs))
        return _SCENE_CLASSES[idx], float(probs[idx]), probs
    except Exception as exc:
        logger.error("Scene tile inference failed: %s", exc)
        return None


def species_probs(tensor: "np.ndarray") -> "tuple[str, float] | None":
    if _species_model is None:
        return None
    try:
        probs = _run(_species_model, tensor)
        idx   = int(np.argmax(probs))
        return _SPECIES_CLASSES[idx], float(probs[idx])
    except Exception as exc:
        logger.error("Species tile inference failed: %s", exc)
        return None


def models_ready() -> bool:
    return _models_ready and _scene_model is not None


def model_status() -> dict:
    return {
        "scene_classifier": {
            "loaded":      _scene_model is not None,
            "model_file":  _scene_model_path,
            "classes":     _SCENE_CLASSES,
            "input_shape": list(_INPUT_SHAPE) + [3],
        },
        "species_classifier": {
            "loaded":     _species_model is not None,
            "model_file": _species_model_path,
            "classes":    _SPECIES_CLASSES,
        },
        "fish_counter": _counter_config(),
    }


def _counter_config() -> dict:
    try:
        from detection.counter import counter_config
        return {"method": "grid_tile", **counter_config()}
    except Exception:
        return {"method": "grid_tile"}


def _inconclusive_scene(error: str | None = None) -> dict:
    base = {
        "scene_class":   "inconclusive",
        "confidence":    0.0,
        "probabilities": {c: 0.0 for c in _SCENE_CLASSES},
    }
    if error:
        base["error"] = error
    return base
