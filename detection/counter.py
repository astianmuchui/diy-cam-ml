import logging
from io import BytesIO

import numpy as np

import config
from detection import vision

logger = logging.getLogger(__name__)

_GRID_N      = config.COUNT_GRID_N
_OVERLAP     = config.COUNT_TILE_OVERLAP
_TILE_THRESH = config.COUNT_TILE_THRESHOLD
_MERGE_IOU   = config.COUNT_MERGE_IOU
_MAX_FISH    = config.COUNT_MAX_FISH


def _tiles(width: int, height: int):
    cw = width / _GRID_N
    ch = height / _GRID_N
    ox = cw * _OVERLAP
    oy = ch * _OVERLAP
    for i in range(_GRID_N):
        for j in range(_GRID_N):
            x0 = max(0, int(j * cw - ox))
            y0 = max(0, int(i * ch - oy))
            x1 = min(width,  int((j + 1) * cw + ox))
            y1 = min(height, int((i + 1) * ch + oy))
            if x1 > x0 and y1 > y0:
                yield (x0, y0, x1, y1)


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _merge(boxes: list) -> list:
    n = len(boxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for k in range(i + 1, n):
            if _iou(boxes[i], boxes[k]) >= _MERGE_IOU:
                parent[find(i)] = find(k)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])

    blobs = []
    for members in groups.values():
        x0 = min(b[0] for b in members)
        y0 = min(b[1] for b in members)
        x1 = max(b[2] for b in members)
        y1 = max(b[3] for b in members)
        blobs.append((x0, y0, x1, y1))
    return blobs


def count_fish(image_bytes: bytes) -> dict:
    empty = {
        "fish_count": 0,
        "species_breakdown": {},
        "boxes": [],
        "method": "grid_tile",
    }

    if not vision.models_ready():
        return {**empty, "note": "models_not_loaded"}

    try:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.error("counter: image decode failed: %s", exc)
        return {**empty, "note": "image_decode_failed"}

    w, h = img.size
    positive: list = []

    for box in _tiles(w, h):
        crop   = img.crop(box)
        tensor = vision.preprocess_pil(crop)
        res    = vision.scene_probs(tensor)
        if res is None:
            return {**empty, "note": "inference_failed"}
        cls, conf, _ = res
        if cls == "fish_detected" and conf >= _TILE_THRESH:
            positive.append(box)

    if not positive:
        return empty

    blobs = _merge(positive)
    if len(blobs) > _MAX_FISH:
        blobs = sorted(
            blobs,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )[:_MAX_FISH]

    breakdown: dict = {}
    out_boxes: list = []
    for (x0, y0, x1, y1) in blobs:
        crop = img.crop((x0, y0, x1, y1))
        sp   = vision.species_probs(vision.preprocess_pil(crop))
        species, sconf = sp if sp is not None else ("unknown", 0.0)
        breakdown[species] = breakdown.get(species, 0) + 1
        out_boxes.append({
            "box":        [round(x0 / w, 4), round(y0 / h, 4),
                           round(x1 / w, 4), round(y1 / h, 4)],
            "species":    species,
            "confidence": round(sconf, 4),
        })

    return {
        "fish_count":        len(blobs),
        "species_breakdown": breakdown,
        "boxes":             out_boxes,
        "method":            "grid_tile",
    }


def counter_config() -> dict:
    return {
        "grid_n":         _GRID_N,
        "tile_overlap":   _OVERLAP,
        "tile_threshold": _TILE_THRESH,
        "merge_iou":      _MERGE_IOU,
        "max_fish":       _MAX_FISH,
    }
