"""Bounded, cheap crop selection; no OCR or sharpening here."""

import cv2
import numpy as np


def crop_descriptor(crop):
    # A tiny descriptor measures view differences, not plate identity.
    return cv2.cvtColor(cv2.resize(crop, (16, 16)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def select_plate_candidates(candidates: list[dict], limit: int) -> list[dict]:
    """Keep the best-quality crop, then prefer varied views of similar quality.

    Diversity contributes at most 0.10 to the existing quality score. A poor
    image therefore cannot displace a clearly better one just by being older.
    """
    remaining = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = []
    while remaining and len(selected) < max(limit, 1):
        def priority(item):
            if not selected:
                return item["score"]
            differences = []
            for other in selected:
                gap = abs(item.get("timestamp", 0) - other.get("timestamp", 0))
                visual = 0.0
                if item.get("descriptor") is not None and other.get("descriptor") is not None:
                    visual = min(float(np.abs(item["descriptor"] - other["descriptor"]).mean()) / 0.15, 1.0)
                differences.append(0.6 * min(gap / 0.5, 1.0) + 0.4 * visual)
            return item["score"] + 0.10 * min(differences)
        index = max(range(len(remaining)), key=lambda index: priority(remaining[index]))
        selected.append(remaining.pop(index))
    return selected
