"""Dedicated Thai license-plate OCR.

Specializes EasyOCR for Thai plates instead of treating them as generic text:

- The recognizer is restricted to an allowlist of Thai characters and Arabic
  digits, so Latin junk reads ("allo", "1o") are impossible by construction.
- Thai plates are multi-line (registration prefix / province / digit group);
  OCR lines are classified by shape and recombined top-to-bottom.
- A plate-format quality score ranks readings across preprocess variants so a
  plate-shaped reading beats raw high-confidence noise.
- Across a rider's track, readings are voted per character position, weighted
  by confidence — a single misread character is outvoted by the samples that
  read it correctly.
"""

import re
from collections import Counter, defaultdict

import cv2

from app.core.config import settings

# Thai plate line shapes: registration prefix ("1กข"), plain digit group
# ("1234"), and a pure-Thai province line ("เชียงใหม่").
PLATE_PREFIX_PATTERN = re.compile(r"^\d{0,2}[ก-ฮ]{1,3}\d{0,4}$")
PLATE_DIGITS_PATTERN = re.compile(r"^\d{1,4}$")
PLATE_PROVINCE_PATTERN = re.compile(r"^[ก-๙]{3,}$")

# Thai consonants, vowels, and tone marks (no Thai digits — plates use Arabic
# digits, and allowing both invites 4/๔-style confusions), plus Arabic digits.
THAI_PLATE_ALLOWLIST = (
    "".join(chr(code) for code in range(0x0E01, 0x0E3B))
    + "".join(chr(code) for code in range(0x0E40, 0x0E4F))
    + "0123456789"
)

_ocr_reader = None


def ocr_uses_gpu() -> bool:
    """OCR_GPU=true/false forces it; unset follows CUDA availability.
    EasyOCR's GPU path is CUDA-only, so Apple Silicon stays on CPU."""
    if settings.ocr_gpu is not None:
        return settings.ocr_gpu
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(
            settings.ocr_languages,
            gpu=ocr_uses_gpu(),
            model_storage_directory=str(settings.cache_dir / "easyocr"),
            user_network_directory=str(settings.cache_dir / "easyocr"),
            verbose=False,
        )
    return _ocr_reader


def read_plate_text(crop) -> tuple[str | None, float | None]:
    if not settings.enable_ocr:
        return None, None

    try:
        reader = get_reader()
        best_text = None
        best_confidence = None
        best_quality = 0.0
        for image in plate_ocr_variants(crop):
            lines = reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=THAI_PLATE_ALLOWLIST,
            )
            combined = combine_plate_lines(lines)
            if combined and combined[2] > best_quality:
                best_text, best_confidence, best_quality = combined
    except Exception:
        return None, None

    return best_text, best_confidence


def plate_ocr_variants(crop) -> list:
    variants = [crop]
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return variants

    scale = max(2.0, min(4.0, 260 / max(width, 1)))
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 7, 45, 45)
    threshold = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    variants.extend([resized, threshold])
    return variants


def combine_plate_lines(lines) -> tuple[str, float, float] | None:
    """Combine OCR line results into one plate reading.

    Returns (text, confidence, quality) where quality ranks readings across
    OCR variants: pattern-conforming readings beat raw high-confidence noise.
    """
    entries = []
    for line in lines:
        bbox, text, confidence = line[0], line[1], line[2]
        normalized = normalize_plate_text(str(text))
        compact = re.sub(r"\s+", "", normalized)
        if not compact:
            continue
        top = min(point[1] for point in bbox)
        entries.append(
            {
                "text": normalized,
                "compact": compact,
                "confidence": float(confidence),
                "top": top,
            }
        )
    if not entries:
        return None

    prefix = best_matching_entry(entries, PLATE_PREFIX_PATTERN, exclude=[])
    digits = best_matching_entry(entries, PLATE_DIGITS_PATTERN, exclude=[prefix])
    province = best_matching_entry(entries, PLATE_PROVINCE_PATTERN, exclude=[prefix, digits])

    registration_parts = sorted(
        [entry for entry in (prefix, digits) if entry], key=lambda entry: entry["top"]
    )
    if registration_parts:
        used = registration_parts + ([province] if province else [])
        text = " ".join(entry["compact"] for entry in registration_parts)
        if province:
            text = f"{text} {province['compact']}"
        confidence = sum(entry["confidence"] for entry in used) / len(used)
        quality = (
            confidence
            + (0.45 if prefix else 0.0)
            + (0.25 if digits else 0.0)
            + (0.15 if province else 0.0)
        )
        if len(re.sub(r"\s+", "", text)) < 2:
            return None
        return text, round(confidence, 4), quality

    # No plate-shaped lines: fall back to the strongest raw line, heavily
    # discounted so any pattern-conforming variant wins.
    fallback = max(entries, key=lambda entry: entry["confidence"])
    if len(fallback["compact"]) < 2:
        return None
    return fallback["text"], round(fallback["confidence"], 4), fallback["confidence"] * 0.40


def best_matching_entry(entries: list[dict], pattern: re.Pattern, exclude: list) -> dict | None:
    best = None
    for entry in entries:
        if any(entry is excluded for excluded in exclude if excluded):
            continue
        if not pattern.fullmatch(entry["compact"]):
            continue
        if best is None or entry["confidence"] > best["confidence"]:
            best = entry
    return best


def vote_plate_texts(reads: list[tuple[str, float]]) -> tuple[str, float] | None:
    """Character-level voting across a track's OCR readings.

    Each reading is split into plate components (registration prefix, digit
    group, province). Within each component, readings vote per character
    position weighted by confidence, so "1กข 1234" read four times and
    "1กข 1284" read once resolves to "1กข 1234" — a single misread character
    is outvoted instead of winning on one lucky confidence score."""
    if not reads:
        return None

    components: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for text, confidence in reads:
        weight = max(confidence, 0.05)
        for token in text.split():
            compact = re.sub(r"\s+", "", token)
            if PLATE_PREFIX_PATTERN.fullmatch(compact):
                components["prefix"].append((compact, weight))
            elif PLATE_DIGITS_PATTERN.fullmatch(compact):
                components["digits"].append((compact, weight))
            elif PLATE_PROVINCE_PATTERN.fullmatch(compact):
                components["province"].append((compact, weight))

    voted_parts = []
    confidences = []
    for name in ("prefix", "digits", "province"):
        voted = vote_component(components.get(name, []))
        if voted:
            voted_parts.append(voted[0])
            confidences.append(voted[1])

    if voted_parts:
        return " ".join(voted_parts), round(sum(confidences) / len(confidences), 4)

    # No component structure in any read: fall back to the reading whose text
    # accumulated the most confidence across samples.
    groups: dict[str, dict] = {}
    for text, confidence in reads:
        key = re.sub(r"\s+", "", text)
        group = groups.setdefault(key, {"total": 0.0, "best_text": text, "best_confidence": confidence})
        group["total"] += max(confidence, 0.05)
        if confidence > group["best_confidence"]:
            group["best_text"] = text
            group["best_confidence"] = confidence
    best_group = max(groups.values(), key=lambda group: group["total"])
    return best_group["best_text"], best_group["best_confidence"]


def vote_component(readings: list[tuple[str, float]]) -> tuple[str, float] | None:
    if not readings:
        return None

    length_votes: Counter = Counter()
    for token, weight in readings:
        length_votes[len(token)] += weight
    winning_length = length_votes.most_common(1)[0][0]
    candidates = [(token, weight) for token, weight in readings if len(token) == winning_length]

    voted_chars = []
    position_confidences = []
    for position in range(winning_length):
        char_votes: dict[str, float] = defaultdict(float)
        for token, weight in candidates:
            char_votes[token[position]] += weight
        winner = max(char_votes.items(), key=lambda item: item[1])
        total = sum(char_votes.values())
        voted_chars.append(winner[0])
        position_confidences.append(winner[1] / total if total > 0 else 0.0)

    text = "".join(voted_chars)
    peak_confidence = max(weight for _token, weight in candidates)
    agreement = sum(position_confidences) / len(position_confidences)
    return text, round(min(peak_confidence * agreement, 1.0), 4)


def normalize_plate_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = re.sub(r"[^\wก-๙\-\s]", "", text, flags=re.UNICODE)
    return text.strip(" -_")
