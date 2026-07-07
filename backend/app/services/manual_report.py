"""Reviewer-reported missed violations (false negatives).

When a reviewer spots a rider the pipeline did not save, this module captures
an evidence frame from the stored source video, classifies why the pipeline
likely missed it using the sampled detection metadata, and stores the record
alongside detected violations with source='manual' and review_status
'confirmed' (a human just witnessed it).
"""

import json
from uuid import uuid4

import cv2

from app.core.config import settings
from app.core.database import utc_now
from app.services.pipeline import detection_metadata_path, enable_capture_orientation_auto, media_url
from app.services.repository import create_violation, increment_job_violation_count

MISS_REASONS = {
    "not_sampled": "No frame near the report was sampled for analysis",
    "motorcycle_not_detected": "No motorcycle was detected near the report",
    "helmet_model_miss": "Motorcycles were detected but no no-helmet box was found",
    "gated_or_undervoted": "No-helmet detections existed but no violation was saved",
    "no_metadata": "Detection metadata is unavailable for this job",
}


class ManualReportError(ValueError):
    """Raised when a manual report cannot be created from the stored video."""


def create_manual_violation(
    job_id: str,
    source_path: str | None,
    *,
    timestamp: float | None,
    frame_number: int | None,
    note: str | None = None,
    plate_text: str | None = None,
) -> dict:
    if not source_path:
        raise ManualReportError("Job has no stored source video")

    frame, frame_number, timestamp = extract_frame(source_path, timestamp, frame_number)
    miss_reason = classify_miss(job_id, timestamp)

    violation_id = uuid4().hex
    annotate_reported_miss(frame, frame_number)
    evidence_path = settings.evidence_dir / f"{job_id}_{frame_number}_{violation_id[:8]}.jpg"
    cv2.imwrite(str(evidence_path), frame)

    record = {
        "id": violation_id,
        "job_id": job_id,
        "detected_at": utc_now(),
        "helmet_status": "no_helmet",
        "helmet_confidence": 1.0,
        "plate_text": (plate_text or "").strip() or None,
        "plate_confidence": None,
        "evidence_image": media_url(evidence_path),
        "plate_image": None,
        "frame_number": frame_number,
        "track_id": None,
        "review_status": "confirmed",
        "source": "manual",
        "note": (note or "").strip() or None,
        "miss_reason": miss_reason,
    }
    create_violation(record)
    increment_job_violation_count(job_id)
    return record


def extract_frame(source_path: str, timestamp: float | None, frame_number: int | None):
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise ManualReportError("Could not open the stored source video")
    enable_capture_orientation_auto(capture)

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if frame_number is None:
            if fps <= 0:
                raise ManualReportError("Video FPS is unknown; report by frame number instead")
            frame_number = int(round((timestamp or 0.0) * fps))
        if total_frames > 0:
            frame_number = min(max(frame_number, 0), total_frames - 1)
        if timestamp is None:
            timestamp = frame_number / fps if fps > 0 else 0.0

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ManualReportError("Could not read the reported frame from the video")
        return frame, frame_number, timestamp
    finally:
        capture.release()


def classify_miss(job_id: str, timestamp: float) -> str:
    """Coarse diagnosis of why the pipeline missed a reported rider.

    Looks at the sampled detection frames around the reported moment: whether
    that moment was sampled at all, whether a motorcycle was even detected,
    whether the helmet model produced a no-helmet box, or whether detections
    existed but were gated/undervoted before saving.
    """
    path = detection_metadata_path(job_id)
    if not path.exists():
        return "no_metadata"

    try:
        frames = json.loads(path.read_text(encoding="utf-8")).get("frames", [])
    except (json.JSONDecodeError, OSError):
        return "no_metadata"

    window = max(1.5, settings.sample_every_seconds * 1.5)
    nearby = [frame for frame in frames if abs(frame.get("timestamp", 0.0) - timestamp) <= window]
    if not nearby:
        return "not_sampled"
    if not any(frame.get("motorcycles") for frame in nearby):
        return "motorcycle_not_detected"
    if not any(frame.get("no_helmets") for frame in nearby):
        return "helmet_model_miss"
    return "gated_or_undervoted"


def annotate_reported_miss(frame, frame_number: int) -> None:
    label = f"REPORTED MISS - frame {frame_number}"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (0, 0), (text_width + 20, text_height + baseline + 14), (45, 61, 186), -1)
    cv2.putText(frame, label, (10, text_height + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
