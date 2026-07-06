"""Offline evaluation harness for the SafeRide violation pipeline.

Runs labeled clips through the real processing pipeline and reports event-level
precision/recall, duplicate rate, plate capture rate, and OCR accuracy.

Usage:
    python scripts/evaluate.py scripts/eval-labels.example.json
    python scripts/evaluate.py my-labels.json --json results.json --keep-jobs

Labels file format:
    {
      "frame_tolerance": 30,
      "clips": [
        {
          "video": "data/eval/clip-01.mp4",
          "events": [
            {"frame_start": 90, "frame_end": 240, "plate": "1กข1234"}
          ]
        },
        {"video": "data/eval/clean-clip.mp4", "events": []}
      ]
    }

Each event is one real no-helmet rider with the frame range where they are
visible. "plate" is optional; when present, OCR accuracy is scored against it.
Clips with an empty events list measure false positives on clean footage.

Eval jobs are written to the normal SafeRide database while running so the
pipeline behaves exactly as in production, then removed afterwards unless
--keep-jobs is passed (kept jobs appear in the dashboard for visual inspection).
Detection settings come from the current config/env, so tune via environment
variables, e.g.  $env:HELMET_CONFIDENCE="0.4"; python scripts/evaluate.py ...
"""

import argparse
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings  # noqa: E402
from app.core.database import get_connection, init_db  # noqa: E402
from app.services.pipeline import process_uploaded_video  # noqa: E402
from app.services.repository import create_job, delete_job, get_job  # noqa: E402
from app.services.storage import delete_job_media  # noqa: E402

DEFAULT_FRAME_TOLERANCE = 30


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the SafeRide pipeline against labeled clips")
    parser.add_argument("labels", help="Path to the labels JSON file")
    parser.add_argument("--json", dest="json_out", help="Write the full report to this JSON file")
    parser.add_argument(
        "--keep-jobs",
        action="store_true",
        help="Keep eval jobs and their media so results can be inspected in the dashboard",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    tolerance = int(labels.get("frame_tolerance", DEFAULT_FRAME_TOLERANCE))
    clips = labels.get("clips", [])
    if not clips:
        print("Labels file has no clips.")
        sys.exit(1)

    init_db()
    reports = []
    for clip in clips:
        video = (labels_path.parent / clip["video"]).resolve() if not Path(clip["video"]).is_absolute() else Path(clip["video"])
        if not video.exists():
            video = Path(clip["video"]).resolve()
        if not video.exists():
            print(f"skip: video not found: {clip['video']}")
            continue
        reports.append(evaluate_clip(video, clip.get("events", []), tolerance, args.keep_jobs))

    if not reports:
        print("No clips were evaluated.")
        sys.exit(1)

    print_report(reports, tolerance)
    if args.json_out:
        payload = {"frame_tolerance": tolerance, "clips": reports, "overall": overall_metrics(reports)}
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.json_out}")


def evaluate_clip(video: Path, events: list[dict], tolerance: int, keep_job: bool) -> dict:
    job_id = uuid4().hex
    print(f"processing {video.name} ...", flush=True)
    create_job(job_id, f"eval:{video.name}", str(video))
    try:
        process_uploaded_video(job_id, str(video))
        job = get_job(job_id) or {}
        predicted = fetch_job_violations(job_id)
        report = score_clip(video.name, events, predicted, tolerance)
        report["status"] = job.get("status")
        report["elapsed_seconds"] = job.get("elapsed_seconds")
        report["processing_fps"] = job.get("processing_fps")
        report["job_id"] = job_id
        return report
    finally:
        if not keep_job:
            delete_job(job_id)
            # source_path is intentionally not passed so the eval clip itself
            # is never deleted; only generated job media matching the job id.
            delete_job_media(job_id, None)


def fetch_job_violations(job_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, frame_number, track_id, plate_text, plate_image, helmet_confidence
            FROM violations
            WHERE job_id = ?
            ORDER BY frame_number
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def score_clip(name: str, events: list[dict], predicted: list[dict], tolerance: int) -> dict:
    matches_per_event: list[list[dict]] = [[] for _ in events]
    false_positives = []

    for violation in predicted:
        frame = violation.get("frame_number")
        matched = False
        if frame is not None:
            for index, event in enumerate(events):
                if event["frame_start"] - tolerance <= frame <= event["frame_end"] + tolerance:
                    matches_per_event[index].append(violation)
                    matched = True
                    break
        if not matched:
            false_positives.append(violation)

    detected_events = sum(1 for matches in matches_per_event if matches)
    duplicates = sum(max(len(matches) - 1, 0) for matches in matches_per_event)
    matched_records = sum(len(matches) for matches in matches_per_event)

    plate_expected = 0
    plate_captured = 0
    ocr_correct = 0
    for event, matches in zip(events, matches_per_event):
        if not event.get("plate") or not matches:
            continue
        plate_expected += 1
        if any(match.get("plate_image") for match in matches):
            plate_captured += 1
        expected_compact = compact_plate(event["plate"])
        if any(expected_compact and expected_compact in compact_plate(match.get("plate_text") or "") for match in matches):
            ocr_correct += 1

    return {
        "clip": name,
        "expected_events": len(events),
        "detected_events": detected_events,
        "missed_events": len(events) - detected_events,
        "predicted_records": len(predicted),
        "false_positive_records": len(false_positives),
        "duplicate_records": duplicates,
        "precision": ratio(matched_records, matched_records + len(false_positives)),
        "recall": ratio(detected_events, len(events)),
        "plate_capture_rate": ratio(plate_captured, plate_expected),
        "ocr_match_rate": ratio(ocr_correct, plate_expected),
    }


def overall_metrics(reports: list[dict]) -> dict:
    expected = sum(report["expected_events"] for report in reports)
    detected = sum(report["detected_events"] for report in reports)
    predicted = sum(report["predicted_records"] for report in reports)
    false_positives = sum(report["false_positive_records"] for report in reports)
    duplicates = sum(report["duplicate_records"] for report in reports)
    matched_records = predicted - false_positives
    return {
        "expected_events": expected,
        "detected_events": detected,
        "predicted_records": predicted,
        "false_positive_records": false_positives,
        "duplicate_records": duplicates,
        "precision": ratio(matched_records, predicted),
        "recall": ratio(detected, expected),
        "duplicate_rate": ratio(duplicates, detected),
    }


def print_report(reports: list[dict], tolerance: int) -> None:
    print(f"\nSafeRide evaluation (frame tolerance ±{tolerance})")
    print("-" * 96)
    header = f"{'clip':32} {'events':>6} {'found':>5} {'miss':>4} {'FP':>4} {'dup':>4} {'prec':>6} {'recall':>6} {'ocr':>5}"
    print(header)
    for report in reports:
        print(
            f"{report['clip'][:32]:32} {report['expected_events']:>6} {report['detected_events']:>5} "
            f"{report['missed_events']:>4} {report['false_positive_records']:>4} {report['duplicate_records']:>4} "
            f"{fmt(report['precision']):>6} {fmt(report['recall']):>6} {fmt(report['ocr_match_rate']):>5}"
        )
    overall = overall_metrics(reports)
    print("-" * 96)
    print(
        f"{'overall':32} {overall['expected_events']:>6} {overall['detected_events']:>5} "
        f"{overall['expected_events'] - overall['detected_events']:>4} {overall['false_positive_records']:>4} "
        f"{overall['duplicate_records']:>4} {fmt(overall['precision']):>6} {fmt(overall['recall']):>6}"
    )
    print(
        f"\nduplicate rate: {fmt(overall['duplicate_rate'])} "
        f"(extra records per detected rider; driven by tracker ID churn)"
    )


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def compact_plate(value: str) -> str:
    return re.sub(r"\s+", "", value or "").strip()


if __name__ == "__main__":
    main()
