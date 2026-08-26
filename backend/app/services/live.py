"""Live ingestion: RTSP streams and local webcams.

Runs the same detection/tracking/violation pipeline as uploaded videos, but
reads frames from a live source in real time. Every frame is recorded to an
MP4 under data/uploads so completed live sessions replay exactly like uploaded
jobs, and annotated frames are published to the MJPEG hub for the Live tab.

Differences from file processing:
- Sampling is wall-clock based (a live source delivers frames at its own pace).
- There is no total frame count, so progress/ETA stay at zero while running.
- A session ends on operator stop, source loss, the violation cap, or the
  live_max_seconds safety limit.
"""

from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from urllib.parse import urlparse

import cv2

from app.core.config import settings
from app.services.pipeline import (
    RiderTrackManager,
    analyze_frame,
    annotate_analysis,
    empty_analysis,
    get_models,
    publish_stream_frame,
    save_preview,
    save_violation,
    serialize_detection_frame,
    status_message,
    write_detection_metadata,
)
from app.services.repository import update_job
from app.services.streaming import frame_hub

_stop_lock = Lock()
_stop_requests: set[str] = set()


def request_stop(job_id: str) -> None:
    with _stop_lock:
        _stop_requests.add(job_id)


def stop_requested(job_id: str) -> bool:
    with _stop_lock:
        return job_id in _stop_requests


def clear_stop(job_id: str) -> None:
    with _stop_lock:
        _stop_requests.discard(job_id)


def live_display_name(source: str) -> str:
    """Human-readable source name with any RTSP credentials stripped."""
    source = source.strip()
    if source.isdigit():
        return f"live:webcam-{source}"
    parsed = urlparse(source)
    if parsed.scheme and parsed.hostname:
        port = f":{parsed.port}" if parsed.port else ""
        return f"live:{parsed.scheme}://{parsed.hostname}{port}"
    return f"live:{Path(source).name or source}"


def open_live_capture(source: str) -> cv2.VideoCapture:
    source = source.strip()
    if source.isdigit():
        capture = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(int(source))
        return capture
    capture = cv2.VideoCapture(source)
    # Keep the network buffer shallow so analysis stays near-live.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_recorder(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter | None:
    # avc1 (H.264) plays natively in browsers; mp4v is the fallback when no
    # H.264 encoder is available — the file still works for evidence/replay
    # download even if in-browser playback is unavailable.
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def process_live_stream(job_id: str, source: str) -> None:
    clear_stop(job_id)
    update_job(
        job_id,
        "processing",
        "Connecting to live source",
        progress=0,
        current_frame=0,
        total_frames=0,
        sampled_frames=0,
        violation_count=0,
        elapsed_seconds=0,
        processing_fps=0,
        eta_seconds=0,
        result="processing",
    )

    capture = open_live_capture(source)
    if not capture.isOpened():
        capture.release()
        update_job(job_id, "failed", "Could not open the live source", result="failed")
        return

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = source_fps if 5 <= source_fps <= 60 else 25.0
    recording_path = settings.upload_dir / f"{job_id}.mp4"
    recorder: cv2.VideoWriter | None = None
    recorder_failed = False

    analysis_interval = max(int(round(fps * settings.sample_every_seconds)), 1)
    cooldown_frames = max(int(fps * settings.violation_cooldown_seconds), analysis_interval)
    collection_frames = max(int(fps * settings.plate_collection_seconds), analysis_interval)
    max_lost_frames = max(int(fps * settings.tracker_max_lost_seconds), analysis_interval)
    dedupe_frames = max(int(fps * settings.rider_dedupe_seconds), max_lost_frames)
    rider_tracks = RiderTrackManager(
        cooldown_frames,
        collection_frames,
        max_lost_frames,
        dedupe_frames,
    )

    dense_interval_seconds = settings.sample_every_seconds / max(settings.adaptive_sample_divisor, 1)
    publish_interval = 1.0 / max(settings.live_preview_fps, 1)

    frame_number = 0
    sampled_count = 0
    violation_count = 0
    read_failures = 0
    latest_analysis = empty_analysis()
    latest_preview_url = None
    started = monotonic()
    last_analysis_at = -1e9
    last_publish_at = 0.0
    last_preview_save = 0.0
    last_status_update = 0.0
    last_metadata_write = 0.0
    dense_until = 0.0
    end_reason = "Live session stopped"
    detection_records: list[dict] = []
    write_detection_metadata(job_id, detection_records)

    try:
        models = get_models()
        update_job(job_id, "processing", "Live analysis running")

        while True:
            now = monotonic()
            if stop_requested(job_id):
                end_reason = "Stopped by operator"
                break
            if now - started >= settings.live_max_seconds:
                end_reason = f"Reached the {settings.live_max_seconds}s session limit"
                break
            if violation_count >= settings.max_violations_per_video:
                end_reason = "Reached the violation cap for one session"
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures > 60:
                    end_reason = "Live source ended or was lost"
                    break
                sleep(0.05)
                continue
            read_failures = 0

            if recorder is None and not recorder_failed:
                height, width = frame.shape[:2]
                recorder = create_recorder(recording_path, fps, (width, height))
                recorder_failed = recorder is None
            if recorder is not None:
                recorder.write(frame)
            frame_number += 1

            now = monotonic()
            in_dense_window = settings.adaptive_sampling and now <= dense_until
            interval = dense_interval_seconds if in_dense_window else settings.sample_every_seconds
            should_analyze = now - last_analysis_at >= interval

            annotated = None
            if should_analyze:
                sampled_count += 1
                last_analysis_at = now
                analysis = analyze_frame(frame, models)
                rider_tracks.update(analysis, frame_number, frame)
                latest_analysis = analysis
                if analysis["no_helmets"]:
                    dense_until = now + settings.adaptive_hold_seconds

                annotated = annotate_analysis(frame, frame_number, analysis, fresh_analysis=True)
                detection_records.append(serialize_detection_frame(frame_number, fps, frame, analysis))
                if now - last_metadata_write >= settings.metadata_write_seconds:
                    write_detection_metadata(job_id, detection_records)
                    last_metadata_write = now

                violations_to_save = rider_tracks.violations_to_save(
                    analysis["associations"], frame_number, frame, annotated
                )
                for payload in violations_to_save:
                    save_violation(job_id, payload)
                violation_count += len(violations_to_save)

                if now - last_preview_save >= 1:
                    latest_preview_url = save_preview(job_id, annotated)
                    last_preview_save = now

            if frame_hub.has_viewers(job_id) and now - last_publish_at >= publish_interval:
                if annotated is None:
                    annotated = annotate_analysis(frame, frame_number, latest_analysis, fresh_analysis=False)
                publish_stream_frame(job_id, annotated)
                last_publish_at = now

            if now - last_status_update >= 1:
                elapsed = now - started
                message = status_message(sampled_count, violation_count, latest_analysis)
                if recorder_failed:
                    message += " (recording unavailable: no video encoder)"
                update_job(
                    job_id,
                    "processing",
                    message,
                    progress=0,
                    current_frame=frame_number,
                    total_frames=0,
                    sampled_frames=sampled_count,
                    violation_count=violation_count,
                    elapsed_seconds=round(elapsed, 1),
                    processing_fps=round(frame_number / elapsed, 1) if elapsed > 0 else 0,
                    eta_seconds=0,
                    preview_image=latest_preview_url,
                    result="processing",
                )
                last_status_update = now

        pending_violations = rider_tracks.pending_violations_to_save()
        for payload in pending_violations:
            save_violation(job_id, payload)
        violation_count += len(pending_violations)
        write_detection_metadata(job_id, detection_records)

        elapsed = monotonic() - started
        result = "violations_detected" if violation_count else "no_violations"
        summary = (
            f"{end_reason}. Detected {violation_count} helmet violation(s) in {round(elapsed)}s"
            if violation_count
            else f"{end_reason}. No violations detected in {round(elapsed)}s ({sampled_count} sampled frame(s))"
        )
        if recorder_failed:
            summary += ". Recording unavailable: no video encoder"
        update_job(
            job_id,
            "completed",
            summary,
            progress=100,
            current_frame=frame_number,
            total_frames=frame_number,
            sampled_frames=sampled_count,
            violation_count=violation_count,
            elapsed_seconds=round(elapsed, 1),
            processing_fps=round(frame_number / elapsed, 1) if elapsed > 0 else 0,
            eta_seconds=0,
            result=result,
        )
    except Exception as exc:
        update_job(job_id, "failed", f"Live processing error: {exc}", result="failed")
    finally:
        if recorder is not None:
            recorder.release()
        capture.release()
        frame_hub.close(job_id)
        clear_stop(job_id)
