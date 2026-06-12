import os
import json
from pathlib import Path
import re
from time import monotonic, sleep
from uuid import uuid4

import cv2

from app.core.config import settings
from app.core.database import utc_now
from app.services.byte_tracker import ByteTrackDetection, ByteTracker
from app.services.repository import create_violation, update_job
from app.services.streaming import frame_hub

os.environ.setdefault("YOLO_CONFIG_DIR", str(settings.cache_dir / "ultralytics"))

PERSON_CLASS_ID = 0
MOTORCYCLE_CLASS_ID = 3
WITH_HELMET_LABEL = "with helmet"
NO_HELMET_LABEL = "without helmet"

_object_model = None
_helmet_model = None
_plate_model = None
_ocr_reader = None


class AssociationTracker:
    def __init__(
        self,
        cooldown_frames: int,
        aggregation_frames: int,
        min_samples: int,
        max_lost_frames: int,
    ):
        self.cooldown_frames = cooldown_frames
        self.aggregation_frames = aggregation_frames
        self.min_samples = max(min_samples, 1)
        self.tracker = ByteTracker(
            high_threshold=settings.tracker_high_confidence,
            low_threshold=settings.tracker_low_confidence,
            new_track_threshold=settings.tracker_new_track_confidence,
            match_threshold=settings.tracker_match_threshold,
            max_time_lost=max_lost_frames,
        )
        self.violation_tracks: dict[int, dict] = {}

    def update_association_tracks(self, associations: list[dict], frame_number: int) -> list[dict]:
        detections = []
        for index, association in enumerate(associations):
            detections.append(
                ByteTrackDetection(
                    xyxy=association_reference_box(association),
                    score=association_track_score(association),
                    metadata={"index": index},
                )
            )

        tracked_detections = self.tracker.update(detections, frame_number)
        for tracked_detection in tracked_detections:
            index = tracked_detection.metadata["index"]
            associations[index]["track_id"] = tracked_detection.track_id
            associations[index]["track_score"] = round(tracked_detection.score, 4)
            associations[index]["track_hits"] = tracked_detection.hits

        self.prune(frame_number)
        return associations

    def violation_associations_to_save(
        self, associations: list[dict], frame_number: int, frame, annotated
    ) -> list[dict]:
        ready = []
        for association in associations:
            if association.get("helmet_status") != "no_helmet":
                continue

            track_id = association.get("track_id")
            if track_id is None:
                continue

            track = self.violation_track(track_id, frame_number)
            track["last_frame"] = frame_number
            if frame_number - track["last_saved_frame"] < self.cooldown_frames:
                continue

            self.update_pending_violation(track, association, frame_number, frame, annotated)
            if not self.pending_ready(track, frame_number):
                continue

            ready.append(self.violation_payload(track))
            track["last_saved_frame"] = frame_number
            self.clear_pending(track)
        return ready

    def violation_track(self, track_id: int, frame_number: int) -> dict:
        existing = self.violation_tracks.get(track_id)
        if existing:
            return existing
        track = {
            "id": track_id,
            "last_frame": frame_number,
            "last_saved_frame": -self.cooldown_frames,
            "pending_started_frame": None,
            "pending_samples": 0,
            "pending_association": None,
            "pending_frame_number": None,
            "pending_frame": None,
            "pending_annotated": None,
            "best_plate_candidate": None,
        }
        self.violation_tracks[track_id] = track
        return track

    def update_pending_violation(
        self, track: dict, association: dict, frame_number: int, frame, annotated
    ) -> None:
        if track["pending_started_frame"] is None:
            track["pending_started_frame"] = frame_number
            track["pending_samples"] = 0
            track["best_plate_candidate"] = None

        track["pending_samples"] += 1
        track["pending_association"] = association.copy()
        track["pending_frame_number"] = frame_number
        track["pending_frame"] = frame.copy()
        track["pending_annotated"] = annotated.copy()

        candidate = build_plate_candidate(frame, association)
        if not candidate:
            return

        best_candidate = track.get("best_plate_candidate")
        if not best_candidate or candidate["score"] > best_candidate["score"]:
            track["best_plate_candidate"] = candidate

    def pending_ready(self, track: dict, frame_number: int) -> bool:
        if track["pending_started_frame"] is None:
            return False
        waited_long_enough = (
            frame_number - track["pending_started_frame"] >= self.aggregation_frames
        )
        sampled_enough = track["pending_samples"] >= self.min_samples
        return waited_long_enough or sampled_enough

    def violation_payload(self, track: dict) -> dict:
        association = track["pending_association"].copy()
        candidate = track.get("best_plate_candidate")
        if candidate:
            association["plate_box"] = candidate["plate_box"]

        return {
            "frame_number": track["pending_frame_number"],
            "frame": track["pending_frame"],
            "annotated": track["pending_annotated"],
            "association": association,
            "plate_candidate": candidate,
        }

    def pending_violations_to_save(self) -> list[dict]:
        ready = []
        for track in self.violation_tracks.values():
            if track["pending_started_frame"] is None:
                continue
            ready.append(self.violation_payload(track))
            track["last_saved_frame"] = track["pending_frame_number"]
            self.clear_pending(track)
        return ready

    def clear_pending(self, track: dict) -> None:
        track["pending_started_frame"] = None
        track["pending_samples"] = 0
        track["pending_association"] = None
        track["pending_frame_number"] = None
        track["pending_frame"] = None
        track["pending_annotated"] = None
        track["best_plate_candidate"] = None

    def prune(self, frame_number: int) -> None:
        max_age = max(self.cooldown_frames * 2, 1)
        active_track_ids = self.tracker.active_track_ids()
        self.violation_tracks = {
            track_id: track
            for track_id, track in self.violation_tracks.items()
            if track["pending_started_frame"] is not None
            or track_id in active_track_ids
            or frame_number - track["last_frame"] <= max_age
        }


def media_url(path: Path) -> str:
    return f"/media/{path.relative_to(settings.data_dir).as_posix()}"


def detection_metadata_path(job_id: str) -> Path:
    return settings.metadata_dir / f"{job_id}_detections.json"


def write_detection_metadata(job_id: str, records: list[dict]) -> None:
    payload = {"frames": records}
    detection_metadata_path(job_id).write_text(json.dumps(payload), encoding="utf-8")


def process_uploaded_video(job_id: str, source_path: str) -> None:
    update_job(
        job_id,
        "processing",
        "Loading detection models",
        progress=0,
        current_frame=0,
        sampled_frames=0,
        violation_count=0,
        elapsed_seconds=0,
        processing_fps=0,
        eta_seconds=0,
        result="processing",
    )

    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        update_job(job_id, "failed", "Could not open the uploaded video", result="failed")
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    analysis_interval = max(int(round(fps * settings.sample_every_seconds)), 1)
    preview_interval = preview_interval_for_fps(fps)
    cooldown_frames = max(int(fps * settings.violation_cooldown_seconds), analysis_interval)
    aggregation_frames = max(int(fps * settings.plate_aggregation_seconds), analysis_interval)
    max_lost_frames = max(int(fps * settings.tracker_max_lost_seconds), analysis_interval)
    frame_number = 0
    sampled_count = 0
    violation_count = 0
    association_tracker = AssociationTracker(
        cooldown_frames,
        aggregation_frames,
        settings.plate_aggregation_min_samples,
        max_lost_frames,
    )
    latest_analysis = empty_analysis()
    latest_preview_url = None
    last_status_update = 0.0
    last_preview_save = 0.0
    playback_started = monotonic()
    detection_records: list[dict] = []
    write_detection_metadata(job_id, detection_records)

    try:
        models = get_models()
        update_job(
            job_id,
            "processing",
            "Scanning video frames",
            total_frames=total_frames,
            progress=0,
            elapsed_seconds=0,
            processing_fps=0,
            eta_seconds=0,
        )

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_analyze = frame_number % analysis_interval == 0
            if should_analyze:
                sampled_count += 1
                analysis = analyze_frame(frame, models)
                association_tracker.update_association_tracks(
                    analysis["associations"], frame_number
                )
                latest_analysis = analysis
                analysis_annotated = annotate_analysis(
                    frame, frame_number, analysis, fresh_analysis=True
                )
                detection_records.append(
                    serialize_detection_frame(frame_number, fps, frame, analysis)
                )
                write_detection_metadata(job_id, detection_records)

                violations_to_save = association_tracker.violation_associations_to_save(
                    analysis["associations"], frame_number, frame, analysis_annotated
                )
                if violations_to_save:
                    for payload in violations_to_save:
                        save_violation(job_id, payload)
                    violation_count += len(violations_to_save)
            else:
                analysis = latest_analysis
                analysis_annotated = None

            should_publish = should_analyze or frame_number % preview_interval == 0
            if should_publish:
                annotated = (
                    analysis_annotated
                    if analysis_annotated is not None
                    else annotate_analysis(frame, frame_number, analysis, fresh_analysis=should_analyze)
                )
                publish_stream_frame(job_id, annotated)

                now = monotonic()
                if should_analyze or now - last_preview_save >= 1:
                    latest_preview_url = save_preview(job_id, annotated)
                    last_preview_save = now

            now = monotonic()
            should_update_status = should_analyze or now - last_status_update >= 1
            if should_update_status:
                progress = progress_for_frame(frame_number, total_frames)
                elapsed_seconds, processing_fps, eta_seconds = timing_metrics(
                    frame_number + 1,
                    total_frames,
                    playback_started,
                    now,
                )

                update_job(
                    job_id,
                    "processing",
                    status_message(sampled_count, violation_count, analysis),
                    progress=progress,
                    current_frame=frame_number,
                    total_frames=total_frames,
                    sampled_frames=sampled_count,
                    violation_count=violation_count,
                    elapsed_seconds=elapsed_seconds,
                    processing_fps=processing_fps,
                    eta_seconds=eta_seconds,
                    preview_image=latest_preview_url,
                    result="processing",
                )
                last_status_update = now

                if violation_count >= settings.max_violations_per_video:
                    break

            pace_preview(frame_number, fps, playback_started)
            frame_number += 1

        pending_violations = association_tracker.pending_violations_to_save()
        for payload in pending_violations:
            save_violation(job_id, payload)
        violation_count += len(pending_violations)

        result = "violations_detected" if violation_count else "no_violations"
        message = (
            f"Detected {violation_count} helmet violation(s)"
            if violation_count
            else f"No violations detected in {sampled_count} sampled frame(s)"
        )
        elapsed_seconds, processing_fps, _eta_seconds = timing_metrics(
            max(frame_number, 0),
            total_frames,
            playback_started,
            monotonic(),
        )
        update_job(
            job_id,
            "completed",
            message,
            progress=100,
            current_frame=max(frame_number, 0),
            total_frames=total_frames,
            sampled_frames=sampled_count,
            violation_count=violation_count,
            elapsed_seconds=elapsed_seconds,
            processing_fps=processing_fps,
            eta_seconds=0,
            result=result,
        )
    except Exception as exc:
        elapsed_seconds, processing_fps, _eta_seconds = timing_metrics(
            max(frame_number, 0),
            total_frames,
            playback_started,
            monotonic(),
        )
        update_job(
            job_id,
            "failed",
            f"Processing error: {exc}",
            elapsed_seconds=elapsed_seconds,
            processing_fps=processing_fps,
            eta_seconds=0,
            result="failed",
        )
    finally:
        capture.release()
        frame_hub.close(job_id)


def get_models():
    global _object_model, _helmet_model, _plate_model
    if _object_model is None or _helmet_model is None or _plate_model is None:
        from ultralytics import YOLO

        _object_model = YOLO(str(settings.object_model_path))
        _helmet_model = YOLO(str(settings.helmet_model_path))
        _plate_model = YOLO(str(settings.plate_model_path))
    return _object_model, _helmet_model, _plate_model


def analyze_frame(frame, models) -> dict:
    object_model, helmet_model, plate_model = models

    object_result = object_model.predict(
        frame.copy(),
        classes=[PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID],
        conf=settings.object_confidence,
        imgsz=settings.object_imgsz,
        verbose=False,
    )[0]
    object_boxes = extract_boxes(object_result)
    motorcycles = [box for box in object_boxes if box["class_id"] == MOTORCYCLE_CLASS_ID]
    people = [box for box in object_boxes if box["class_id"] == PERSON_CLASS_ID]

    helmet_result = helmet_model.predict(
        frame.copy(),
        conf=settings.helmet_confidence,
        imgsz=settings.helmet_imgsz,
        verbose=False,
    )[0]
    helmet_boxes = extract_boxes(helmet_result)
    no_helmet_boxes = [
        box for box in helmet_boxes if normalize_label(box["label"]) == NO_HELMET_LABEL
    ]
    with_helmet_boxes = [
        box for box in helmet_boxes if normalize_label(box["label"]) == WITH_HELMET_LABEL
    ]

    plate_result = plate_model.predict(
        frame.copy(),
        conf=settings.plate_confidence,
        imgsz=settings.plate_imgsz,
        verbose=False,
    )[0]
    plate_boxes = extract_boxes(plate_result)
    associations = associate_riders(people, motorcycles, with_helmet_boxes, no_helmet_boxes, plate_boxes)
    no_helmet_associations = [
        association for association in associations if association["helmet_status"] == "no_helmet"
    ]
    primary_violation = max(
        no_helmet_associations,
        key=lambda association: association["association_score"],
        default=None,
    )
    no_helmet = primary_violation["helmet_box"] if primary_violation else None
    plate = primary_violation["plate_box"] if primary_violation else None

    return {
        "objects": object_boxes,
        "motorcycles": motorcycles,
        "people": people,
        "helmets": with_helmet_boxes,
        "no_helmets": no_helmet_boxes,
        "plates": plate_boxes,
        "associations": associations,
        "helmet_box": no_helmet,
        "plate_box": plate,
        "has_no_helmet": bool(no_helmet_associations),
    }


def empty_analysis() -> dict:
    return {
        "objects": [],
        "motorcycles": [],
        "people": [],
        "helmets": [],
        "no_helmets": [],
        "plates": [],
        "associations": [],
        "helmet_box": None,
        "plate_box": None,
        "has_no_helmet": False,
    }


def serialize_detection_frame(frame_number: int, fps: float, frame, analysis: dict) -> dict:
    height, width = frame.shape[:2]
    return {
        "frame_number": frame_number,
        "timestamp": frame_number / fps if fps > 0 else 0,
        "width": width,
        "height": height,
        "people": serialize_boxes(analysis.get("people", [])),
        "motorcycles": serialize_boxes(analysis.get("motorcycles", [])),
        "helmets": serialize_boxes(analysis.get("helmets", [])),
        "no_helmets": serialize_boxes(analysis.get("no_helmets", [])),
        "plates": serialize_boxes(analysis.get("plates", [])),
        "associations": [
            serialize_association(association)
            for association in analysis.get("associations", [])
            if association.get("helmet_status") == "no_helmet"
        ],
    }


def serialize_boxes(boxes: list[dict]) -> list[dict]:
    return [
        {
            "label": box["label"],
            "confidence": box["confidence"],
            "xyxy": box["xyxy"],
        }
        for box in boxes
    ]


def serialize_association(association: dict) -> dict:
    return {
        "track_id": association.get("track_id"),
        "track_hits": association.get("track_hits", 0),
        "helmet_status": association.get("helmet_status"),
        "association_score": association.get("association_score", 0),
        "person_box": serialize_optional_box(association.get("person_box")),
        "motorcycle_box": serialize_optional_box(association.get("motorcycle_box")),
        "helmet_box": serialize_optional_box(association.get("helmet_box")),
        "plate_box": serialize_optional_box(association.get("plate_box")),
    }


def serialize_optional_box(box: dict | None) -> dict | None:
    if not box:
        return None
    return {
        "label": box["label"],
        "confidence": box["confidence"],
        "xyxy": box["xyxy"],
    }


def preview_interval_for_fps(source_fps: float) -> int:
    target_fps = max(settings.live_preview_fps, 1)
    return max(int(round(source_fps / target_fps)), 1)


def pace_preview(frame_number: int, source_fps: float, started_at: float) -> None:
    if not settings.realtime_preview or source_fps <= 0:
        return

    target_elapsed = (frame_number + 1) / source_fps
    delay = target_elapsed - (monotonic() - started_at)
    if delay > 0:
        sleep(min(delay, 0.25))


def save_preview(job_id: str, annotated) -> str:
    preview_path = settings.preview_dir / f"{job_id}_latest.jpg"
    cv2.imwrite(str(preview_path), annotated)
    return media_url(preview_path)


def publish_stream_frame(job_id: str, annotated) -> None:
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if ok:
        frame_hub.publish(job_id, encoded.tobytes())


def save_violation(job_id: str, payload: dict) -> None:
    frame_number = payload["frame_number"]
    frame = payload["frame"]
    annotated = payload["annotated"]
    association = payload["association"]
    plate_candidate = payload.get("plate_candidate")
    violation_id = uuid4().hex
    evidence_path = settings.evidence_dir / f"{job_id}_{frame_number}_{violation_id[:8]}.jpg"
    cv2.imwrite(str(evidence_path), annotated)

    plate_path = None
    plate_text = None
    plate_confidence = None
    if plate_candidate:
        plate_path = settings.plate_dir / f"{job_id}_{frame_number}_{violation_id[:8]}_plate.jpg"
        cv2.imwrite(str(plate_path), plate_candidate["crop"])
        plate_text = plate_candidate.get("plate_text")
        plate_confidence = plate_candidate.get("plate_confidence")
    else:
        plate_box = association.get("plate_box")
        if plate_box:
            plate_path = settings.plate_dir / f"{job_id}_{frame_number}_{violation_id[:8]}_plate.jpg"
            crop = crop_box(frame, plate_box["xyxy"], padding=8)
            if crop is not None and crop.size:
                cv2.imwrite(str(plate_path), crop)
                plate_text, plate_confidence = read_plate_text(crop)
            else:
                plate_path = None

    helmet_box = association["helmet_box"]
    create_violation(
        {
            "id": violation_id,
            "job_id": job_id,
            "detected_at": utc_now(),
            "helmet_status": "no_helmet",
            "helmet_confidence": helmet_box["confidence"],
            "plate_text": plate_text,
            "plate_confidence": plate_confidence,
            "evidence_image": media_url(evidence_path),
            "plate_image": media_url(plate_path) if plate_path else None,
            "frame_number": frame_number,
            "track_id": association.get("track_id"),
        }
    )


def build_plate_candidate(frame, association: dict) -> dict | None:
    plate_box = association.get("plate_box")
    if not plate_box:
        return None

    crop = crop_box(frame, plate_box["xyxy"], padding=8)
    if crop is None or not crop.size:
        return None

    plate_text, plate_confidence = read_plate_text(crop)
    return {
        "crop": crop.copy(),
        "plate_box": plate_box,
        "plate_text": plate_text,
        "plate_confidence": plate_confidence,
        "score": plate_candidate_score(crop, plate_box, plate_text, plate_confidence),
    }


def plate_candidate_score(
    crop,
    plate_box: dict,
    plate_text: str | None,
    plate_confidence: float | None,
) -> float:
    height, width = crop.shape[:2]
    size_score = min((width * height) / 12000, 1.0)
    sharpness_score = min(crop_sharpness(crop) / 450, 1.0)
    detector_score = plate_box["confidence"]
    ocr_score = plate_confidence or 0.0
    text_bonus = 0.10 if plate_text else 0.0
    return round(
        0.30 * detector_score
        + 0.25 * size_score
        + 0.25 * sharpness_score
        + 0.20 * ocr_score
        + text_bonus,
        4,
    )


def crop_sharpness(crop) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_boxes(result) -> list[dict]:
    names = result.names
    boxes = []
    if result.boxes is None:
        return boxes

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        xyxy = [int(value) for value in box.xyxy[0].tolist()]
        boxes.append(
            {
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
                "confidence": confidence,
                "xyxy": clamp_box(xyxy, result.orig_shape[1], result.orig_shape[0]),
            }
        )
    return boxes


def associate_riders(
    people: list[dict],
    motorcycles: list[dict],
    with_helmet_boxes: list[dict],
    no_helmet_boxes: list[dict],
    plate_boxes: list[dict],
) -> list[dict]:
    helmet_detections = [
        {"box": box, "status": "with_helmet"} for box in with_helmet_boxes
    ] + [{"box": box, "status": "no_helmet"} for box in no_helmet_boxes]
    associations = []
    assigned_helmet_ids: set[int] = set()

    for person in people:
        helmet_detection, helmet_score = best_helmet_for_person(person, helmet_detections)
        if not helmet_detection or helmet_score < 0.20:
            continue

        helmet_box = helmet_detection["box"]
        motorcycle, motorcycle_score = best_motorcycle_for_person(person, motorcycles)
        plate, plate_score = best_plate_for_motorcycle(motorcycle, plate_boxes)
        association_score = combined_score(
            helmet_box["confidence"],
            helmet_score,
            motorcycle_score,
            plate_score,
        )
        associations.append(
            {
                "person_box": person,
                "motorcycle_box": motorcycle,
                "helmet_box": helmet_box,
                "helmet_status": helmet_detection["status"],
                "plate_box": plate,
                "association_score": association_score,
            }
        )
        assigned_helmet_ids.add(id(helmet_box))

    for helmet_box in no_helmet_boxes:
        if id(helmet_box) in assigned_helmet_ids:
            continue

        person, helmet_score = best_person_for_helmet(helmet_box, people)
        if person:
            motorcycle, motorcycle_score = best_motorcycle_for_person(person, motorcycles)
        else:
            motorcycle, motorcycle_score = best_motorcycle_for_helmet(helmet_box, motorcycles)

        plate, plate_score = best_plate_for_motorcycle(motorcycle, plate_boxes)
        if not plate:
            plate, plate_score = best_plate_for_helmet(helmet_box, plate_boxes)

        association_score = combined_score(
            helmet_box["confidence"],
            helmet_score,
            motorcycle_score,
            plate_score,
        )
        associations.append(
            {
                "person_box": person,
                "motorcycle_box": motorcycle,
                "helmet_box": helmet_box,
                "helmet_status": "no_helmet",
                "plate_box": plate,
                "association_score": association_score,
            }
        )

    return sorted(associations, key=lambda association: association["association_score"], reverse=True)


def best_helmet_for_person(person: dict, helmet_detections: list[dict]) -> tuple[dict | None, float]:
    best_detection = None
    best_score = 0.0
    for detection in helmet_detections:
        score = score_helmet_to_person(detection["box"], person)
        if score > best_score:
            best_detection = detection
            best_score = score
    return best_detection, best_score


def best_person_for_helmet(helmet_box: dict, people: list[dict]) -> tuple[dict | None, float]:
    best_person = None
    best_score = 0.0
    for person in people:
        score = score_helmet_to_person(helmet_box, person)
        if score > best_score:
            best_person = person
            best_score = score
    return best_person, best_score


def best_motorcycle_for_person(person: dict | None, motorcycles: list[dict]) -> tuple[dict | None, float]:
    if not person:
        return None, 0.0

    best_motorcycle = None
    best_score = 0.0
    for motorcycle in motorcycles:
        score = score_person_to_motorcycle(person, motorcycle)
        if score > best_score:
            best_motorcycle = motorcycle
            best_score = score
    return best_motorcycle, best_score


def best_motorcycle_for_helmet(helmet_box: dict, motorcycles: list[dict]) -> tuple[dict | None, float]:
    best_motorcycle = None
    best_score = 0.0
    for motorcycle in motorcycles:
        score = score_helmet_to_motorcycle(helmet_box, motorcycle)
        if score > best_score:
            best_motorcycle = motorcycle
            best_score = score
    return best_motorcycle, best_score


def best_plate_for_motorcycle(motorcycle: dict | None, plate_boxes: list[dict]) -> tuple[dict | None, float]:
    if not motorcycle:
        return None, 0.0

    best_plate = None
    best_score = 0.0
    for plate in plate_boxes:
        score = score_plate_to_motorcycle(plate, motorcycle)
        if score > best_score:
            best_plate = plate
            best_score = score
    return best_plate, best_score


def best_plate_for_helmet(helmet_box: dict, plate_boxes: list[dict]) -> tuple[dict | None, float]:
    best_plate = None
    best_score = 0.0
    for plate in plate_boxes:
        score = score_plate_to_helmet(plate, helmet_box)
        if score > best_score:
            best_plate = plate
            best_score = score
    return best_plate, best_score


def combined_score(
    helmet_confidence: float,
    helmet_score: float,
    motorcycle_score: float,
    plate_score: float,
) -> float:
    return round(
        0.30 * helmet_confidence
        + 0.30 * helmet_score
        + 0.25 * motorcycle_score
        + 0.15 * plate_score,
        4,
    )


def score_helmet_to_person(helmet_box: dict, person: dict) -> float:
    hx, hy = box_center(helmet_box["xyxy"])
    x1, y1, x2, y2 = person["xyxy"]
    person_width = max(x2 - x1, 1)
    person_height = max(y2 - y1, 1)
    upper_person = [x1, y1, x2, int(y1 + person_height * 0.62)]
    expanded_upper = expand_box(upper_person, 0.20)
    top_center = ((x1 + x2) / 2, y1 + person_height * 0.20)
    normalized_distance = point_distance((hx, hy), top_center) / max(person_width, person_height)
    center_bonus = 0.55 if point_in_box((hx, hy), expanded_upper) else 0.0
    overlap_bonus = min(box_iou(helmet_box["xyxy"], expanded_upper) * 2.0, 0.30)
    distance_score = max(0.0, 1.0 - normalized_distance) * 0.15
    return min(center_bonus + overlap_bonus + distance_score, 1.0)


def score_person_to_motorcycle(person: dict, motorcycle: dict) -> float:
    px1, py1, px2, py2 = person["xyxy"]
    person_height = max(py2 - py1, 1)
    lower_person = [px1, int(py1 + person_height * 0.35), px2, py2]
    lower_center = ((px1 + px2) / 2, py1 + person_height * 0.82)
    expanded_motorcycle = expand_box(motorcycle["xyxy"], 0.25)
    mx1, my1, mx2, my2 = motorcycle["xyxy"]
    motorcycle_width = max(mx2 - mx1, 1)
    motorcycle_height = max(my2 - my1, 1)
    motorcycle_center = box_center(motorcycle["xyxy"])
    normalized_distance = point_distance(lower_center, motorcycle_center) / max(
        motorcycle_width, motorcycle_height
    )
    center_bonus = 0.45 if point_in_box(lower_center, expanded_motorcycle) else 0.0
    overlap_bonus = min(box_iou(lower_person, expanded_motorcycle) * 1.5, 0.35)
    distance_score = max(0.0, 1.0 - normalized_distance) * 0.20
    return min(center_bonus + overlap_bonus + distance_score, 1.0)


def score_helmet_to_motorcycle(helmet_box: dict, motorcycle: dict) -> float:
    hx, hy = box_center(helmet_box["xyxy"])
    mx1, my1, mx2, my2 = motorcycle["xyxy"]
    motorcycle_width = max(mx2 - mx1, 1)
    target = ((mx1 + mx2) / 2, my1)
    normalized_distance = point_distance((hx, hy), target) / max(motorcycle_width, 1)
    x_bonus = 0.35 if mx1 <= hx <= mx2 else 0.0
    y_bonus = 0.25 if hy <= my2 else 0.0
    distance_score = max(0.0, 1.0 - normalized_distance) * 0.40
    return min(x_bonus + y_bonus + distance_score, 1.0)


def score_plate_to_motorcycle(plate: dict, motorcycle: dict) -> float:
    px, py = box_center(plate["xyxy"])
    mx1, my1, mx2, my2 = motorcycle["xyxy"]
    motorcycle_width = max(mx2 - mx1, 1)
    motorcycle_height = max(my2 - my1, 1)
    expanded_motorcycle = expand_box(motorcycle["xyxy"], 0.20)
    lower_target = ((mx1 + mx2) / 2, my1 + motorcycle_height * 0.72)
    normalized_distance = point_distance((px, py), lower_target) / max(
        motorcycle_width, motorcycle_height
    )
    center_bonus = 0.45 if point_in_box((px, py), expanded_motorcycle) else 0.0
    lower_bonus = 0.20 if py >= my1 + motorcycle_height * 0.35 else 0.0
    distance_score = max(0.0, 1.0 - normalized_distance) * 0.25
    confidence_score = plate["confidence"] * 0.10
    return min(center_bonus + lower_bonus + distance_score + confidence_score, 1.0)


def score_plate_to_helmet(plate: dict, helmet_box: dict) -> float:
    plate_center = box_center(plate["xyxy"])
    helmet_center = box_center(helmet_box["xyxy"])
    vertical_gap = max(plate_center[1] - helmet_center[1], 0)
    horizontal_gap = abs(plate_center[0] - helmet_center[0])
    _, y1, _, y2 = helmet_box["xyxy"]
    reference = max(y2 - y1, 1) * 8
    if vertical_gap <= 0:
        return 0.0
    distance_score = max(0.0, 1.0 - (horizontal_gap / max(reference, 1))) * 0.55
    vertical_score = min(vertical_gap / max(reference, 1), 1.0) * 0.25
    confidence_score = plate["confidence"] * 0.20
    return min(distance_score + vertical_score + confidence_score, 1.0)


def annotate_analysis(frame, frame_number: int, analysis: dict, *, fresh_analysis: bool):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    banner_height = max(52, height // 13)
    status = "VIOLATION" if analysis["has_no_helmet"] else "SCANNING"
    banner_color = (35, 35, 190) if analysis["has_no_helmet"] else (20, 20, 20)
    analysis_age = "detected" if fresh_analysis else "live preview"

    cv2.rectangle(annotated, (0, 0), (width, banner_height), banner_color, -1)
    cv2.putText(
        annotated,
        f"SafeRide {analysis_age} | {status} | frame {frame_number}",
        (24, min(42, banner_height - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for person in analysis.get("people", []):
        draw_box(annotated, person, (220, 180, 80), f"person {person['confidence']:.2f}")

    for motorcycle in analysis.get("motorcycles", []):
        draw_box(annotated, motorcycle, (45, 125, 255), f"motorcycle {motorcycle['confidence']:.2f}")

    for helmet in analysis.get("helmets", []):
        draw_box(annotated, helmet, (40, 175, 70), f"helmet {helmet['confidence']:.2f}")

    for no_helmet in analysis.get("no_helmets", []):
        draw_box(annotated, no_helmet, (35, 35, 230), f"no helmet {no_helmet['confidence']:.2f}")

    for plate in analysis.get("plates", []):
        draw_box(annotated, plate, (30, 170, 220), f"plate {plate['confidence']:.2f}")

    for association in analysis.get("associations", []):
        if association.get("helmet_status") == "no_helmet":
            draw_association(annotated, association)

    return annotated


def draw_box(image, box: dict, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box["xyxy"]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = max(y1 - 8, 18)
    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_association(image, association: dict) -> None:
    helmet_box = association.get("helmet_box")
    plate_box = association.get("plate_box")
    motorcycle_box = association.get("motorcycle_box")
    if not helmet_box:
        return

    helmet_center = tuple(int(value) for value in box_center(helmet_box["xyxy"]))
    cv2.circle(image, helmet_center, 4, (255, 255, 255), -1)
    if association.get("track_id") is not None:
        cv2.putText(
            image,
            f"track {association['track_id']}",
            (helmet_center[0] + 8, max(helmet_center[1] - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if motorcycle_box:
        motorcycle_center = tuple(int(value) for value in box_center(motorcycle_box["xyxy"]))
        cv2.line(image, helmet_center, motorcycle_center, (255, 255, 255), 2)

    if plate_box:
        plate_center = tuple(int(value) for value in box_center(plate_box["xyxy"]))
        start = tuple(int(value) for value in box_center(motorcycle_box["xyxy"])) if motorcycle_box else helmet_center
        cv2.line(image, start, plate_center, (30, 220, 255), 2)
        cv2.circle(image, plate_center, 4, (30, 220, 255), -1)


def crop_box(frame, xyxy: list[int], padding: int = 0):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(
        [xyxy[0] - padding, xyxy[1] - padding, xyxy[2] + padding, xyxy[3] + padding],
        width,
        height,
    )
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def clamp_box(xyxy: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = xyxy
    return [
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    ]


def expand_box(xyxy: list[int], ratio: float) -> list[int]:
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * ratio)
    pad_y = int(height * ratio)
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]


def box_center(xyxy: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_box(point: tuple[float, float], xyxy: list[int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def box_area(xyxy: list[int]) -> int:
    x1, y1, x2, y2 = xyxy
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = box_area(
        [
            max(ax1, bx1),
            max(ay1, by1),
            min(ax2, bx2),
            min(ay2, by2),
        ]
    )
    union = box_area(a) + box_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def association_reference_box(association: dict) -> list[int]:
    for key in ["motorcycle_box", "person_box", "helmet_box"]:
        box = association.get(key)
        if box:
            return box["xyxy"]
    return [0, 0, 0, 0]


def association_track_score(association: dict) -> float:
    helmet_box = association.get("helmet_box") or {}
    confidence = float(helmet_box.get("confidence", 0.0))
    association_score = float(association.get("association_score", 0.0))
    return max(confidence * 0.55 + association_score * 0.45, association_score)


def track_match_score(previous_xyxy: list[int], current_xyxy: list[int]) -> float:
    iou_score = box_iou(previous_xyxy, current_xyxy)
    previous_center = box_center(previous_xyxy)
    current_center = box_center(current_xyxy)
    previous_width = max(previous_xyxy[2] - previous_xyxy[0], 1)
    previous_height = max(previous_xyxy[3] - previous_xyxy[1], 1)
    normalized_distance = point_distance(previous_center, current_center) / max(
        previous_width, previous_height
    )
    distance_score = max(0.0, 1.0 - normalized_distance)
    return max(iou_score, distance_score * 0.70)


def normalize_label(label: str) -> str:
    return label.strip().replace("_", " ").lower()


def progress_for_frame(frame_number: int, total_frames: int) -> float:
    if total_frames <= 0:
        return 0
    return min(round((frame_number / total_frames) * 100, 1), 99)


def timing_metrics(
    processed_frames: int,
    total_frames: int,
    started_at: float,
    now: float,
) -> tuple[float, float, float]:
    elapsed_seconds = max(now - started_at, 0)
    processing_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 and processed_frames > 0 else 0
    remaining_frames = max(total_frames - processed_frames, 0)
    eta_seconds = remaining_frames / processing_fps if processing_fps > 0 and total_frames > 0 else 0
    return round(elapsed_seconds, 1), round(processing_fps, 1), round(eta_seconds, 1)


def status_message(sampled_count: int, violation_count: int, analysis: dict) -> str:
    parts = [
        f"Scanned {sampled_count} sampled frame(s)",
        f"{len(analysis['motorcycles'])} motorcycle(s)",
        f"{len(analysis['helmets'])} helmet(s)",
        f"{len(analysis['no_helmets'])} no-helmet rider(s)",
        f"{len(analysis['plates'])} plate(s)",
    ]
    if violation_count:
        parts.append(f"{violation_count} saved violation(s)")
    return ", ".join(parts)


def read_plate_text(crop) -> tuple[str | None, float | None]:
    if not settings.enable_ocr:
        return None, None

    global _ocr_reader
    try:
        if _ocr_reader is None:
            import easyocr

            _ocr_reader = easyocr.Reader(
                settings.ocr_languages,
                gpu=settings.ocr_gpu,
                model_storage_directory=str(settings.cache_dir / "easyocr"),
                user_network_directory=str(settings.cache_dir / "easyocr"),
                verbose=False,
            )
        results = []
        for image in plate_ocr_variants(crop):
            results.extend(_ocr_reader.readtext(image, detail=1, paragraph=False))
    except Exception:
        return None, None

    if not results:
        return None, None

    candidates = []
    for result in results:
        text = normalize_plate_text(str(result[1]))
        if text:
            candidates.append((text, float(result[2])))

    if not candidates:
        return None, None

    text, confidence = max(candidates, key=lambda candidate: candidate[1])
    return text, confidence


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


def normalize_plate_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = re.sub(r"[^\wก-๙\-\s]", "", text, flags=re.UNICODE)
    return text.strip(" -_")
