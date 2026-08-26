import os
import json
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

import cv2

from app.core.config import settings
from app.core.database import utc_now
from app.services.byte_tracker import ByteTrackDetection, ByteTracker
from app.services.plate_ocr import read_plate_text, vote_plate_texts
from app.services.repository import create_violation, update_job
from app.services.streaming import frame_hub

os.environ.setdefault("YOLO_CONFIG_DIR", str(settings.cache_dir / "ultralytics"))

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2
MOTORCYCLE_CLASS_ID = 3
BUS_CLASS_ID = 5
TRUCK_CLASS_ID = 7
NEGATIVE_VEHICLE_CLASS_IDS = {CAR_CLASS_ID, BUS_CLASS_ID, TRUCK_CLASS_ID}
WITH_HELMET_LABEL = "with helmet"
NO_HELMET_LABEL = "without helmet"

_object_model = None
_helmet_model = None
_plate_model = None
_model_device: str | None = None


def resolve_model_device() -> str:
    """Pick the inference device once: CUDA on NVIDIA machines, MPS on Apple
    Silicon, CPU otherwise. Override with MODEL_DEVICE=cpu/cuda/mps."""
    global _model_device
    if _model_device is not None:
        return _model_device
    if settings.model_device != "auto":
        _model_device = settings.model_device
        return _model_device
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        else:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                device = "mps"
    except Exception:
        device = "cpu"
    _model_device = device
    return device


def predict_kwargs() -> dict:
    device = resolve_model_device()
    # FP16 halves inference time on CUDA; MPS and CPU stay FP32.
    return {"device": device, "half": device == "cuda"}


class RiderTrackManager:
    """Anchors rider identity to motorcycle tracks and votes helmet status per track.

    Raw motorcycle detections are far more stable than gated rider associations,
    so the ByteTracker runs on motorcycle boxes every sampled frame. Helmet
    observations accumulate as per-track votes, and a violation only becomes
    eligible once a track has enough no-helmet votes that are not drowned out by
    with-helmet votes. This suppresses single-frame helmet-model flickers while
    still allowing mixed rider/passenger tracks through to human review.
    """

    def __init__(
        self,
        cooldown_frames: int,
        collection_frames: int,
        max_lost_frames: int,
        dedupe_frames: int,
    ):
        self.cooldown_frames = cooldown_frames
        self.collection_frames = max(collection_frames, 1)
        self.dedupe_frames = max(dedupe_frames, max_lost_frames, cooldown_frames)
        self.tracker = ByteTracker(
            high_threshold=settings.tracker_high_confidence,
            low_threshold=settings.tracker_low_confidence,
            new_track_threshold=settings.tracker_new_track_confidence,
            match_threshold=settings.tracker_match_threshold,
            max_time_lost=max_lost_frames,
            appearance_weight=settings.tracker_appearance_weight,
        )
        self.violation_tracks: dict[int, dict] = {}
        self.helmet_votes: dict[int, dict] = {}
        self.recent_saves: list[dict] = []
        self.saved_track_ids: set[int] = set()
        self.saved_violation_signatures: list[dict] = []

    def update(self, analysis: dict, frame_number: int, frame=None) -> None:
        motorcycles = analysis["motorcycles"]
        detections = [
            ByteTrackDetection(
                xyxy=motorcycle["xyxy"],
                score=motorcycle["confidence"],
                metadata={"index": index},
                feature=None if frame is None else appearance_feature(frame, motorcycle["xyxy"]),
            )
            for index, motorcycle in enumerate(motorcycles)
        ]

        tracked_detections = self.tracker.update(detections, frame_number)
        for tracked_detection in tracked_detections:
            motorcycle = motorcycles[tracked_detection.metadata["index"]]
            motorcycle["track_id"] = tracked_detection.track_id
            motorcycle["track_hits"] = tracked_detection.hits

            # Buffer a few strong plate crops for every motorcycle track. If
            # the no-helmet vote confirms later, early readable views are still
            # available; collection then continues even through helmet-model
            # flicker. Plate detection already ran, so this is cheap scoring.
            violation_track = self.violation_track(
                tracked_detection.track_id, frame_number
            )
            violation_track["last_frame"] = frame_number
            self.collect_plate_candidate(
                violation_track,
                motorcycle.get("plate_box"),
                frame_number,
                frame,
            )

        for association in analysis["associations"]:
            motorcycle = association.get("motorcycle_box")
            if not motorcycle or motorcycle.get("track_id") is None:
                continue
            association["track_id"] = motorcycle["track_id"]
            association["track_hits"] = motorcycle.get("track_hits", 0)
            association["track_score"] = round(association_track_score(association), 4)
            self.record_helmet_vote(
                motorcycle["track_id"], association.get("helmet_status"), frame_number
            )

        self.prune(frame_number)

    def record_helmet_vote(self, track_id: int, helmet_status: str | None, frame_number: int) -> None:
        if helmet_status not in ("no_helmet", "with_helmet"):
            return
        votes = self.helmet_votes.setdefault(
            track_id, {"no_helmet": 0, "with_helmet": 0, "last_frame": frame_number}
        )
        votes[helmet_status] += 1
        votes["last_frame"] = frame_number

    def no_helmet_vote_passes(self, track_id: int) -> bool:
        votes = self.helmet_votes.get(track_id)
        if not votes:
            return False
        no_helmet = votes["no_helmet"]
        with_helmet = votes["with_helmet"]
        if no_helmet < settings.min_no_helmet_votes:
            return False
        # A lone flicker against many with-helmet votes fails; a genuinely mixed
        # track (driver helmeted, passenger not) still reaches human review.
        return no_helmet * 2 >= with_helmet

    def violations_to_save(
        self, associations: list[dict], frame_number: int, frame, annotated
    ) -> list[dict]:
        ready = []
        for association in associations:
            if association.get("helmet_status") != "no_helmet":
                continue
            if not valid_no_helmet_association(association):
                continue

            track_id = association.get("track_id")
            if track_id is None:
                continue
            if not self.no_helmet_vote_passes(track_id):
                continue

            track = self.violation_track(track_id, frame_number)
            track["last_frame"] = frame_number
            if track["pending_started_frame"] is None:
                duplicate_signature = self.saved_duplicate_signature(association, frame_number)
                if duplicate_signature:
                    self.mark_duplicate_track(
                        track, duplicate_signature, association, frame_number
                    )
                    continue

                if frame_number - track["last_saved_frame"] < self.cooldown_frames:
                    continue

            self.update_pending_violation(track, association, frame_number, frame, annotated)

        for track in list(self.violation_tracks.values()):
            if not self.pending_ready(track, frame_number):
                continue
            payload = self.finalize_pending_track(track)
            if payload:
                ready.append(payload)
        return ready

    def is_duplicate_save(self, association: dict, frame_number: int) -> bool:
        """Suppress re-saves of the same physical rider under a churned track id.

        When the tracker splits one motorcycle across two track identities (box
        jitter, or camera pan in handheld footage), both tracks can pass the vote
        gate. Overlap alone is not enough under pan, so a save whose motorcycle
        center sits within ~0.8 bike-sizes of a recent save also counts as the
        same rider - distinct motorcycles ride well over one bike-size apart."""
        motorcycle = association.get("motorcycle_box")
        if not motorcycle:
            return False
        for save in self.recent_saves:
            if frame_number - save["frame"] > self.cooldown_frames:
                continue
            if box_iou(motorcycle["xyxy"], save["xyxy"]) >= 0.35:
                return True
            reference = max(
                save["xyxy"][2] - save["xyxy"][0],
                save["xyxy"][3] - save["xyxy"][1],
                motorcycle["xyxy"][2] - motorcycle["xyxy"][0],
                motorcycle["xyxy"][3] - motorcycle["xyxy"][1],
                1,
            )
            distance = point_distance(box_center(motorcycle["xyxy"]), box_center(save["xyxy"]))
            if distance / reference <= 1.0:
                return True
        return False

    def record_save(self, association: dict, frame_number: int) -> None:
        motorcycle = association.get("motorcycle_box")
        if not motorcycle:
            return
        self.recent_saves.append({"frame": frame_number, "xyxy": motorcycle["xyxy"]})
        self.recent_saves = [
            save for save in self.recent_saves
            if frame_number - save["frame"] <= self.cooldown_frames
        ]

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
            "plate_candidates": [],
            "plate_sightings": 0,
            "last_plate_frame": None,
        }
        self.violation_tracks[track_id] = track
        return track

    def update_pending_violation(
        self, track: dict, association: dict, frame_number: int, frame, annotated
    ) -> None:
        if track["pending_started_frame"] is None:
            track["pending_started_frame"] = frame_number
            track["pending_samples"] = 0

        track["pending_samples"] += 1
        track["pending_association"] = association.copy()
        track["pending_frame_number"] = frame_number
        track["pending_frame"] = frame.copy()
        track["pending_annotated"] = annotated.copy()

        self.collect_plate_candidate(
            track, association.get("plate_box"), frame_number, frame
        )

    def collect_plate_candidate(
        self,
        track: dict,
        plate_box: dict | None,
        frame_number: int,
        frame,
    ) -> None:
        if not plate_box or frame is None:
            return
        if track.get("last_plate_frame") == frame_number:
            return

        track["last_plate_frame"] = frame_number
        track["plate_sightings"] += 1
        candidate = build_plate_candidate(frame, plate_box, run_ocr=False)
        if not candidate:
            return

        candidates = track["plate_candidates"]
        candidates.append(candidate)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        del candidates[max(settings.plate_candidate_limit, 1):]

    def pending_ready(self, track: dict, frame_number: int) -> bool:
        if track["pending_started_frame"] is None:
            return False
        collection_expired = (
            frame_number - track["pending_started_frame"] >= self.collection_frames
        )
        track_ended = track["id"] not in self.tracker.active_track_ids()
        return collection_expired or track_ended

    def violation_payload(self, track: dict) -> dict:
        association = track["pending_association"].copy()
        candidate, ocr_reads = finalize_plate_candidates(track["plate_candidates"])

        # Co-travel gate: the plate must have been seen with this track in enough
        # samples, otherwise a one-off plate (a passing car's) is dropped rather
        # than attached to the rider.
        required_sightings = max(settings.plate_min_track_sightings, 1)
        if candidate and track["plate_sightings"] >= required_sightings:
            association["plate_box"] = candidate["plate_box"]
            voted = vote_plate_texts(ocr_reads)
            if voted:
                candidate = dict(candidate)
                candidate["plate_text"], candidate["plate_confidence"] = voted
        else:
            candidate = None
            association["plate_box"] = None

        return {
            "frame_number": track["pending_frame_number"],
            "frame": track["pending_frame"],
            "annotated": track["pending_annotated"],
            "association": association,
            "plate_candidate": candidate,
        }

    def pending_violations_to_save(self) -> list[dict]:
        ready = []
        for track in list(self.violation_tracks.values()):
            if track["pending_started_frame"] is None:
                continue
            payload = self.finalize_pending_track(track)
            if payload:
                ready.append(payload)
        return ready

    def finalize_pending_track(self, track: dict) -> dict | None:
        association = track["pending_association"]
        frame_number = track["pending_frame_number"]
        duplicate_signature = self.saved_duplicate_signature(association, frame_number)
        if duplicate_signature:
            self.mark_duplicate_track(track, duplicate_signature, association, frame_number)
            return None

        payload = self.violation_payload(track)
        frame_number = payload["frame_number"]
        duplicate = self.is_duplicate_save(payload["association"], frame_number)
        # Record the location either way so a moving rider keeps extending the
        # dedup chain even while its saves are being suppressed.
        self.record_save(payload["association"], frame_number)
        self.mark_track_saved(track, payload["association"], frame_number)
        self.clear_pending(track)
        return None if duplicate else payload

    def saved_duplicate_signature(self, association: dict, frame_number: int) -> dict | None:
        track_id = association.get("track_id")
        if track_id in self.saved_track_ids:
            return self.saved_signature_for_track(track_id)

        for signature in self.saved_violation_signatures:
            if track_id in signature["track_ids"]:
                return signature

        for signature in self.saved_violation_signatures:
            if frame_number - signature["frame_number"] > self.dedupe_frames:
                continue
            if association_signature_score(signature, association) >= settings.rider_dedupe_match_threshold:
                return signature
        return None

    def mark_track_saved(self, track: dict, association: dict, frame_number: int) -> None:
        track_id = track["id"]
        track["last_saved_frame"] = frame_number
        self.saved_track_ids.add(track_id)

        signature = self.saved_signature_for_track(track_id)
        if signature:
            self.update_saved_signature(signature, association, frame_number)
            return

        self.saved_violation_signatures.append(
            {
                "track_ids": {track_id},
                "frame_number": frame_number,
                "reference_box": copy_xyxy(association_reference_box(association)),
                "motorcycle_box": association_box_xyxy(association, "motorcycle_box"),
                "person_box": association_box_xyxy(association, "person_box"),
                "helmet_box": association_box_xyxy(association, "helmet_box"),
                "plate_box": association_box_xyxy(association, "plate_box"),
            }
        )

    def mark_duplicate_track(
        self, track: dict, signature: dict, association: dict, frame_number: int
    ) -> None:
        track_id = track["id"]
        track["last_saved_frame"] = frame_number
        self.saved_track_ids.add(track_id)
        signature["track_ids"].add(track_id)
        self.update_saved_signature(signature, association, frame_number)
        self.clear_pending(track)

    def saved_signature_for_track(self, track_id: int) -> dict | None:
        for signature in self.saved_violation_signatures:
            if track_id in signature["track_ids"]:
                return signature
        return None

    def update_saved_signature(
        self, signature: dict, association: dict, frame_number: int
    ) -> None:
        signature["frame_number"] = frame_number
        signature["reference_box"] = copy_xyxy(association_reference_box(association))
        signature["motorcycle_box"] = association_box_xyxy(association, "motorcycle_box")
        signature["person_box"] = association_box_xyxy(association, "person_box")
        signature["helmet_box"] = association_box_xyxy(association, "helmet_box")
        signature["plate_box"] = association_box_xyxy(association, "plate_box")

    def clear_pending(self, track: dict) -> None:
        track["pending_started_frame"] = None
        track["pending_samples"] = 0
        track["pending_association"] = None
        track["pending_frame_number"] = None
        track["pending_frame"] = None
        track["pending_annotated"] = None
        track["plate_candidates"] = []
        track["plate_sightings"] = 0
        track["last_plate_frame"] = None

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
        self.helmet_votes = {
            track_id: votes
            for track_id, votes in self.helmet_votes.items()
            if track_id in active_track_ids
            or track_id in self.violation_tracks
            or frame_number - votes["last_frame"] <= max_age
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
    enable_capture_orientation_auto(capture)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    analysis_interval = max(int(round(fps * settings.sample_every_seconds)), 1)
    dense_interval = max(analysis_interval // max(settings.adaptive_sample_divisor, 1), 1)
    dense_until_frame = -1
    preview_interval = preview_interval_for_fps(fps)
    cooldown_frames = max(int(fps * settings.violation_cooldown_seconds), analysis_interval)
    collection_frames = max(int(fps * settings.plate_collection_seconds), analysis_interval)
    max_lost_frames = max(int(fps * settings.tracker_max_lost_seconds), analysis_interval)
    dedupe_frames = max(int(fps * settings.rider_dedupe_seconds), max_lost_frames)
    frame_number = 0
    sampled_count = 0
    violation_count = 0
    rider_tracks = RiderTrackManager(
        cooldown_frames,
        collection_frames,
        max_lost_frames,
        dedupe_frames,
    )
    latest_analysis = empty_analysis()
    latest_preview_url = None
    last_status_update = 0.0
    last_preview_save = 0.0
    last_metadata_write = 0.0
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
            # Adaptive sampling: while a no-helmet detection is recent, sample
            # several times per interval so short-lived riders and passengers
            # accumulate enough helmet votes before leaving the frame.
            in_dense_window = settings.adaptive_sampling and frame_number <= dense_until_frame
            should_analyze = frame_number % analysis_interval == 0 or (
                in_dense_window and frame_number % dense_interval == 0
            )
            should_preview = frame_number % preview_interval == 0
            has_viewers = frame_hub.has_viewers(job_id)

            # Only decode pixels for frames that are actually used; grab() advances
            # the stream without the color-conversion cost of read().
            if should_analyze or (should_preview and has_viewers):
                ok, frame = capture.read()
            else:
                ok = capture.grab()
                frame = None
            if not ok:
                break

            if should_analyze:
                sampled_count += 1
                analysis = analyze_frame(frame, models)
                rider_tracks.update(analysis, frame_number, frame)
                latest_analysis = analysis
                if analysis["no_helmets"]:
                    dense_until_frame = frame_number + int(fps * settings.adaptive_hold_seconds)
                analysis_annotated = annotate_analysis(
                    frame, frame_number, analysis, fresh_analysis=True
                )
                detection_records.append(
                    serialize_detection_frame(frame_number, fps, frame, analysis)
                )

                now = monotonic()
                if now - last_metadata_write >= settings.metadata_write_seconds:
                    write_detection_metadata(job_id, detection_records)
                    last_metadata_write = now

                violations_to_save = rider_tracks.violations_to_save(
                    analysis["associations"], frame_number, frame, analysis_annotated
                )
                if violations_to_save:
                    for payload in violations_to_save:
                        save_violation(job_id, payload)
                    violation_count += len(violations_to_save)
            else:
                analysis = latest_analysis
                analysis_annotated = None

            if frame is not None and (should_analyze or (should_preview and has_viewers)):
                annotated = (
                    analysis_annotated
                    if analysis_annotated is not None
                    else annotate_analysis(frame, frame_number, analysis, fresh_analysis=False)
                )
                if has_viewers:
                    publish_stream_frame(job_id, annotated)

                now = monotonic()
                if should_analyze and now - last_preview_save >= 1:
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

            # Real-time pacing only matters for someone watching the MJPEG stream;
            # otherwise process as fast as the hardware allows.
            if settings.realtime_preview or has_viewers:
                pace_preview(frame_number, fps, playback_started)
            frame_number += 1

        pending_violations = rider_tracks.pending_violations_to_save()
        for payload in pending_violations:
            save_violation(job_id, payload)
        violation_count += len(pending_violations)
        write_detection_metadata(job_id, detection_records)

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
        frame,
        classes=[PERSON_CLASS_ID, CAR_CLASS_ID, MOTORCYCLE_CLASS_ID, BUS_CLASS_ID, TRUCK_CLASS_ID],
        conf=settings.object_confidence,
        imgsz=settings.object_imgsz,
        verbose=False,
        **predict_kwargs(),
    )[0]
    object_boxes = extract_boxes(object_result)
    motorcycles = [box for box in object_boxes if box["class_id"] == MOTORCYCLE_CLASS_ID]
    people = [box for box in object_boxes if box["class_id"] == PERSON_CLASS_ID]
    negative_vehicles = [box for box in object_boxes if box["class_id"] in NEGATIVE_VEHICLE_CLASS_IDS]

    helmet_boxes = detect_helmet_boxes(frame, helmet_model, motorcycles, people)
    no_helmet_boxes = [
        box for box in helmet_boxes if normalize_label(box["label"]) == NO_HELMET_LABEL
    ]
    with_helmet_boxes = [
        box for box in helmet_boxes if normalize_label(box["label"]) == WITH_HELMET_LABEL
    ]

    plate_result = plate_model.predict(
        frame,
        conf=settings.plate_confidence,
        imgsz=settings.plate_imgsz,
        verbose=False,
        **predict_kwargs(),
    )[0]
    plate_boxes = extract_boxes(plate_result)
    associations = associate_riders(
        people,
        motorcycles,
        with_helmet_boxes,
        no_helmet_boxes,
        plate_boxes,
        negative_vehicles,
    )
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
        "negative_vehicles": negative_vehicles,
        "people": people,
        "helmets": with_helmet_boxes,
        "no_helmets": no_helmet_boxes,
        "plates": plate_boxes,
        "associations": associations,
        "helmet_box": no_helmet,
        "plate_box": plate,
        "has_no_helmet": bool(no_helmet_associations),
    }


def detect_helmet_boxes(frame, helmet_model, motorcycles: list[dict], people: list[dict]) -> list[dict]:
    """Run the helmet model on rider-focused crops instead of the whole frame.

    Distant rider heads are only a few pixels at full-frame imgsz; cropping around
    each motorcycle gives the helmet model several times more effective resolution
    exactly where violations happen, and skips inference entirely when no
    motorcycles are present.
    """
    if not settings.helmet_crop_inference:
        result = helmet_model.predict(
            frame,
            conf=settings.helmet_confidence,
            imgsz=settings.helmet_imgsz,
            verbose=False,
            **predict_kwargs(),
        )[0]
        return extract_boxes(result)

    height, width = frame.shape[:2]
    regions = rider_focus_regions(motorcycles, people, width, height)
    if not regions:
        return []

    crops = []
    crop_regions = []
    for region in regions:
        crop = crop_box(frame, region)
        if crop is not None and crop.size:
            crops.append(crop)
            crop_regions.append(region)
    if not crops:
        return []

    results = helmet_model.predict(
        crops,
        conf=settings.helmet_confidence,
        imgsz=settings.helmet_crop_imgsz,
        verbose=False,
        **predict_kwargs(),
    )
    boxes = []
    for region, result in zip(crop_regions, results):
        for box in extract_boxes(result):
            box["xyxy"] = offset_xyxy(box["xyxy"], region[0], region[1], width, height)
            boxes.append(box)
    return dedupe_boxes(boxes)


def rider_focus_regions(
    motorcycles: list[dict], people: list[dict], width: int, height: int
) -> list[list[int]]:
    regions = []
    for motorcycle in motorcycles:
        x1, y1, x2, y2 = motorcycle["xyxy"]
        moto_width = max(x2 - x1, 1)
        moto_height = max(y2 - y1, 1)
        # The rider's head sits well above the motorcycle box top.
        region = [
            x1 - int(moto_width * 0.25),
            y1 - int(moto_height * 1.10),
            x2 + int(moto_width * 0.25),
            y2 + int(moto_height * 0.10),
        ]
        for person in people:
            if boxes_intersect(person["xyxy"], region):
                px1, py1, px2, py2 = person["xyxy"]
                region = [
                    min(region[0], px1),
                    min(region[1], py1),
                    max(region[2], px2),
                    max(region[3], py2),
                ]
        regions.append(clamp_box(region, width, height))
    return merge_intersecting_regions(regions)


def merge_intersecting_regions(regions: list[list[int]]) -> list[list[int]]:
    regions = [list(region) for region in regions]
    changed = True
    while changed:
        changed = False
        merged: list[list[int]] = []
        for region in regions:
            placed = False
            for existing in merged:
                if boxes_intersect(existing, region):
                    existing[0] = min(existing[0], region[0])
                    existing[1] = min(existing[1], region[1])
                    existing[2] = max(existing[2], region[2])
                    existing[3] = max(existing[3], region[3])
                    placed = True
                    changed = True
                    break
            if not placed:
                merged.append(region)
        regions = merged
    return regions


def boxes_intersect(a: list[int], b: list[int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def offset_xyxy(xyxy: list[int], dx: int, dy: int, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = xyxy
    return clamp_box([x1 + dx, y1 + dy, x2 + dx, y2 + dy], width, height)


def dedupe_boxes(boxes: list[dict], iou_threshold: float = 0.60) -> list[dict]:
    """Drop duplicate detections produced by overlapping crops, keeping the
    higher-confidence box regardless of label so a crop pair cannot report the
    same head as both helmet and no-helmet."""
    kept: list[dict] = []
    for box in sorted(boxes, key=lambda item: item["confidence"], reverse=True):
        if all(box_iou(box["xyxy"], existing["xyxy"]) < iou_threshold for existing in kept):
            kept.append(box)
    return kept


def empty_analysis() -> dict:
    return {
        "objects": [],
        "motorcycles": [],
        "negative_vehicles": [],
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
    if source_fps <= 0:
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


def enable_capture_orientation_auto(capture) -> None:
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        capture.set(orientation_auto, 1)


def save_violation(job_id: str, payload: dict) -> None:
    frame_number = payload["frame_number"]
    frame = payload["frame"]
    annotated = payload["annotated"]
    association = payload["association"]
    plate_candidate = payload.get("plate_candidate")
    violation_id = uuid4().hex
    evidence_path = settings.evidence_dir / f"{job_id}_{frame_number}_{violation_id[:8]}.jpg"
    cv2.imwrite(str(evidence_path), highlight_violation_rider(annotated, association))

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


VIOLATION_HIGHLIGHT_COLOR = (250, 80, 240)


def highlight_violation_rider(annotated, association: dict):
    """Copy of the annotated frame with this record's rider called out.

    Several violations can share one frame; without a per-record highlight
    their evidence images are identical and reviewers cannot tell which rider
    a record refers to, especially when plates are unreadable.
    """
    highlighted = annotated.copy()
    xyxys = [
        association[key]["xyxy"]
        for key in ("person_box", "motorcycle_box", "helmet_box")
        if association.get(key)
    ]
    if not xyxys:
        return highlighted

    height, width = highlighted.shape[:2]
    pad = 8
    x1 = max(int(min(box[0] for box in xyxys)) - pad, 0)
    y1 = max(int(min(box[1] for box in xyxys)) - pad, 0)
    x2 = min(int(max(box[2] for box in xyxys)) + pad, width - 1)
    y2 = min(int(max(box[3] for box in xyxys)) + pad, height - 1)
    cv2.rectangle(highlighted, (x1, y1), (x2, y2), VIOLATION_HIGHLIGHT_COLOR, 4)

    track_id = association.get("track_id")
    label = f"VIOLATION track {track_id}" if track_id is not None else "VIOLATION"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    tag_top = max(y1 - text_height - baseline - 10, 0)
    cv2.rectangle(highlighted, (x1, tag_top), (x1 + text_width + 14, tag_top + text_height + baseline + 10), VIOLATION_HIGHLIGHT_COLOR, -1)
    cv2.putText(
        highlighted,
        label,
        (x1 + 7, tag_top + text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return highlighted


def build_plate_candidate(frame, plate_box: dict, *, run_ocr: bool = True) -> dict | None:
    crop = crop_box(frame, plate_box["xyxy"], padding=8)
    if crop is None or not crop.size:
        return None

    plate_text, plate_confidence = read_plate_text(crop) if run_ocr else (None, None)
    return {
        "crop": crop.copy(),
        "plate_box": plate_box,
        "plate_text": plate_text,
        "plate_confidence": plate_confidence,
        "score": plate_candidate_score(crop, plate_box, plate_text, plate_confidence),
    }


def finalize_plate_candidates(candidates: list[dict]) -> tuple[dict | None, list[tuple[str, float]]]:
    if not candidates:
        return None, []

    finalized = []
    ocr_reads = []
    ocr_limit = max(settings.plate_ocr_candidate_limit, 1)
    for candidate in candidates[:ocr_limit]:
        item = dict(candidate)
        plate_text, plate_confidence = read_plate_text(item["crop"])
        item["plate_text"] = plate_text
        item["plate_confidence"] = plate_confidence
        item["score"] = plate_candidate_score(
            item["crop"], item["plate_box"], plate_text, plate_confidence
        )
        finalized.append(item)
        if plate_text:
            ocr_reads.append((plate_text, plate_confidence or 0.0))

    finalized.sort(key=lambda item: item["score"], reverse=True)
    return finalized[0], ocr_reads


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


def appearance_feature(frame, xyxy: list[int]):
    """HSV color histogram of a crop, L1-normalized — a cheap appearance
    descriptor the tracker blends with motion to keep rider identity when
    sampled boxes barely overlap."""
    crop = crop_box(frame, xyxy)
    if crop is None or not crop.size:
        return None
    small = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten().astype("float64")
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


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
    negative_vehicles: list[dict],
) -> list[dict]:
    plate_assignments = assign_plates_to_motorcycles(plate_boxes, motorcycles, negative_vehicles)
    for motorcycle in motorcycles:
        plate, plate_score = plate_assignments.get(id(motorcycle), (None, 0.0))
        motorcycle["plate_box"] = plate
        motorcycle["plate_score"] = plate_score
    helmet_detections = [
        {"box": box, "status": "with_helmet"} for box in with_helmet_boxes
    ] + [{"box": box, "status": "no_helmet"} for box in no_helmet_boxes]
    helmet_assignments = assign_helmets_to_people(people, helmet_detections)
    associations = []
    assigned_helmet_ids: set[int] = set()
    assigned_person_ids: set[int] = set()

    for person in people:
        helmet_detection, helmet_score = helmet_assignments.get(id(person), (None, 0.0))
        if not helmet_detection or helmet_score < settings.min_helmet_person_score:
            continue

        helmet_box = helmet_detection["box"]
        motorcycle, motorcycle_score = best_motorcycle_for_person(person, motorcycles)
        if motorcycle_score < settings.min_person_motorcycle_score:
            motorcycle = None
            motorcycle_score = 0.0

        plate, plate_score = plate_assignments.get(id(motorcycle), (None, 0.0)) if motorcycle else (None, 0.0)
        association_score = combined_score(
            helmet_box["confidence"],
            helmet_score,
            motorcycle_score,
            plate_score,
        )
        if helmet_detection["status"] == "no_helmet" and (
            not motorcycle or association_score < settings.min_no_helmet_association_score
        ):
            continue
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
        assigned_person_ids.add(id(person))

    for helmet_box in no_helmet_boxes:
        if id(helmet_box) in assigned_helmet_ids:
            continue

        available_people = [person for person in people if id(person) not in assigned_person_ids]
        person, helmet_score = best_person_for_helmet(helmet_box, available_people)
        if person and helmet_score >= settings.min_helmet_person_score:
            motorcycle, motorcycle_score = best_motorcycle_for_person(person, motorcycles)
            assigned_person_ids.add(id(person))
        else:
            person = None
            motorcycle, motorcycle_score = best_motorcycle_for_helmet(helmet_box, motorcycles)
        if person and motorcycle_score < settings.min_person_motorcycle_score:
            motorcycle = None
            motorcycle_score = 0.0
        if not person and motorcycle_score < settings.min_helmet_motorcycle_score:
            motorcycle = None
            motorcycle_score = 0.0

        if not motorcycle:
            continue

        plate, plate_score = plate_assignments.get(id(motorcycle), (None, 0.0))

        association_score = combined_score(
            helmet_box["confidence"],
            helmet_score,
            motorcycle_score,
            plate_score,
        )
        if association_score < settings.min_no_helmet_association_score:
            continue
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


def assign_plates_to_motorcycles(
    plate_boxes: list[dict],
    motorcycles: list[dict],
    negative_vehicles: list[dict],
) -> dict[int, tuple[dict, float]]:
    """One-to-one plate-to-motorcycle assignment, keyed by id(motorcycle).

    Each plate picks its single best motorcycle; ambiguous plates (runner-up
    motorcycle within the assignment margin) and plates that fit a nearby
    car/bus/truck better are dropped. Greedy assignment guarantees two riders
    can never claim the same plate in one frame.
    """
    pairs = []
    for plate_index, plate in enumerate(plate_boxes):
        scored = []
        for motorcycle_index, motorcycle in enumerate(motorcycles):
            if not plausible_plate_for_motorcycle(plate, motorcycle):
                continue
            score = score_plate_to_motorcycle(plate, motorcycle)
            if score < settings.min_plate_motorcycle_score:
                continue
            scored.append((score, motorcycle_index))
        if not scored:
            continue
        scored.sort(reverse=True)
        best_score, best_motorcycle_index = scored[0]
        if len(scored) > 1 and best_score - scored[1][0] < settings.plate_assignment_margin:
            continue
        if plate_prefers_negative_vehicle(plate, best_score, negative_vehicles):
            continue
        pairs.append((best_score, plate_index, best_motorcycle_index))

    pairs.sort(reverse=True)
    assigned_plates: set[int] = set()
    assigned_motorcycles: set[int] = set()
    assignments: dict[int, tuple[dict, float]] = {}
    for score, plate_index, motorcycle_index in pairs:
        if plate_index in assigned_plates or motorcycle_index in assigned_motorcycles:
            continue
        assigned_plates.add(plate_index)
        assigned_motorcycles.add(motorcycle_index)
        assignments[id(motorcycles[motorcycle_index])] = (plate_boxes[plate_index], score)
    return assignments


def assign_helmets_to_people(
    people: list[dict], helmet_detections: list[dict]
) -> dict[int, tuple[dict, float]]:
    """One-to-one helmet-to-person assignment, keyed by id(person).

    On two-up motorcycles the driver's and passenger's heads are close together;
    if each person independently picked their best helmet box, both could claim
    the driver's helmet and the passenger's own no-helmet box would be orphaned
    or mislabeled. Greedy one-to-one assignment gives every rider - driver and
    passenger - their own helmet observation."""
    pairs = []
    for person_index, person in enumerate(people):
        for detection_index, detection in enumerate(helmet_detections):
            score = score_helmet_to_person(detection["box"], person)
            if score >= settings.min_helmet_person_score:
                pairs.append((score, person_index, detection_index))

    pairs.sort(reverse=True)
    assigned_people: set[int] = set()
    assigned_detections: set[int] = set()
    assignments: dict[int, tuple[dict, float]] = {}
    for score, person_index, detection_index in pairs:
        if person_index in assigned_people or detection_index in assigned_detections:
            continue
        assigned_people.add(person_index)
        assigned_detections.add(detection_index)
        assignments[id(people[person_index])] = (helmet_detections[detection_index], score)
    return assignments


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


def plausible_plate_for_motorcycle(plate: dict, motorcycle: dict) -> bool:
    px, py = box_center(plate["xyxy"])
    mx1, my1, mx2, my2 = motorcycle["xyxy"]
    motorcycle_width = max(mx2 - mx1, 1)
    motorcycle_height = max(my2 - my1, 1)
    plate_width = max(plate["xyxy"][2] - plate["xyxy"][0], 1)
    plate_height = max(plate["xyxy"][3] - plate["xyxy"][1], 1)
    plate_area_ratio = box_area(plate["xyxy"]) / max(box_area(motorcycle["xyxy"]), 1)
    plate_aspect = plate_width / max(plate_height, 1)
    expanded_motorcycle = expand_box(motorcycle["xyxy"], 0.28)

    if not point_in_box((px, py), expanded_motorcycle):
        return False
    if py < my1 + motorcycle_height * 0.20:
        return False
    if not 0.0015 <= plate_area_ratio <= 0.20:
        return False
    # Thai motorcycle plates are near-square; Thai car plates are much wider,
    # so the aspect gate rejects most car plates before scoring.
    if not settings.plate_min_aspect <= plate_aspect <= settings.plate_max_aspect:
        return False

    horizontal_slop = motorcycle_width * settings.plate_horizontal_slop
    return mx1 - horizontal_slop <= px <= mx2 + horizontal_slop


def plate_prefers_negative_vehicle(
    plate: dict,
    motorcycle_score: float,
    negative_vehicles: list[dict],
) -> bool:
    for vehicle in negative_vehicles:
        if not plausible_plate_for_vehicle(plate, vehicle):
            continue
        vehicle_score = score_plate_to_motorcycle(plate, vehicle)
        if vehicle_score >= motorcycle_score + 0.03:
            return True
    return False


def plausible_plate_for_vehicle(plate: dict, vehicle: dict) -> bool:
    px, py = box_center(plate["xyxy"])
    x1, y1, x2, y2 = vehicle["xyxy"]
    height = max(y2 - y1, 1)
    expanded_vehicle = expand_box(vehicle["xyxy"], 0.12)
    if not point_in_box((px, py), expanded_vehicle):
        return False
    return py >= y1 + height * 0.28


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


def association_box_xyxy(association: dict, key: str) -> list[int] | None:
    box = association.get(key)
    if not box:
        return None
    return copy_xyxy(box["xyxy"])


def copy_xyxy(xyxy: list[int]) -> list[int]:
    return [int(value) for value in xyxy]


def association_signature_score(signature: dict, association: dict) -> float:
    candidates = [
        (
            signature.get("reference_box"),
            association_reference_box(association),
            1.0,
        ),
        (
            signature.get("motorcycle_box"),
            association_box_xyxy(association, "motorcycle_box"),
            1.08,
        ),
        (
            signature.get("person_box"),
            association_box_xyxy(association, "person_box"),
            0.92,
        ),
        (
            signature.get("helmet_box"),
            association_box_xyxy(association, "helmet_box"),
            0.82,
        ),
        (
            signature.get("plate_box"),
            association_box_xyxy(association, "plate_box"),
            1.12,
        ),
    ]
    scores = [
        min(track_match_score(previous, current) * weight, 1.0)
        for previous, current, weight in candidates
        if previous and current
    ]
    return max(scores, default=0.0)


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


def valid_no_helmet_association(association: dict) -> bool:
    if association.get("helmet_status") != "no_helmet":
        return False
    if not association.get("helmet_box") or not association.get("motorcycle_box"):
        return False
    return float(association.get("association_score", 0.0)) >= settings.min_no_helmet_association_score


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


# OCR lives in app.services.plate_ocr; read_plate_text is re-exported above
# for callers that import it from this module (scripts/backfill_plate_ocr.py).
