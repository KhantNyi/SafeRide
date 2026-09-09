"""Optional replay-only optical flow. Never imports or writes violation storage."""
import json
from copy import deepcopy
from pathlib import Path
from threading import Lock
from time import monotonic

import cv2
import numpy as np

from app.core.config import settings

GROUPS = ("people", "motorcycles", "helmets", "no_helmets", "plates")
FIELDS = ("person_box", "motorcycle_box", "helmet_box", "plate_box")
_lock = Lock()
_active: str | None = None
_states: dict[str, dict] = {}


def playback_path(job_id: str) -> Path:
    return settings.metadata_dir / f"{job_id}_playback.json"


def playback_status(job_id: str) -> dict:
    with _lock:
        state = _states.get(job_id)
        if state and state["status"] != "completed":
            return dict(state)
    path = playback_path(job_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "idle"}


def reserve_playback(job_id: str) -> None:
    global _active
    with _lock:
        if _active is not None:
            raise ValueError("Another playback enhancement is running. Try again when it finishes.")
        _active = job_id
        _states[job_id] = {"status": "processing", "progress": 0}


def key(box: dict) -> tuple:
    return box["label"], tuple(box["xyxy"])


def overlap(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    return intersection / max(area(a) + area(b) - intersection, 1)


def move_box(previous, current, box):
    """Sparse forward/backward flow; reject weak, inconsistent or collapsing fits."""
    height, width = previous.shape
    x1, y1, x2, y2 = np.rint(box).astype(int)
    mask = np.zeros_like(previous)
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 255
    points = cv2.goodFeaturesToTrack(previous, maxCorners=40, qualityLevel=.02, minDistance=3, mask=mask)
    if points is None or len(points) < 6:
        return None
    options = dict(winSize=(21, 21), maxLevel=3)
    forward, valid, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    if forward is None:
        return None
    backward, back_valid, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **options)
    if backward is None:
        return None
    good = (valid.ravel() == 1) & (back_valid.ravel() == 1) & (np.linalg.norm(points - backward, axis=2).ravel() < 1.5)
    if good.sum() < 6 or good.mean() < .5:
        return None
    transform, inliers = cv2.estimateAffinePartial2D(points[good], forward[good], method=cv2.RANSAC, ransacReprojThreshold=2)
    if transform is None or inliers.mean() < .65 or not np.isfinite(transform).all():
        return None
    scale = np.hypot(transform[0, 0], transform[1, 0])
    if not .8 <= scale <= 1.25:
        return None
    corners = np.array([[box[0], box[1], 1], [box[2], box[1], 1], [box[2], box[3], 1], [box[0], box[3], 1]]) @ transform.T
    result = np.array([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width - 1)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height - 1)
    return result if min(result[2:] - result[:2]) >= 3 else None


def render_record(base, positions, number, timestamp):
    record = deepcopy(base)
    record.update(frame_number=number, timestamp=timestamp, playback_generated=True)
    replacements = {}
    for group in GROUPS:
        record[group] = []
        for index, box in enumerate(base[group]):
            position = positions.get((group, index))
            if position is None:
                continue
            updated = {**box, "xyxy": [round(float(v), 2) for v in position],
                       "playback_id": f"{base['frame_number']}:{group}:{index}"}
            replacements[key(box)] = updated
            record[group].append(updated)
    for collection in ("associations", "tracking_associations"):
        record[collection] = []
        for association in base.get(collection, []):
            updated = dict(association)
            for field in FIELDS:
                box = association.get(field)
                updated[field] = replacements.get(key(box)) if box else None
            if updated.get("motorcycle_box"):
                record[collection].append(updated)
    return record


def build_playback(source_path: str, frames: list[dict], progress=lambda value: None) -> list[dict]:
    if len(frames) < 2:
        raise ValueError("At least two detection samples are needed")
    capture = cv2.VideoCapture(source_path)
    try:
        if not capture.isOpened():
            raise ValueError("Source video cannot be opened")
        # Saved dimensions determine orientation, independently of mutable settings.
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        auto_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        auto_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (auto_width, auto_height) != (frames[0]["width"], frames[0]["height"]):
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps):
            raise ValueError("Source video has no valid frame rate")
        step = max(1, round(fps / 10))
        observations = {f["frame_number"]: f for f in frames}
        last = max(observations)
        output, segment = [], []
        base, previous, positions = None, None, {}
        factor = min(1., 640 / max(frames[0]["width"], frames[0]["height"]))
        for number in range(last + 1):
            if not capture.grab():
                raise ValueError("Source ended before the last detection sample")
            if number not in observations and number % step:
                continue
            ok, image = capture.retrieve()
            if not ok:
                raise ValueError("Could not decode source video")
            if image.shape[:2] != (frames[0]["height"], frames[0]["width"]):
                raise ValueError("Source orientation does not match saved detections")
            gray = cv2.cvtColor(cv2.resize(image, None, fx=factor, fy=factor), cv2.COLOR_BGR2GRAY)
            if base is not None and previous is not None:
                positions = {identity: moved for identity, box in positions.items()
                             if (moved := move_box(previous, gray, box)) is not None}
            if number in observations:
                anchor = observations[number]
                if base is not None:
                    # Validate each path against the next observed boxes, without
                    # joining detection IDs or feeding anything back to detection.
                    accepted = set()
                    for group in GROUPS:
                        candidates = []
                        for identity, box in positions.items():
                            if identity[0] != group:
                                continue
                            for index, target in enumerate(anchor[group]):
                                score = overlap(box / factor, target["xyxy"])
                                if score >= .2:
                                    candidates.append((score, identity, index))
                        used = set()
                        for _, identity, index in sorted(candidates, reverse=True):
                            if identity not in accepted and index not in used:
                                accepted.add(identity)
                                used.add(index)
                    for frame_number, timestamp, snapshot in segment:
                        # Exact detection anchors remain visible; uncertain estimates disappear.
                        selected = snapshot if frame_number == base["frame_number"] else {i: b for i, b in snapshot.items() if i in accepted}
                        output.append(render_record(base, selected, frame_number, timestamp))
                base = anchor
                positions = {(group, index): np.array(box["xyxy"], dtype=float) * factor
                             for group in GROUPS for index, box in enumerate(base[group])}
                segment = []
            if base is not None:
                segment.append((number, number / fps, {i: b / factor for i, b in positions.items()}))
            previous = gray
            if number % max(round(fps), 1) == 0:
                progress(round(number / max(last, 1) * 100))
        if base is not None:
            output.append(render_record(base, segment[-1][2], last, last / fps))
        return output
    finally:
        capture.release()


def enhance_playback(job_id: str, source_path: str) -> None:
    global _active
    started = monotonic()
    metadata = settings.metadata_dir / f"{job_id}_detections.json"
    try:
        original = metadata.read_bytes()
        def progress(value):
            with _lock:
                _states[job_id] = {"status": "processing", "progress": value}
        frames = build_playback(source_path, json.loads(original)["frames"], progress)
        if not metadata.exists() or metadata.read_bytes() != original:
            raise ValueError("Detection data changed; enhancement was discarded")
        result = {"status": "completed", "progress": 100, "elapsed_seconds": round(monotonic() - started, 2), "frames": frames}
        temporary = playback_path(job_id).with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(playback_path(job_id))
        with _lock:
            _states.pop(job_id, None)
    except Exception as exc:
        with _lock:
            _states[job_id] = {"status": "failed", "message": str(exc)}
    finally:
        with _lock:
            _active = None
