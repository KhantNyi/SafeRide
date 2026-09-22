from dataclasses import dataclass, field

import numpy as np

from app.services.assignment import match_scores


@dataclass
class ByteTrackDetection:
    xyxy: list[int]
    score: float
    metadata: dict = field(default_factory=dict)
    feature: np.ndarray | None = None


@dataclass
class TrackedDetection:
    track_id: int
    xyxy: list[int]
    score: float
    metadata: dict
    state: str
    hits: int


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over (cx, cy, w, h).

    State is [cx, cy, w, h, vcx, vcy, vw, vh]. Sampled video means irregular
    frame gaps between updates, so the transition matrix is rebuilt per step
    with the actual dt in frames. Noise scales with box height, following the
    original ByteTrack weighting, so large near-camera boxes tolerate more
    absolute movement than small distant ones.
    """

    STD_POSITION = 1 / 20
    STD_VELOCITY = 1 / 160

    def __init__(self, xyxy: list[int] | list[float]):
        cx, cy, w, h = xyxy_to_cxcywh(xyxy)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        std = [
            2 * self.STD_POSITION * h,
            2 * self.STD_POSITION * h,
            2 * self.STD_POSITION * h,
            2 * self.STD_POSITION * h,
            10 * self.STD_VELOCITY * h,
            10 * self.STD_VELOCITY * h,
            10 * self.STD_VELOCITY * h,
            10 * self.STD_VELOCITY * h,
        ]
        self.P = np.diag(np.square(std))

    def predict_state(self, dt: float) -> np.ndarray:
        transition = np.eye(8)
        for i in range(4):
            transition[i, i + 4] = dt
        h = max(self.x[3], 1.0)
        process_std = [
            self.STD_POSITION * h,
            self.STD_POSITION * h,
            self.STD_POSITION * h,
            self.STD_POSITION * h,
            self.STD_VELOCITY * h,
            self.STD_VELOCITY * h,
            self.STD_VELOCITY * h,
            self.STD_VELOCITY * h,
        ]
        process_noise = np.diag(np.square(process_std)) * max(dt, 1.0)
        predicted_x = transition @ self.x
        predicted_p = transition @ self.P @ transition.T + process_noise
        return predicted_x, predicted_p, transition

    def predict_box(self, dt: float) -> list[int]:
        predicted_x, _p, _f = self.predict_state(dt)
        return cxcywh_to_xyxy(predicted_x[:4])

    def update(self, xyxy: list[int] | list[float], dt: float) -> None:
        predicted_x, predicted_p, _f = self.predict_state(dt)
        measurement = np.array(xyxy_to_cxcywh(xyxy), dtype=np.float64)
        observation = np.zeros((4, 8))
        observation[:4, :4] = np.eye(4)
        h = max(predicted_x[3], 1.0)
        measurement_std = [
            self.STD_POSITION * h,
            self.STD_POSITION * h,
            self.STD_POSITION * h,
            self.STD_POSITION * h,
        ]
        measurement_noise = np.diag(np.square(measurement_std))

        innovation = measurement - observation @ predicted_x
        innovation_cov = observation @ predicted_p @ observation.T + measurement_noise
        gain = predicted_p @ observation.T @ np.linalg.inv(innovation_cov)
        self.x = predicted_x + gain @ innovation
        self.P = (np.eye(8) - gain @ observation) @ predicted_p


class ByteTracker:
    """ByteTrack-style tracker with Kalman motion and appearance cues.

    Keeps ByteTrack's two-stage behavior: high-confidence detections are
    matched first, then still-unmatched tracks get a second chance with
    lower-confidence detections. Motion is predicted with a per-track Kalman
    filter instead of a raw velocity EMA, and when detections carry an
    appearance feature (a color histogram of the crop) the match score blends
    motion with appearance similarity — so a rider keeps their identity even
    when sampled boxes barely overlap.
    """

    def __init__(
        self,
        *,
        high_threshold: float,
        low_threshold: float,
        new_track_threshold: float,
        match_threshold: float,
        max_time_lost: int,
        appearance_weight: float = 0.30,
        source_fps: float = 30.0,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.new_track_threshold = new_track_threshold
        self.match_threshold = match_threshold
        self.max_time_lost = max(max_time_lost, 1)
        self.appearance_weight = min(max(appearance_weight, 0.0), 0.9)
        self.source_fps = max(source_fps, 1.0)
        self.next_track_id = 1
        self.tracks: list[dict] = []
        self.uncertain_detection_indexes: set[int] = set()

    def update(self, detections: list[ByteTrackDetection], frame_number: int) -> list[TrackedDetection]:
        self.uncertain_detection_indexes = set()
        # Expired tracks must not adopt a new arrival before the age check.
        for track in self.tracks:
            lifetime = min(self.max_time_lost, self.source_fps * .75) if track["hits"] == 1 else self.max_time_lost
            if frame_number - track["last_frame"] > lifetime:
                track["state"] = "removed"
        candidates = [
            detection
            for detection in detections
            if detection.score >= self.low_threshold and valid_box(detection.xyxy)
        ]
        candidates.sort(key=lambda detection: detection.score, reverse=True)
        high_detections = [
            detection for detection in candidates if detection.score >= self.high_threshold
        ]
        low_detections = [
            detection for detection in candidates if detection.score < self.high_threshold
        ]

        live_tracks = [track for track in self.tracks if track["state"] != "removed"]
        tracked: list[TrackedDetection] = []

        high_matches, unmatched_tracks, unmatched_high = self.match_tracks(
            live_tracks,
            high_detections,
            frame_number,
            self.match_threshold,
        )
        for track, detection in high_matches:
            self.update_track(track, detection, frame_number)
            tracked.append(self.tracked_detection(track, detection))

        low_matches, unmatched_tracks, _unmatched_low = self.match_tracks(
            unmatched_tracks,
            low_detections,
            frame_number,
            max(self.match_threshold * 0.80, 0.10),
        )
        for track, detection in low_matches:
            self.update_track(track, detection, frame_number)
            tracked.append(self.tracked_detection(track, detection))

        for track in unmatched_tracks:
            if frame_number - track["last_frame"] > self.max_time_lost:
                track["state"] = "removed"
            else:
                track["state"] = "lost"

        for detection in unmatched_high:
            if detection.metadata.get("index") in self.uncertain_detection_indexes:
                continue  # Coast through ambiguous overlap, do not create a duplicate.
            if detection.score < self.new_track_threshold:
                continue
            track = self.create_track(detection, frame_number)
            tracked.append(self.tracked_detection(track, detection))

        self.prune()
        tracked.sort(key=lambda item: item.metadata.get("index", 0))
        return tracked

    def match_tracks(
        self,
        tracks: list[dict],
        detections: list[ByteTrackDetection],
        frame_number: int,
        threshold: float,
    ) -> tuple[list[tuple[dict, ByteTrackDetection]], list[dict], list[ByteTrackDetection]]:
        scores = np.zeros((len(tracks), len(detections)))
        for track_index, track in enumerate(tracks):
            dt = max(frame_number - track["last_frame"], 0)
            # Linear extrapolation can shrink a receding, occluded motorcycle
            # to zero size and overshoot its position. Dampen only lost-track
            # extrapolation; retain uncertainty rather than inventing a new ID.
            horizon = max(self.max_time_lost / 2, 1)
            prediction_dt = horizon * (1 - np.exp(-dt / horizon)) if track["state"] == "lost" else dt
            predicted_box = track["kalman"].predict_box(prediction_dt)
            center = np.array(xyxy_to_cxcywh(predicted_box), dtype=float)
            previous_size = np.maximum(xyxy_to_cxcywh(track["xyxy"])[2:],1)
            center[2:] = np.clip(center[2:], previous_size*.35, previous_size*2.8)
            predicted_box = cxcywh_to_xyxy(center)
            for detection_index, detection in enumerate(detections):
                motion = motion_match_score(predicted_box, detection.xyxy)
                # Before a second observation there is no learned velocity.
                # Fast bikes can travel beyond one box-height between samples.
                # Permit a wider first-step gate ONLY with strong appearance
                # evidence and a short gap, not for an old stationary ghost.
                if (track["hits"] == 1 and dt <= self.source_fps * .6
                        and track.get("feature") is not None and detection.feature is not None
                        and feature_similarity(track["feature"], detection.feature) >= .65):
                    previous = track["xyxy"]
                    distance = point_distance(box_center(previous),box_center(detection.xyxy))
                    extent = max(previous[2]-previous[0],previous[3]-previous[1],1)
                    motion = max(motion, max(0,1-distance/(2.5*extent))*.65)
                predicted_size = np.maximum(xyxy_to_cxcywh(predicted_box)[2:], 1)
                detected_size = np.maximum(xyxy_to_cxcywh(detection.xyxy)[2:], 1)
                ratios = detected_size / predicted_size
                if np.any(ratios < .35) or np.any(ratios > 2.8):
                    continue
                # A plausible position alone cannot justify a sudden scale jump.
                motion *= float(np.exp(-.5 * np.abs(np.log(ratios)).mean()))
                score = self.blend_appearance(motion, track, detection)
                if score >= threshold:
                    scores[track_index, detection_index] = score
        for index, detection in enumerate(detections):
            contenders = sorted([(scores[i, index], tracks[i]) for i in range(len(tracks))
                                 if scores[i, index] >= threshold], key=lambda x: x[0], reverse=True)
            if len(contenders) < 2:
                continue
            best_score, best = contenders[0]
            rival = next(((score, track) for score, track in contenders[1:]
                          if track.get("identity", track["id"]) != best.get("identity", best["id"])), None)
            if rival and best_score - rival[0] < .08 and detection.metadata.get("occluded"):
                scores[:, index] = 0
                self.uncertain_detection_indexes.add(detection.metadata.get("index", index))
        matched_track_indexes: set[int] = set()
        matched_detection_indexes: set[int] = set()
        matches = []

        for track_index, detection_index in match_scores(scores, threshold):
            matched_track_indexes.add(track_index)
            matched_detection_indexes.add(detection_index)
            matches.append((tracks[track_index], detections[detection_index]))

        unmatched_tracks = [
            track for index, track in enumerate(tracks) if index not in matched_track_indexes
        ]
        unmatched_detections = [
            detection
            for index, detection in enumerate(detections)
            if index not in matched_detection_indexes
        ]
        return matches, unmatched_tracks, unmatched_detections

    def blend_appearance(self, motion: float, track: dict, detection: ByteTrackDetection) -> float:
        track_feature = track.get("feature")
        if (
            self.appearance_weight <= 0
            or track_feature is None
            or detection.feature is None
        ):
            return motion
        # Appearance can rescue a weak motion match (fast riders between
        # samples) but never a hopeless one — a tiny motion floor prevents
        # identity jumps across the whole frame between similar-looking bikes.
        if motion < 0.02:
            return motion
        appearance = feature_similarity(track_feature, detection.feature)
        return (1.0 - self.appearance_weight) * motion + self.appearance_weight * appearance

    def create_track(self, detection: ByteTrackDetection, frame_number: int) -> dict:
        track = {
            "id": self.next_track_id,
            "xyxy": [float(value) for value in detection.xyxy],
            "kalman": KalmanBoxFilter(detection.xyxy),
            "feature": None if detection.feature is None or detection.metadata.get("occluded") else detection.feature.copy(),
            "score": detection.score,
            "first_frame": frame_number,
            "last_frame": frame_number,
            "hits": 1,
            "state": "tracked",
        }
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def update_track(self, track: dict, detection: ByteTrackDetection, frame_number: int) -> None:
        dt = max(frame_number - track["last_frame"], 1)
        track["kalman"].update(detection.xyxy, dt)
        track["xyxy"] = [float(value) for value in detection.xyxy]
        track["score"] = detection.score
        track["last_frame"] = frame_number
        track["hits"] += 1
        track["state"] = "tracked"
        if detection.feature is not None and not detection.metadata.get("occluded"):
            if track["feature"] is None:
                track["feature"] = detection.feature.copy()
            else:
                blended = 0.8 * track["feature"] + 0.2 * detection.feature
                total = blended.sum()
                track["feature"] = blended / total if total > 0 else blended

    def tracked_detection(self, track: dict, detection: ByteTrackDetection) -> TrackedDetection:
        return TrackedDetection(
            track_id=track["id"],
            xyxy=[int(round(value)) for value in track["xyxy"]],
            score=detection.score,
            metadata=detection.metadata,
            state=track["state"],
            hits=track["hits"],
        )

    def active_track_ids(self) -> set[int]:
        return {track["id"] for track in self.tracks if track["state"] != "removed"}

    def prune(self) -> None:
        self.tracks = [track for track in self.tracks if track["state"] != "removed"]


def valid_box(xyxy: list[int]) -> bool:
    return len(xyxy) == 4 and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]


def motion_match_score(previous_xyxy: list[int], current_xyxy: list[int]) -> float:
    iou_score = box_iou(previous_xyxy, current_xyxy)
    previous_center = box_center(previous_xyxy)
    current_center = box_center(current_xyxy)
    previous_width = max(previous_xyxy[2] - previous_xyxy[0], 1)
    previous_height = max(previous_xyxy[3] - previous_xyxy[1], 1)
    normalized_distance = point_distance(previous_center, current_center) / max(
        previous_width, previous_height
    )
    distance_score = max(0.0, 1.0 - normalized_distance)
    return max(iou_score, distance_score * 0.65)


def feature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between L1-normalized histograms, mapped to [0, 1]."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def xyxy_to_cxcywh(xyxy: list[int] | list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0))


def cxcywh_to_xyxy(cxcywh) -> list[int]:
    cx, cy, w, h = cxcywh
    half_w = max(w, 1.0) / 2
    half_h = max(h, 1.0) / 2
    return [int(round(cx - half_w)), int(round(cy - half_h)), int(round(cx + half_w)), int(round(cy + half_h))]


def box_center(xyxy: list[int] | list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def box_area(xyxy: list[int] | list[float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def box_iou(a: list[int] | list[float], b: list[int] | list[float]) -> float:
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
