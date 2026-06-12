from dataclasses import dataclass, field


@dataclass
class ByteTrackDetection:
    xyxy: list[int]
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class TrackedDetection:
    track_id: int
    xyxy: list[int]
    score: float
    metadata: dict
    state: str
    hits: int


class ByteTracker:
    """Small ByteTrack-style tracker for sampled video detections.

    ByteTrack keeps identities by matching high-confidence detections first, then
    giving still-unmatched tracks a second chance with lower-confidence detections.
    This local version avoids optional native assignment dependencies while keeping
    that two-stage behavior for SafeRide's rider association boxes.
    """

    def __init__(
        self,
        *,
        high_threshold: float,
        low_threshold: float,
        new_track_threshold: float,
        match_threshold: float,
        max_time_lost: int,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.new_track_threshold = new_track_threshold
        self.match_threshold = match_threshold
        self.max_time_lost = max(max_time_lost, 1)
        self.next_track_id = 1
        self.tracks: list[dict] = []

    def update(self, detections: list[ByteTrackDetection], frame_number: int) -> list[TrackedDetection]:
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

        high_matches, unmatched_tracks, unmatched_high = match_tracks(
            live_tracks,
            high_detections,
            frame_number,
            self.match_threshold,
        )
        for track, detection in high_matches:
            self.update_track(track, detection, frame_number)
            tracked.append(self.tracked_detection(track, detection))

        low_matches, unmatched_tracks, _unmatched_low = match_tracks(
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
            if detection.score < self.new_track_threshold:
                continue
            track = self.create_track(detection, frame_number)
            tracked.append(self.tracked_detection(track, detection))

        self.prune()
        tracked.sort(key=lambda item: item.metadata.get("index", 0))
        return tracked

    def create_track(self, detection: ByteTrackDetection, frame_number: int) -> dict:
        track = {
            "id": self.next_track_id,
            "xyxy": [float(value) for value in detection.xyxy],
            "velocity": [0.0, 0.0],
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
        previous_center = box_center(track["xyxy"])
        current_center = box_center(detection.xyxy)
        frame_gap = max(frame_number - track["last_frame"], 1)
        measured_velocity = [
            (current_center[0] - previous_center[0]) / frame_gap,
            (current_center[1] - previous_center[1]) / frame_gap,
        ]
        track["velocity"] = [
            0.70 * track["velocity"][0] + 0.30 * measured_velocity[0],
            0.70 * track["velocity"][1] + 0.30 * measured_velocity[1],
        ]
        track["xyxy"] = [float(value) for value in detection.xyxy]
        track["score"] = detection.score
        track["last_frame"] = frame_number
        track["hits"] += 1
        track["state"] = "tracked"

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


def match_tracks(
    tracks: list[dict],
    detections: list[ByteTrackDetection],
    frame_number: int,
    threshold: float,
) -> tuple[list[tuple[dict, ByteTrackDetection]], list[dict], list[ByteTrackDetection]]:
    pairs = []
    for track_index, track in enumerate(tracks):
        predicted_box = predict_box(track, frame_number)
        for detection_index, detection in enumerate(detections):
            score = detection_match_score(predicted_box, detection.xyxy)
            if score >= threshold:
                pairs.append((score, track_index, detection_index))

    pairs.sort(reverse=True, key=lambda item: item[0])
    matched_track_indexes: set[int] = set()
    matched_detection_indexes: set[int] = set()
    matches = []

    for _score, track_index, detection_index in pairs:
        if track_index in matched_track_indexes or detection_index in matched_detection_indexes:
            continue
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


def predict_box(track: dict, frame_number: int) -> list[int]:
    frame_gap = max(frame_number - track["last_frame"], 0)
    dx = track["velocity"][0] * frame_gap
    dy = track["velocity"][1] * frame_gap
    x1, y1, x2, y2 = track["xyxy"]
    return [
        int(round(x1 + dx)),
        int(round(y1 + dy)),
        int(round(x2 + dx)),
        int(round(y2 + dy)),
    ]


def valid_box(xyxy: list[int]) -> bool:
    return len(xyxy) == 4 and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]


def detection_match_score(previous_xyxy: list[int], current_xyxy: list[int]) -> float:
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
