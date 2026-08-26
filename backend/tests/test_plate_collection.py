import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.pipeline import RiderTrackManager  # noqa: E402


def plate_box(confidence: float) -> dict:
    return {
        "class_id": 0,
        "label": "license plate",
        "confidence": confidence,
        "xyxy": [10, 10, 50, 35],
    }


class WholeTrackPlateCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = RiderTrackManager(
            cooldown_frames=120,
            collection_frames=180,
            max_lost_frames=90,
            dedupe_frames=360,
        )
        self.frame = np.zeros((80, 100, 3), dtype=np.uint8)

    @patch("app.services.pipeline.read_plate_text")
    def test_collection_is_bounded_and_defers_ocr(self, read_plate_text) -> None:
        track = self.manager.violation_track(7, 0)

        for frame_number in range(8):
            self.manager.collect_plate_candidate(
                track,
                plate_box(0.40 + frame_number / 100),
                frame_number,
                self.frame,
            )

        self.assertEqual(len(track["plate_candidates"]), 5)
        self.assertEqual(track["plate_sightings"], 8)
        read_plate_text.assert_not_called()

        track["pending_started_frame"] = 0
        track["pending_samples"] = 2
        track["pending_association"] = {
            "track_id": 7,
            "plate_box": None,
        }
        track["pending_frame_number"] = 30
        track["pending_frame"] = self.frame
        track["pending_annotated"] = self.frame
        read_plate_text.return_value = ("1กข 1234", 0.8)

        payload = self.manager.violation_payload(track)

        self.assertEqual(read_plate_text.call_count, 3)
        self.assertEqual(payload["plate_candidate"]["plate_text"], "1กข 1234")
        self.assertIsNotNone(payload["association"]["plate_box"])

    def test_preconfirmation_candidates_survive_violation_start(self) -> None:
        track = self.manager.violation_track(9, 0)
        self.manager.collect_plate_candidate(track, plate_box(0.9), 0, self.frame)
        association = {
            "track_id": 9,
            "plate_box": plate_box(0.7),
        }

        self.manager.update_pending_violation(
            track, association, 30, self.frame, self.frame
        )

        self.assertEqual(track["pending_started_frame"], 30)
        self.assertEqual(track["plate_sightings"], 2)
        self.assertEqual(len(track["plate_candidates"]), 2)

    def test_minimum_sample_count_no_longer_finalizes_early(self) -> None:
        track = self.manager.violation_track(11, 0)
        track["pending_started_frame"] = 0
        track["pending_samples"] = 20

        with patch.object(self.manager.tracker, "active_track_ids", return_value={11}):
            self.assertFalse(self.manager.pending_ready(track, 179))
            self.assertTrue(self.manager.pending_ready(track, 180))


if __name__ == "__main__":
    unittest.main()
