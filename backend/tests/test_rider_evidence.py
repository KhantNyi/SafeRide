import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.pipeline import RiderTrackManager, rider_evidence_score


class RiderEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.manager = RiderTrackManager(120, 180, 90, 360)
        self.track = self.manager.violation_track(1, 0)
        self.frame = np.random.default_rng(7).integers(0, 256, (400, 600, 3), dtype=np.uint8)
        self.association = {
            "track_id": 1,
            "helmet_box": {"xyxy": [220, 80, 280, 140], "confidence": 0.85},
            "person_box": {"xyxy": [200, 70, 310, 260]},
            "motorcycle_box": {"xyxy": [180, 200, 340, 350]},
        }

    def update(self, association, number, frame=None):
        frame = self.frame if frame is None else frame
        self.manager.update_pending_violation(self.track, association, number, frame, frame)

    def test_late_edge_view_keeps_earlier_evidence_and_latest_dedup_position(self):
        self.update(self.association, 30)
        edge = deepcopy(self.association)
        edge["motorcycle_box"]["xyxy"][2] = 600
        edge["helmet_box"]["confidence"] = 0.99
        self.update(edge, 90)
        with patch.object(self.manager, "is_duplicate_save", return_value=False) as duplicate:
            payload = self.manager.finalize_pending_track(self.track)
        self.assertEqual(payload["frame_number"], 30)
        self.assertEqual(payload["association"]["motorcycle_box"], self.association["motorcycle_box"])
        duplicate.assert_called_once_with(edge, 90)
        self.assertEqual(self.manager.recent_saves[-1]["frame"], 90)
        self.assertEqual(self.manager.recent_saves[-1]["xyxy"], edge["motorcycle_box"]["xyxy"])
        self.assertIsNone(self.track["evidence_association"])

    def test_clearer_later_view_replaces_blurred_view_and_copies_pixels(self):
        blurred = cv2.GaussianBlur(self.frame, (15, 15), 5)
        self.update(self.association, 10, blurred)
        self.update(self.association, 20)
        expected = self.frame.copy()
        self.frame[:] = 0
        self.association["helmet_box"]["xyxy"][0] = 0
        payload = self.manager.violation_payload(self.track)
        self.assertEqual(payload["frame_number"], 20)
        self.assertEqual(payload["association"]["helmet_box"]["xyxy"][0], 220)
        np.testing.assert_array_equal(payload["frame"], expected)

    def test_plate_collection_continues_when_screenshot_is_not_replaced(self):
        self.update(self.association, 10)
        edge = deepcopy(self.association)
        edge["motorcycle_box"]["xyxy"][2] = 600
        edge["plate_box"] = {"xyxy": [230, 250, 290, 280], "confidence": 0.9}
        self.update(edge, 20)
        self.update(edge, 30)
        self.assertEqual(self.track["pending_frame_number"], 10)
        self.assertEqual(self.track["plate_sightings"], 2)
        with patch("app.services.pipeline.read_plate_text", return_value=("1234", 0.9)):
            payload = self.manager.violation_payload(self.track)
        self.assertEqual(payload["plate_candidate"]["plate_text"], "1234")
        self.assertEqual(payload["frame_number"], 10)

    def test_equal_quality_keeps_first_view_and_edge_only_view_is_still_saved(self):
        self.association["motorcycle_box"]["xyxy"][2] = 600
        self.update(self.association, 10)
        self.update(self.association, 20)
        self.assertEqual(self.manager.violation_payload(self.track)["frame_number"], 10)

    def test_size_and_confidence_improve_otherwise_equal_views(self):
        flat = np.zeros_like(self.frame)
        small = deepcopy(self.association)
        small["helmet_box"]["xyxy"] = [220, 80, 240, 100]
        weak = deepcopy(self.association)
        weak["helmet_box"]["confidence"] = 0.5
        baseline = rider_evidence_score(flat, self.association)
        self.assertGreater(baseline, rider_evidence_score(flat, small))
        self.assertGreater(baseline, rider_evidence_score(flat, weak))


if __name__ == "__main__":
    unittest.main()
