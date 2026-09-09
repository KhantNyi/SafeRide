import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.pipeline import serialize_detection_frame


class PlaybackMetadataTests(unittest.TestCase):
    def test_playback_identity_is_independent_of_violation_status(self):
        bike = {"label": "motorcycle", "confidence": .9, "xyxy": [0, 80, 100, 200], "track_id": 7}
        person = {"label": "person", "confidence": .9, "xyxy": [0, 0, 100, 180]}
        for status in ("with_helmet", "no_helmet", "unknown"):
            with self.subTest(status=status):
                association = {"person_box": person, "motorcycle_box": bike,
                               "track_id": 7, "helmet_status": status}
                analysis = {"people": [person], "motorcycles": [bike],
                            "associations": [] if status == "unknown" else [association]}
                result = serialize_detection_frame(30, 30, np.zeros((240, 320, 3)), analysis)
                self.assertEqual(result["motorcycles"][0]["track_id"], 7)
                self.assertEqual(result["tracking_associations"][0]["track_id"], 7)
                self.assertEqual(result["tracking_associations"][0]["helmet_status"], status)
                self.assertEqual(len(result["associations"]), int(status == "no_helmet"))
                self.assertEqual(len(analysis["associations"]), int(status != "unknown"))

    def test_unknown_passenger_is_preserved_for_ambiguity_check(self):
        bike = {"label": "motorcycle", "confidence": .9, "xyxy": [0, 80, 100, 200], "track_id": 7}
        people = [{"label": "person", "confidence": .9, "xyxy": [x, 0, x + 60, 180]} for x in (0, 30)]
        result = serialize_detection_frame(0, 30, np.zeros((240, 320, 3)), {
            "people": people, "motorcycles": [bike], "associations": [{
                "person_box": people[0], "motorcycle_box": bike, "track_id": 7,
                "helmet_status": "with_helmet"}]})
        self.assertEqual([a["track_id"] for a in result["tracking_associations"]], [7, 7])


if __name__ == "__main__":
    unittest.main()
