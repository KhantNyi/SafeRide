import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.plate_ocr import combine_plate_lines, vote_plate_texts, plate_reading_status
from app.services.plate_candidates import select_plate_candidates
from app.services.pipeline import finalize_plate_candidates, RiderTrackManager


class PlateOCRTests(unittest.TestCase):
    def test_merged_and_separate_numbers_vote_as_one_registration(self):
        reads = [("1กข1234", .9), ("1กข 1234", .8)]
        voted = vote_plate_texts(reads)
        self.assertEqual(voted[0], "1กข 1234")
        self.assertEqual(plate_reading_status(voted, reads), "read")

    def test_joined_line_does_not_append_duplicate_number(self):
        lines = [([[0, 0], [100, 0], [100, 20], [0, 20]], "1กข1234", .9),
                 ([[0, 30], [100, 30], [100, 50], [0, 50]], "1234", .8)]
        self.assertEqual(combine_plate_lines(lines)[0], "1กข 1234")

    def test_canonical_order_preserves_leading_zeros_and_province(self):
        reads = [("1กข0123 เชียงใหม่", .9), ("1กข 0123 เชียงใหม่", .8)]
        voted = vote_plate_texts(reads)
        self.assertEqual(voted[0], "1กข 0123 เชียงใหม่")
        self.assertEqual(plate_reading_status(voted, reads), "read")

    def test_weak_single_partial_and_conflicting_readings_are_uncertain(self):
        cases = [[("1กข 1234", .12)], [("1กข 1234", .95)],
                 [("1234", .9), ("1234", .9)],
                 [("1กข 1234", .9), ("1กข 5678", .9)],
                 [("1กข 1235", .8), ("1กข 1284", .8), ("1กข 9234", .8)]]
        for reads in cases:
            with self.subTest(reads=reads):
                voted = vote_plate_texts(reads)
                self.assertIsNotNone(voted)  # Keep suggestions available for review.
                self.assertEqual(plate_reading_status(voted, reads), "uncertain")
        self.assertEqual(plate_reading_status(None, []), "unreadable")

    def test_supported_majority_can_survive_one_bad_read(self):
        reads = [("1กข 1234", .8), ("1กข 1234", .8), ("1กข 1284", .8)]
        self.assertEqual(plate_reading_status(vote_plate_texts(reads), reads), "read")


class PlateDiversityTests(unittest.TestCase):
    def candidate(self, timestamp, score):
        return {"timestamp": timestamp, "score": score, "descriptor": np.zeros((16, 16)),
                "crop": np.zeros((30, 50, 3), dtype=np.uint8),
                "plate_box": {"confidence": .8, "xyxy": [0, 0, 50, 30]}}

    def test_prefers_different_moments_without_losing_best_crop(self):
        candidates = [self.candidate(0, .8), self.candidate(.02, .79),
                      self.candidate(.04, .78), self.candidate(.6, .77), self.candidate(1.2, .76)]
        result = select_plate_candidates(candidates, 3)
        self.assertEqual([c["timestamp"] for c in result], [0, .6, 1.2])
        self.assertEqual(len(candidates), 5)

    def test_diversity_cannot_promote_a_much_worse_crop(self):
        candidates = [self.candidate(0, .8), self.candidate(.01, .79), self.candidate(10, .4)]
        self.assertEqual([c["timestamp"] for c in select_plate_candidates(candidates, 2)], [0, .01])

    @patch("app.services.pipeline.read_plate_text", return_value=("1กข 1234", .8))
    def test_ocr_budget_stays_at_three_crops(self, reader):
        candidates = [self.candidate(i * .5, .8) for i in range(5)]
        from app.core.config import settings
        with patch.object(settings, "plate_ocr_candidate_limit", 3):
            _, reads = finalize_plate_candidates(candidates)
        self.assertEqual(reader.call_count, 3)
        self.assertEqual(len(reads), 3)

    def test_collection_uses_source_time_and_stays_bounded(self):
        manager = RiderTrackManager(120, 180, 90, 360, source_fps=60)
        track = manager.violation_track(1, 0)
        candidate = self.candidate(0, .8)
        for number in range(0, 600, 60):
            manager.collect_plate_candidate(track, candidate["plate_box"], number, candidate["crop"])
        self.assertLessEqual(len(track["plate_candidates"]), 5)
        self.assertTrue(all(c["timestamp"] < 10 for c in track["plate_candidates"]))


if __name__ == "__main__":
    unittest.main()
