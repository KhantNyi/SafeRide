import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import playback


class PlaybackFlowTests(unittest.TestCase):
    def test_video_pass_bridges_visual_motion_without_joining_detection_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "motion.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 160))
            self.assertTrue(writer.isOpened())
            texture = np.random.default_rng(2).integers(0, 256, (60, 60, 3), dtype=np.uint8)
            for i in range(11):
                image = np.zeros((160, 200, 3), dtype=np.uint8)
                image[40:100, 40 + i:100 + i] = texture
                writer.write(image)
            writer.release()
            frames = []
            for i, identity in ((0, 1), (10, 9)):
                frames.append({"frame_number": i, "timestamp": i / 10, "width": 200, "height": 160,
                    **{g: [] for g in playback.GROUPS}, "associations": []})
                frames[-1]["motorcycles"] = [{"label": "motorcycle", "confidence": .8,
                    "xyxy": [40 + i, 40, 100 + i, 100], "track_id": identity}]
            snapshot = deepcopy(frames)
            output = playback.build_playback(path, frames)
            self.assertEqual(len(output), 11)
            self.assertEqual(frames, snapshot)
            np.testing.assert_allclose(output[5]["motorcycles"][0]["xyxy"], [45, 40, 105, 100], atol=2)
            self.assertEqual(output[5]["motorcycles"][0]["track_id"], 1)
            self.assertEqual(output[-1]["motorcycles"][0]["track_id"], 9)
            frames[-1]["motorcycles"] = []
            rejected = playback.build_playback(path, frames)
            self.assertEqual(rejected[5]["motorcycles"], [])

    def test_translation_and_texture_loss(self):
        previous = np.zeros((160, 200), dtype=np.uint8)
        previous[40:100, 40:100] = np.random.default_rng(2).integers(0, 256, (60, 60), dtype=np.uint8)
        current = cv2.warpAffine(previous, np.float32([[1, 0, 8], [0, 1, 4]]), (200, 160))
        moved = playback.move_box(previous, current, np.array([40., 40., 100., 100.]))
        np.testing.assert_allclose(moved, [48, 44, 108, 104], atol=1)
        self.assertIsNone(playback.move_box(previous, np.zeros_like(previous), np.array([40., 40., 100., 100.])))
        self.assertIsNone(playback.move_box(np.zeros_like(previous), current, np.array([40., 40., 100., 100.])))

    def test_render_is_visual_only_and_drops_lost_associations(self):
        bike = {"label": "motorcycle", "xyxy": [10, 10, 50, 50], "confidence": .8, "track_id": 3}
        base = {"frame_number": 30, "timestamp": 1, "width": 100, "height": 100,
                **{g: [] for g in playback.GROUPS}, "associations": [{"motorcycle_box": bike, "track_id": 3}]}
        base["motorcycles"] = [bike]
        original = deepcopy(base)
        output = playback.render_record(base, {("motorcycles", 0): [12, 10, 52, 50]}, 33, 1.1)
        self.assertEqual(base, original)
        self.assertEqual(output["motorcycles"][0]["track_id"], 3)
        self.assertTrue(output["playback_generated"])
        self.assertEqual(playback.render_record(base, {}, 33, 1.1)["associations"], [])

    def test_separate_output_and_failed_pass_preserve_detection_bytes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(playback.settings, "metadata_dir", Path(directory)):
            source = Path(directory) / "job_detections.json"
            source.write_text('{"frames": []}', encoding="utf-8")
            original = source.read_bytes()
            with patch.object(playback, "build_playback", return_value=[]):
                playback.reserve_playback("job")
                playback.enhance_playback("job", "unused")
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(playback.playback_status("job")["status"], "completed")
            with patch.object(playback, "build_playback", side_effect=ValueError("lost source")):
                playback.reserve_playback("job")
                playback.enhance_playback("job", "unused")
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(playback.playback_status("job")["status"], "failed")
            self.assertIsNone(playback._active)
            playback._states.clear()


if __name__ == "__main__":
    unittest.main()
