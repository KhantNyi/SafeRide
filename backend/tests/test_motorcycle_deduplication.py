"""Regression coverage for duplicate whole-bike/rear-bike detections."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.pipeline import RiderTrackManager, analyze_frame
from app.services.byte_tracker import TrackedDetection


class MotorcycleDeduplicationTests(unittest.TestCase):
    def analysis(self, objects):
        object_model = Mock()
        object_model.predict.return_value = [objects]
        plate_model = Mock()
        plate_model.predict.return_value = [[]]
        head = {'label': 'without helmet', 'class_id': 1, 'confidence': .73,
                'xyxy': [791, 2010, 884, 2102]}
        with patch('app.services.pipeline.extract_boxes', side_effect=lambda value: value), \
             patch('app.services.pipeline.detect_helmet_boxes', return_value=[head]), \
             patch('app.services.pipeline.predict_kwargs', return_value={}):
            return analyze_frame(None, (object_model, Mock(), plate_model))

    def test_duplicate_detector_boxes_do_not_spawn_second_rider_identity(self):
        # Actual boxes from IMG_7086 frame 365. At detector NMS's looser
        # overlap threshold, both survived and the association jumped to ID 2.
        bike = {'label': 'motorcycle', 'class_id': 3, 'confidence': .595,
                'xyxy': [788, 2252, 929, 2468]}
        duplicate = {'label': 'motorcycle', 'class_id': 3, 'confidence': .434,
                     'xyxy': [745, 2198, 929, 2470]}
        person = {'label': 'person', 'class_id': 0, 'confidence': .837,
                  'xyxy': [733, 2027, 930, 2396]}
        manager = RiderTrackManager(120, 180, 90, 360)
        previous = self.analysis(deepcopy([bike, person]))
        manager.update(previous, 364)
        current = self.analysis(deepcopy([bike, duplicate, person]))
        manager.update(current, 365)
        self.assertEqual(len(current['motorcycles']), 2)
        self.assertEqual(len(current['associations']), 1)
        self.assertTrue(any(current['associations'][0]['motorcycle_box'] is b for b in current['motorcycles']))
        self.assertEqual(current['associations'][0]['track_id'], previous['associations'][0]['track_id'])
        self.assertEqual(manager.active_violation_track_ids(), {1})

    def test_nearby_distinct_motorcycles_keep_separate_ids(self):
        bikes = [
            {'label': 'motorcycle', 'class_id': 3, 'confidence': .7, 'xyxy': [100, 100, 200, 300]},
            {'label': 'motorcycle', 'class_id': 3, 'confidence': .6, 'xyxy': [170, 100, 270, 300]},
        ]
        manager = RiderTrackManager(120, 180, 90, 360)
        for frame in [0, 3]:
            analysis = self.analysis(deepcopy(bikes))
            manager.update(analysis, frame)
            self.assertEqual(len(analysis['motorcycles']), 2)
            self.assertEqual({b['track_id'] for b in analysis['motorcycles']}, {1, 2})

    def test_established_tracks_are_not_merged_when_their_boxes_overlap(self):
        manager = RiderTrackManager(120, 180, 90, 360)
        bikes = [{'xyxy': [100, 200, 200, 400], 'confidence': .8},
                 {'xyxy': [110, 200, 210, 400], 'confidence': .7}]
        manager.tracker = Mock()
        manager.tracker.update.return_value = [
            TrackedDetection(i+1, b['xyxy'], b['confidence'], {'index': i}, 'tracked', 5)
            for i, b in enumerate(bikes)]
        manager.tracker.active_track_ids.return_value = {1, 2}
        analysis = {'motorcycles': bikes, 'associations': [],
                    'people': [{'xyxy': [100, 100, 200, 350]}]}
        manager.update(analysis, 30)
        self.assertEqual([b['track_id'] for b in bikes], [1, 2])

    def test_duplicate_alias_keeps_event_active_after_original_raw_track_ends(self):
        manager = RiderTrackManager(120, 180, 90, 360)
        manager.track_aliases = {1: 1, 2: 1}
        manager.tracker = Mock()
        manager.tracker.active_track_ids.return_value = {2}
        event = manager.violation_track(1, 0)
        event['pending_started_frame'] = 0
        self.assertFalse(manager.pending_ready(event, 30))
        self.assertTrue(manager.pending_ready(event, 180))

    def test_distinct_passage_is_not_suppressed_at_previous_bikes_position(self):
        manager = RiderTrackManager(120, 180, 90, 360)
        bikes = [{'xyxy': [100, 200, 200, 400], 'confidence': .8},
                 {'xyxy': [400, 200, 500, 400], 'confidence': .7}]
        manager.update({'motorcycles': bikes, 'associations': [], 'people': []}, 0)
        first = {'track_id': 1, 'motorcycle_box': bikes[0]}
        manager.mark_track_saved(manager.violation_track(1, 0), first, 0)
        manager.record_save(first, 0)
        following = {'track_id': 2, 'motorcycle_box': {'xyxy': bikes[0]['xyxy']}}
        self.assertIsNone(manager.saved_duplicate_signature(following, 60))
        self.assertFalse(manager.is_duplicate_save(following, 60))


if __name__ == '__main__':
    unittest.main()
