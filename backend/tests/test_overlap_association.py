"""General overlap, missing-bike, ownership and sampling regressions."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.assignment import match_scores
from app.services.byte_tracker import ByteTracker, ByteTrackDetection
from app.services.pipeline import (RiderTrackManager, associate_riders,
    best_motorcycle_for_person, assign_plates_to_motorcycles)


def box(xyxy, label='motorcycle'):
    return dict(xyxy=xyxy, confidence=.9, label=label)


class AssignmentTests(unittest.TestCase):
    def test_joint_matching_avoids_greedy_identity_swap(self):
        self.assertEqual(set(match_scores([[.9,.8],[.85,.1]],.25)), {(0,1),(1,0)})

    def test_unmatched_choice_does_not_force_bad_pair(self):
        self.assertEqual(match_scores([[.9,.1],[.2,.1]],.25),[(0,0)])
        self.assertEqual(match_scores(np.zeros((0,2)),.25),[])

    def test_front_rider_does_not_attach_to_rear_bike_when_front_bike_missing(self):
        # Test different positions/scales; not tied to a camera or clip.
        for scale,offset in [(1,0),(2,300),(.5,100)]:
            def b(coords):return box([int(v*scale+offset) for v in coords])
            rear=b([100,200,200,400]);front_person=b([140,70,190,210])
            own_person=b([100,110,200,340])
            self.assertIsNone(best_motorcycle_for_person(front_person,[rear])[0])
            self.assertIs(best_motorcycle_for_person(own_person,[rear])[0],rear)

    def test_driver_and_passenger_can_share_one_motorcycle(self):
        bike=box([100,200,260,430])
        people=[box([100,100,200,370]),box([160,120,260,390])]
        heads=[box([120,100,150,130]),box([200,120,230,150])]
        result=associate_riders(people,[bike],[heads[0]],[heads[1]],[],[])
        self.assertEqual({a['helmet_status'] for a in result},{'with_helmet','no_helmet'})
        self.assertTrue(all(a['motorcycle_box'] is bike for a in result))

    def test_ambiguous_bikes_do_not_receive_plates(self):
        bikes=[box([100,200,200,400]),box([110,205,210,405])]
        for b in bikes:b['association_uncertain']=True
        self.assertEqual(assign_plates_to_motorcycles([box([140,330,170,360])],bikes,[]),{})


class TemporalTests(unittest.TestCase):
    def manager(self):return RiderTrackManager(120,180,90,360,30)

    def test_dense_samples_and_duplicate_head_boxes_do_not_inflate_votes(self):
        manager=self.manager()
        for frame in range(7):
            manager.record_helmet_vote(1,'no_helmet',frame)
            manager.record_helmet_vote(1,'no_helmet',frame)
        self.assertEqual(manager.helmet_votes[1]['no_helmet'],2)

    def test_conflicting_plate_does_not_enter_existing_candidates(self):
        m=self.manager();track=m.violation_track(1,0)
        bike=box([0,0,100,200]);frame=np.full((220,120,3),160,np.uint8)
        for f in [0,6]:m.collect_plate_candidate(track,box([40,140,60,160]),f,frame,motorcycle=bike)
        m.collect_plate_candidate(track,box([5,50,25,70]),12,frame,motorcycle=bike)
        self.assertEqual(track['plate_sightings'],2)
        self.assertEqual(len(track['plate_candidates']),2)

    def test_different_plate_colour_cannot_contaminate_ocr_buffer(self):
        m=self.manager();track=m.violation_track(1,0)
        bike=box([0,0,100,200]);plate=box([40,140,60,160])
        yellow=np.full((220,120,3),(0,210,230),np.uint8)
        white=np.full((220,120,3),230,np.uint8)
        for f in [0,6]:
            m.collect_plate_candidate(track,plate,f,yellow,motorcycle=bike)
        m.collect_plate_candidate(track,plate,12,white,motorcycle=bike)
        self.assertEqual(track['plate_sightings'],2)
        self.assertEqual([c['frame_number'] for c in track['plate_candidates']],[0,6])

    def test_overlap_pauses_votes_and_crops_then_recovers(self):
        m=self.manager();frame=np.zeros((500,500,3),np.uint8)
        bikes=[box([50,200,150,400]),box([300,200,400,400])]
        def update(bikes,f):
            analysis={'motorcycles':deepcopy(bikes),'people':[], 'associations':[],
                      'helmets':[], 'no_helmets':[], 'plates':[]}
            m.update(analysis,f,frame)
            return analysis
        before=update(bikes,0)
        overlap=update([box([150,200,250,400]),box([155,200,255,400])],6)
        self.assertTrue(all(b['association_uncertain'] for b in overlap['motorcycles']))
        self.assertEqual(overlap['associations'],[])
        self.assertEqual(m.helmet_votes,{})
        after=update(bikes,12)
        self.assertFalse(any(b['association_uncertain'] for b in after['motorcycles']))

    def test_ambiguous_single_detection_coasts_without_new_identity(self):
        tracker=ByteTracker(high_threshold=.25,low_threshold=.1,new_track_threshold=.25,
                            match_threshold=.25,max_time_lost=90)
        def detection(coords,i,occluded=False):return ByteTrackDetection(coords,.9,{'index':i,'occluded':occluded})
        original=[detection([0,100,100,300],0),detection([120,100,220,300],1)]
        tracker.update(original,0)
        result=tracker.update([detection([60,100,160,300],0,True)],6)
        self.assertEqual(result,[])
        self.assertEqual(tracker.next_track_id,3)
        self.assertEqual({x.track_id for x in tracker.update(original,12)},{1,2})

    def test_expired_track_cannot_adopt_later_arrival(self):
        tracker=ByteTracker(high_threshold=.25,low_threshold=.1,new_track_threshold=.25,
                            match_threshold=.25,max_time_lost=30)
        detection=ByteTrackDetection([0,0,100,200],.9,{'index':0})
        self.assertEqual(tracker.update([detection],0)[0].track_id,1)
        self.assertEqual(tracker.update([detection],60)[0].track_id,2)

    def test_fast_first_step_uses_appearance_then_learns_motion(self):
        tracker=ByteTracker(high_threshold=.25,low_threshold=.1,new_track_threshold=.25,
                            match_threshold=.25,max_time_lost=180,source_fps=60)
        feature=np.array([1.,0.,0.])
        def detection(x):return ByteTrackDetection([x,100,x+100,250],.9,{'index':0},feature)
        self.assertEqual(tracker.update([detection(600)],0)[0].track_id,1)
        self.assertEqual(tracker.update([detection(420)],30)[0].track_id,1)
        self.assertEqual(tracker.update([detection(260)],60)[0].track_id,1)

    def test_overlap_does_not_pollute_appearance(self):
        tracker=ByteTracker(high_threshold=.25,low_threshold=.1,new_track_threshold=.25,
                            match_threshold=.25,max_time_lost=90)
        original=np.array([1.,0.]); mixed=np.array([.5,.5])
        tracker.update([ByteTrackDetection([0,100,100,300],.9,{'index':0},original)],0)
        tracker.update([ByteTrackDetection([0,100,100,300],.9,{'index':0,'occluded':True},mixed)],6)
        np.testing.assert_array_equal(tracker.tracks[0]['feature'],original)

    def test_aligned_motorcycles_with_different_rear_anchors_are_not_aliased(self):
        m=self.manager()
        analysis={'motorcycles':[box([100,200,200,400]),box([100,250,200,450])],
                  'people':[],'associations':[]}
        with patch.object(m,'nearest_rider_index',return_value=0):m.update(analysis,0)
        self.assertEqual(len({b['track_id'] for b in analysis['motorcycles']}),2)
        self.assertTrue(all(b['association_uncertain'] for b in analysis['motorcycles']))


if __name__=='__main__': unittest.main()
