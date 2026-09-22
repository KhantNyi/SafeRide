import json
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'backend'))
from app.services.pipeline import appearance_feature
from app.services.byte_tracker import ByteTracker, ByteTrackDetection, box_iou, motion_match_score, feature_similarity
from app.core.config import settings

p = Path(__file__).resolve().parent / 'v3'
d = json.loads((p/'results.json').read_text(encoding='utf-8'))
clip = next(c for c in d['runs'][1]['clips'] if c['clip'] == 'IMG_7090.MOV')
frames = json.loads((p/'media/metadata'/f"{clip['job']['id']}_detections.json").read_text())['frames']
cap = cv2.VideoCapture('C:/Users/ADMIN/Downloads/Sp data 2 september/'+clip['clip'])
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
t = ByteTracker(high_threshold=.25, low_threshold=.1, new_track_threshold=.25, match_threshold=.25, max_time_lost=89, appearance_weight=.3)
at = -1
for f in frames:
    while at < f['frame_number']:
        cap.grab(); at += 1
    ok, im = cap.retrieve()
    ds = [ByteTrackDetection(xyxy=b['xyxy'], score=b['confidence'], metadata={'index':i}, feature=appearance_feature(im,b['xyxy'])) for i,b in enumerate(f['motorcycles'])]
    old = t.next_track_id
    candidates = [(tr['id'], at-tr['last_frame'], box_iou(tr['xyxy'], det.xyxy), motion_match_score(tr['kalman'].predict_box(at-tr['last_frame']),det.xyxy), feature_similarity(tr['feature'],det.feature)) for tr in t.tracks for det in ds]
    tracked = t.update(ds, at)
    if t.next_track_id != old:
        print(at, 'new', [(x.track_id,x.xyxy) for x in tracked if x.track_id>=old], 'candidates', candidates)
cap.release()
