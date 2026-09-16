"""Extract event-focused source frames and model-assisted annotation proposals.

The proposals require explicit visual review before publication.
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'training/.ultralytics')
import cv2
from ultralytics import YOLO

SOURCE = Path('C:/Users/ADMIN/Downloads/Sp vd')
BENCH = ROOT/'analysis/sp-vd-v4-full-s050'
OUT = ROOT/'analysis/train-video-footage-staging'
NAMES = ['Helmet', 'License Plate', 'Motorcycle', 'No Helmet', 'No License Plate', 'Person']


def main():
    OUT.mkdir(exist_ok=False)
    for name in ['images', 'labels', 'annotated', 'review']:
        (OUT/name).mkdir()
    references = json.loads((BENCH/'reference-labels.json').read_text(encoding='utf-8'))
    benchmark = json.loads((BENCH/'results.json').read_text(encoding='utf-8'))
    clips = {c['clip']:c for c in benchmark['clips']}
    models = [(YOLO(str(ROOT/'training/round4/models/helmet-v4-best.pt')), 'head'),
              (YOLO(str(ROOT/'training/round4/models/license-plate-v4-best.pt')), 'plate'),
              (YOLO(str(ROOT/'models/yolo11s.pt')), 'object')]
    manifest = []
    for ref in references['clips']:
        clip = clips[ref['clip']]
        meta = json.loads((BENCH/'media/metadata'/f"{clip['job']['id']}_detections.json").read_text())['frames']
        chosen = {e['reference_frame']:'confirmed no-helmet passage '+e['id'] for e in ref['events']}
        candidates = []
        for f in meta:
            if any(u['start_seconds']-.5 <= f['timestamp'] <= u['end_seconds']+.5 for u in ref['uncertain']):
                continue
            heads = f['helmets']
            if not heads:
                continue
            score = max((b['xyxy'][2]-b['xyxy'][0])*(b['xyxy'][3]-b['xyxy'][1])*b['confidence'] for b in heads)
            candidates.append((score, f['frame_number']))
        added = 0
        for score,frame in sorted(candidates,reverse=True):
            if all(abs(frame-old) >= clip['fps']*1.25 for old in chosen):
                chosen[frame] = 'helmeted rider comparison'
                added += 1
                if added >= (1 if chosen and ref['events'] else 2):
                    break
        if not chosen:
            raise RuntimeError(f'No clear sample for {ref["clip"]}')
        cap=cv2.VideoCapture(str(SOURCE/ref['clip']))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
        for frame_number,reason in sorted(chosen.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ok,frame=cap.read()
            assert ok
            h,w=frame.shape[:2]
            crop=[0,int(h*.25),w,int(h*.8)]
            frame=frame[crop[1]:crop[3],:]
            stem=f"{Path(ref['clip']).stem}_f{frame_number:06d}"
            cv2.imwrite(str(OUT/'images'/f'{stem}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,97])
            boxes=[]
            for model,kind in models:
                result=model.predict(frame,imgsz=1280,conf=.18 if kind=='head' else .25,iou=.5,device=0,verbose=False)[0]
                for b in result.boxes:
                    label=model.names[int(b.cls)].lower()
                    if kind=='head':
                        cls=3 if 'without' in label or 'no helmet' in label else 0
                    elif kind=='plate':cls=1
                    elif label=='person':cls=5
                    elif label=='motorcycle':cls=2
                    else:continue
                    boxes.append(dict(class_id=cls,xyxy=[round(float(x)) for x in b.xyxy[0]],confidence=round(float(b.conf),4)))
            rec=dict(image=f'{stem}.jpg',source_clip=ref['clip'],frame_number=frame_number,
                     timestamp=frame_number/clip['fps'],source_size=[w,h],crop_xyxy=crop,
                     image_size=[frame.shape[1],frame.shape[0]],selection_reason=reason,
                     image_sha256=hashlib.sha256((OUT/'images'/f'{stem}.jpg').read_bytes()).hexdigest(),
                     boxes=boxes,review_status='pending')
            manifest.append(rec)
        cap.release()
        print(ref['clip'],len(chosen),flush=True)
    (OUT/'proposals.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'classes.txt').write_text('\n'.join(NAMES)+'\n')
    print('TOTAL',len(manifest),flush=True)

if __name__=='__main__':main()
