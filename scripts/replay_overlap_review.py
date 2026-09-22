"""Replay identical cached detections through tracking, without rerunning YOLO/OCR.

This isolates association changes. It is not a detector or OCR accuracy test.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import cv2

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',type=Path,default=ROOT/'backend')
    parser.add_argument('--name',required=True)
    parser.add_argument('--all-clips',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.backend.resolve()))
    from app.services import pipeline as p
    p.settings.enable_ocr=False
    out=ROOT/'analysis/overlap-fix'/args.name
    out.mkdir(parents=True,exist_ok=True)
    jobs=[('IMG_6665-0.5','ea4b8bad9b0d49788408ca64c7b1b51a'),
          ('IMG_6665-1.0','1f80053a5faa43ad99d1c87773e83977'),
          ('IMG_6805-0.5','469013c45c0f4b0783e800cc7ff3f7d4')]
    cases=[(name,ROOT/f'data/metadata/{id}_detections.json',ROOT/f'data/uploads/{id}.mov') for name,id in jobs]
    if args.all_clips:
        previous=json.loads((ROOT/'analysis/sp-vd-v4-full-s050/results.json').read_text(encoding='utf-8'))
        for clip in previous['clips']:
            cases.append((Path(clip['clip']).stem+'-fixed0.5',
                ROOT/f"analysis/sp-vd-v4-full-s050/media/metadata/{clip['job']['id']}_detections.json",
                Path(r'C:\Users\ADMIN\Downloads\Sp vd')/clip['clip']))
    summaries=[]
    for name,metadata,video in cases:
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
        assert cap.isOpened(),video
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        manager=p.RiderTrackManager(int(fps*4),int(fps*6),int(fps*3),int(fps*12),fps)
        frames=json.loads(metadata.read_text(encoding='utf-8'))['frames']
        saved=[];traces=[];pos=0
        def record(payload):
            number=len(saved); plate=payload.get('plate_candidate')
            cv2.imwrite(str(out/f'{name}-{number}-evidence.jpg'),payload['annotated'])
            if plate:cv2.imwrite(str(out/f'{name}-{number}-plate.jpg'),plate['crop'])
            saved.append({'frame':payload['frame_number'],
                'association':p.serialize_association(payload['association']),
                'plate_frame':plate.get('frame_number') if plate else None,
                'plate_box':plate['plate_box']['xyxy'] if plate else None})
        for f in frames:
            target=f['frame_number']
            while pos<target:
                assert cap.grab();pos+=1
            ok,frame=cap.read();pos+=1
            assert ok and list(frame.shape[:2])==[f['height'],f['width']]
            a={key:deepcopy(f.get(key,[])) for key in ('motorcycles','people','helmets','no_helmets','plates')}
            # Old metadata did not store negative vehicles. Their absence is a
            # shared limitation for BOTH replay variants, explicitly reported.
            a['negative_vehicles']=[]
            for b in a['motorcycles']:
                b.pop('track_id',None);b.pop('association_uncertain',None)
            a['associations']=p.associate_riders(a['people'],a['motorcycles'],a['helmets'],a['no_helmets'],a['plates'],[])
            a['has_no_helmet']=any(v.get('helmet_status')=='no_helmet' for v in a['associations'])
            manager.update(a,target,frame)
            annotated=p.annotate_analysis(frame,target,a,fresh_analysis=True)
            traces.append(p.serialize_detection_frame(target,fps,frame,a))
            for payload in manager.violations_to_save(a['associations'],target,frame,annotated):record(payload)
        for payload in manager.pending_violations_to_save():record(payload)
        cap.release()
        summary={'case':name,'frames':len(frames),'track_ids_created':manager.tracker.next_track_id-1,
                 'records':saved,'record_count':len(saved),
                 'uncertain_motorcycle_observations':sum(b.get('association_uncertain',False) for f in traces for b in f['motorcycles'])}
        summaries.append(summary)
        (out/f'{name}-trace.json').write_text(json.dumps(traces,ensure_ascii=False),encoding='utf-8')
        (out/'results.json').write_text(json.dumps({'limitations':'Cached detections; OCR disabled; negative vehicle boxes unavailable in legacy metadata. Counts are not accuracy.',
                                                  'cases':summaries},indent=2),encoding='utf-8')
        print(name,'records',len(saved),'IDs',summary['track_ids_created'],flush=True)


if __name__=='__main__':main()
