"""Compare the pink helmet at four fixed views; familiar-clip diagnostic only."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'training/.ultralytics')


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection,1)


def main():
    import cv2
    from ultralytics import YOLO
    out=ROOT/'training/round6/pink-helmet-check'
    out.mkdir(exist_ok=False)
    cap=cv2.VideoCapture(r'C:\Users\ADMIN\Downloads\Sp vd\IMG_6712.MOV')
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
    # Approximate pink-head locations verified against the earlier clip metadata.
    views={300:[570,781,652,875],318:[401,791,460,853],342:[239,790,287,838],366:[134,781,175,829]}
    results=[]
    for version in (5,6):
        model=YOLO(str(ROOT/f'training/round{version}/models/helmet-v{version}-best.pt'))
        for number,region in views.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES,number);ok,frame=cap.read();assert ok
            pred=model.predict(frame,imgsz=960,conf=.35,device=0,verbose=False)[0]
            matches=[dict(label=pred.names[int(box.cls.item())],confidence=float(box.conf.item()),xyxy=box.xyxy[0].tolist())
                     for box in pred.boxes if iou(box.xyxy[0].tolist(),region)>.3]
            results.append(dict(version=version,frame=number,pink_head_region=region,detections=matches))
            annotated=pred.plot();x,y,X,Y=region
            crop=annotated[max(0,y-55):Y+100,max(0,x-60):X+100]
            cv2.imwrite(str(out/f'v{version}-frame{number}.jpg'),crop)
    cap.release()
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
