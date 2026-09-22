"""Create source contact sheets and a local evidence browser after timing finishes."""
import html
import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
if '--v3' in sys.argv:
    OUT = OUT / 'v3'
if '--output' in sys.argv:
    OUT = Path(__file__).resolve().parent / sys.argv[sys.argv.index('--output') + 1]
SOURCE = Path('C:/Users/ADMIN/Downloads/Sp data 2 september')
data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
review = OUT / 'review'
review.mkdir(exist_ok=True)

for item in ([] if '--v3' in sys.argv or '--output' in sys.argv else data['inventory']):
    cap = cv2.VideoCapture(str(SOURCE / item['clip']))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    times = [i * item['seconds'] / 8 for i in range(8)]
    sheet = Image.new('RGB', (1200, 1100), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        im.thumbnail((300, 520))
        x, y = i % 4 * 300, i // 4 * 550
        sheet.paste(im, (x, y + 25))
        draw.text((x + 5, y + 5), f"{item['clip']} {t:.1f}s", fill='black')
    cap.release()
    sheet.save(review / f"{Path(item['clip']).stem}-source.jpg")

pages = ['<!doctype html><meta charset="utf-8"><title>September SafeRide evidence</title><style>body{font:16px system-ui;margin:30px;background:#eee}img{max-width:100%;max-height:650px}article{background:white;padding:15px;margin:12px;display:inline-block;vertical-align:top;width:420px}h2{clear:both}</style><h1>September benchmark evidence</h1><p>Model outputs, not ground truth. Source sheets sample eight moments per clip.</p>']
for run in data['runs']:
    pages.append(f"<h2>{run['interval']} second interval; OCR {run['ocr']}</h2>")
    crops = []
    for clip in run['clips']:
        metadata = json.loads((OUT / 'media' / 'metadata' / f"{clip['job']['id']}_detections.json").read_text(encoding='utf-8'))['frames']
        pages.append(f"<h3>{clip['clip']} — {len(clip['violations'])} records — {clip['wall_seconds']:.1f}s</h3>")
        for v in clip['violations']:
            evidence = v['evidence_image'].replace('/media/', 'media/')
            plate = (v['plate_image'] or '').replace('/media/', 'media/')
            pages.append(f'<article><p>Frame {v["frame_number"]}, track {v["track_id"]}, confidence {v["helmet_confidence"]:.2f}</p><a href="{evidence}"><img src="{evidence}"></a><p>OCR: {html.escape(v["plate_text"] or "none")}</p>')
            if plate:
                pages.append(f'<img src="{plate}">')
            pages.append('</article>')
            nearby = sorted(metadata, key=lambda f: abs(f['frame_number'] - v['frame_number']))
            association = next((a for f in nearby for a in f['associations'] if a.get('track_id') == v['track_id']), None)
            if association:
                boxes = [association[k]['xyxy'] for k in ['person_box','motorcycle_box','helmet_box'] if association.get(k)]
                if boxes:
                    im = Image.open(OUT / evidence)
                    box = (max(0, min(b[0] for b in boxes)-40), max(0,min(b[1] for b in boxes)-60), min(im.width,max(b[2] for b in boxes)+40), min(im.height,max(b[3] for b in boxes)+40))
                    crop = im.crop(box)
                    crop.thumbnail((300,380))
                    tile = Image.new('RGB',(320,420),'white')
                    tile.paste(crop,(10,35))
                    ImageDraw.Draw(tile).text((5,5), f"{clip['clip']} f{v['frame_number']} t{v['track_id']}",fill='black')
                    crops.append(tile)
    for start in range(0,len(crops),12):
        batch=crops[start:start+12]
        sheet=Image.new('RGB',(1280,420*((len(batch)+3)//4)),'#ccc')
        for i,tile in enumerate(batch): sheet.paste(tile,(i%4*320,i//4*420))
        sheet.save(review / f"evidence-{run['interval']}-ocr{run['ocr']}-{start//12}.jpg")
(OUT / 'evidence.html').write_text('\n'.join(pages), encoding='utf-8')
print('Wrote source sheets and evidence.html')
