"""Source-only dense contact sheets for independent event annotation."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
DEST = OUT / 'review' / 'labeling'
DEST.mkdir(exist_ok=True)
data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
for item in data['inventory']:
    cap = cv2.VideoCapture('C:/Users/ADMIN/Downloads/Sp data 2 september/' + item['clip'])
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    tiles = []
    frame_index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_index % 15 == 0 or frame_index == item['frames'] - 1:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                # Full road width; exclude only treetops and near empty pavement.
                crop = frame[int(h*.40):int(h*.82), :]
                im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                im.thumbnail((400, 300))
                tile = Image.new('RGB', (400, 325), 'white')
                tile.paste(im, (0,25))
                ImageDraw.Draw(tile).text((5,5), f"{item['clip']} f{frame_index} {frame_index/item['fps']:.2f}s",fill='black')
                tiles.append(tile)
        frame_index += 1
    cap.release()
    for start in range(0, len(tiles), 20):
        batch = tiles[start:start+20]
        sheet = Image.new('RGB',(1600,325*((len(batch)+3)//4)),'#ddd')
        for i,tile in enumerate(batch): sheet.paste(tile,(i%4*400,i//4*325))
        sheet.save(DEST / f"{Path(item['clip']).stem}-{start//20}.jpg", quality=90)
    print(item['clip'],len(tiles),'review frames',flush=True)
