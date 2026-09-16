"""Create source-only contact sheets for visual reference annotation."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
dest = OUT / "review/source"
dest.mkdir(parents=True, exist_ok=True)
manifest = []
for c in data["clips"]:
    cap = cv2.VideoCapture(str(Path(data["source"]) / c["clip"]))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    tiles = []
    interval = max(1, round(c["fps"] * .5))
    frame_number = 0
    while cap.grab():
        if frame_number % interval == 0 or frame_number == c["frames"] - 1:
            ok, frame = cap.retrieve()
            if ok:
                h,w = frame.shape[:2]
                crop = frame[int(h*.28):int(h*.80), :]
                im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                im.thumbnail((400,370))
                tile = Image.new("RGB", (400,395), "white")
                tile.paste(im,(0,25))
                ImageDraw.Draw(tile).text((5,5), f'{c["clip"]} f{frame_number} {frame_number/c["fps"]:.2f}s',fill="black")
                tiles.append(tile)
        frame_number += 1
    cap.release()
    paths = []
    for start in range(0,len(tiles),24):
        batch = tiles[start:start+24]
        sheet = Image.new("RGB", (1600,395*((len(batch)+3)//4)), "#ddd")
        for i,tile in enumerate(batch):
            sheet.paste(tile,(i%4*400,i//4*395))
        path = dest / f'{Path(c["clip"]).stem}-{start//24}.jpg'
        sheet.save(path,quality=94)
        paths.append(str(path.relative_to(OUT)))
    manifest.append(dict(clip=c["clip"],frames=len(tiles),sheets=paths))
    print(c["clip"], len(tiles), "frames", len(paths), "sheets",flush=True)
(dest / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
