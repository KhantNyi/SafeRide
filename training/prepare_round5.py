"""Prepare reviewed round-five exports without changing the source folders."""
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "train-data5"
OUT = ROOT / "training/round5"
SOURCES = [Path(r"C:\Users\ADMIN\Downloads") / n for n in ("train-video-footage", "train-data5")]
NAMES = ["Helmet", "License Plate", "Motorcycle", "No Helmet", "No License Plate", "Person"]
# Duplicate decisions follow visual review, keeping one annotation set per image.
DROP = {"97978527-IMG_7438.PNG", "27ac0032-IMG_7439.PNG", "2fa07b29-IMG_7439.PNG", "c2469a48-IMG_7440.PNG"}
AMBIGUOUS_HEADS = {"c3486d0a-IMG_7439.PNG", "f34e7e99-IMG_7278.jpeg"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_sha(im):
    return hashlib.sha256(str(im.size).encode() + im.convert("RGB").tobytes()).hexdigest()


def group(name):
    match = re.search(r"IMG_\d+", name, re.I)
    return match[0].upper() if match else name


def main():
    assert not DEST.exists() and not OUT.exists(), "Refusing to overwrite a previous dataset or run"
    records, source_counts, removed = [], {}, []
    for folder in SOURCES:
        names = (folder / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
        assert [n.lower().replace(" ", "") for n in names] == [n.lower().replace(" ", "") for n in NAMES[:len(names)]]
        metadata = {r["image"]: r for r in json.loads((folder / "manifest.json").read_text())} if (folder / "manifest.json").exists() else {}
        images = {p.stem: p for p in (folder / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        labels = {p.stem: p for p in (folder / "labels").glob("*.txt")}
        assert images.keys() == labels.keys(), "Unpaired labels or images"
        source_counts[folder.name] = len(images)
        for stem, path in sorted(images.items()):
            with Image.open(path) as source:
                orientation = source.getexif().get(274, 1)
                im = ImageOps.exif_transpose(source).convert("RGB")
                im.load()
            rows = []
            for line in labels[stem].read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5
                cls = int(parts[0]); x, y, w, h = map(float, parts[1:])
                assert 0 <= cls < len(names) and all(math.isfinite(v) for v in (x,y,w,h))
                assert min(w,h) > 0 and min(x-w/2,y-h/2) >= -.000001 and max(x+w/2,y+h/2) <= 1.000001
                rows.append([cls,x,y,w,h])
            if path.name in DROP:
                removed.append({"source":str(path), "reason":"visually reviewed redundant/conflicting duplicate", "sha256":sha(path)})
                continue
            provenance = metadata.get(path.name, {})
            if provenance.get("image_sha256"):
                assert sha(path) == provenance["image_sha256"], "Footage changed since its visual review"
            W,H = im.size
            crop = [0,0,W,H]
            original_pixel_sha = pixel_sha(im)
            if folder.name == "train-data5":
                # Context around every labeled object: remove screenshot bars and
                # distant unlabeled background while retaining all annotations.
                crop = [max(0, math.floor(min((r[1]-r[3]/2)*W for r in rows)-W*.06)),
                        max(0, math.floor(min((r[2]-r[4]/2)*H for r in rows)-H*.04)),
                        min(W, math.ceil(max((r[1]+r[3]/2)*W for r in rows)+W*.06)),
                        min(H, math.ceil(max((r[2]+r[4]/2)*H for r in rows)+H*.04))]
                a,b,c,d = crop
                rows = [[cl,(x*W-a)/(c-a),(y*H-b)/(d-b),w*W/(c-a),h*H/(d-b)] for cl,x,y,w,h in rows]
                im = im.crop(crop)
            record = dict(source=folder.name, original_image=str(path), original_label=str(labels[stem]),
                original_sha256=sha(path), original_label_sha256=sha(labels[stem]),
                original_pixel_sha256=original_pixel_sha, pixel_sha256=pixel_sha(im),
                image=f"{folder.name}__{stem}.png", original_size=[W,H], crop_xyxy=crop,
                orientation_normalized=orientation != 1, rows=rows, provenance=provenance,
                group=group(provenance.get("source_clip",path.name)),
                ignore_helmet=path.name in AMBIGUOUS_HEADS)
            records.append(record)
    # Guard old validation sets against exact-image and known capture-ID overlap.
    old_hashes, old_val_hashes, old_groups, old_val_groups = set(),set(),set(),set()
    for folder in ("train-data", "train-data2", "train-data3", "train-data4"):
        for p in (ROOT/folder/"images").iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
            with Image.open(p) as im: old_hashes.add(pixel_sha(ImageOps.exif_transpose(im)))
            old_groups.add(group(p.name))
    for rnd in (3,4):
        for p in (ROOT/f"training/round{rnd}/datasets/helmet/images/val").iterdir():
            with Image.open(p) as im: old_val_hashes.add(pixel_sha(ImageOps.exif_transpose(im)))
            old_val_groups.add(group(p.name))
    val_candidates = sorted({r['group'] for r in records if r['source']=='train-video-footage'
        and r['group'] not in old_groups and r['original_pixel_sha256'] not in old_hashes})
    random.Random(42).shuffle(val_candidates)
    val_groups = set(val_candidates[:max(1,round(len(val_candidates)*.2))])
    assert val_groups, "No clean validation clips"
    # All new still exports belong to one road/capture session and stay together
    # in training. Validation uses complete, previously unseen footage clips.
    for r in records:
        r['prior_exact_overlap'] = r['original_pixel_sha256'] in old_hashes
        r['split'] = 'excluded' if (r['group'] in old_val_groups or r['original_pixel_sha256'] in old_val_hashes) else ('val' if r['group'] in val_groups else 'train')
    by_hash=defaultdict(list)
    for r in records: by_hash[r['pixel_sha256']].append(r)
    assert all(len(v)==1 for v in by_hash.values()), "Unresolved duplicates"
    assert {r['group'] for r in records if r['split']=='train'}.isdisjoint({r['group'] for r in records if r['split']=='val'})
    OUT.mkdir(parents=True)
    inventory = {str(p.relative_to(ROOT)):sha(p) for base in (ROOT/'models',ROOT/'training') for p in base.rglob('*.pt')}
    (OUT/'prior-model-inventory.json').write_text(json.dumps(inventory,indent=2))
    for sub in ('images','labels'): (DEST/sub).mkdir(parents=True)
    (DEST/'classes.txt').write_text('\n'.join(NAMES)+'\n')
    for folder in SOURCES:
        target=DEST/'source_metadata'/folder.name; target.mkdir(parents=True)
        for p in folder.iterdir():
            if p.is_file(): shutil.copy2(p,target/p.name)
    for r in records:
        with Image.open(r['original_image']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB').crop(r['crop_xyxy'])
            im.save(DEST/'images'/r['image'])
        text='\n'.join(str(row[0])+' '+' '.join(f'{v:.9f}' for v in row[1:]) for row in r['rows'])+'\n'
        (DEST/'labels'/(Path(r['image']).stem+'.txt')).write_text(text)
    task_summaries={}
    for task,mapping,names in [('helmet',{0:0,3:1},['with helmet','without helmet']),('plate',{1:0},['license plate'])]:
        folder=OUT/'datasets'/task; counts=defaultdict(Counter)
        for split in ('train','val'):
            for sub in ('images','labels'): (folder/sub/split).mkdir(parents=True)
        for r in records:
            split=r['split']
            if split=='excluded' or (task=='helmet' and r['ignore_helmet']): continue
            shutil.copy2(DEST/'images'/r['image'],folder/'images'/split/r['image'])
            lines=[str(mapping[row[0]])+' '+' '.join(f'{v:.9f}' for v in row[1:]) for row in r['rows'] if row[0] in mapping]
            (folder/'labels'/split/(Path(r['image']).stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''))
            counts[split]['images']+=1
            for row in r['rows']:
                if row[0] in mapping: counts[split][names[mapping[row[0]]]]+=1
        assert all(counts['val'][n]>0 and counts['train'][n]>0 for n in names)
        (folder/'dataset.yaml').write_text(f'path: {folder.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(names)))
        task_summaries[task]=dict(counts)
    report=dict(sources=source_counts,merged_images=len(records),removed_duplicates=removed,
        orientation_normalized=sum(r['orientation_normalized'] for r in records),
        prior_exact_overlap=sum(r['prior_exact_overlap'] for r in records),
        splits=dict(Counter(r['split'] for r in records)),validation_groups=sorted(val_groups),
        ambiguous_helmet_exclusions=sorted(AMBIGUOUS_HEADS),tasks=task_summaries,records=records)
    (DEST/'manifest.json').write_text(json.dumps(report,indent=2))
    (OUT/'data-audit.json').write_text(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__': main()
