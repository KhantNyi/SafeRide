"""Build a street-level continuation dataset, including former validation data."""
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'training/round6'
DEST = ROOT / 'train-data6'
NAMES = ['Helmet', 'License Plate', 'Motorcycle', 'No Helmet', 'No License Plate', 'Person']
# All were training clips in round 4. They monitor this continuation only;
# the starting weights have seen them, so this is NOT an independent test.
MONITOR_GROUPS = {'IMG_7080', 'IMG_7082', 'IMG_7086', 'IMG_7094'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pixels(im):
    return hashlib.sha256(str(im.size).encode() + im.convert('RGB').tobytes()).hexdigest()


def group(name):
    match = re.search(r'IMG_\d+', name, re.I)
    return match[0].upper() if match else name


def main():
    assert not OUT.exists() and not DEST.exists(), 'Refusing to overwrite round 6'
    manifests = {n: json.loads((ROOT/f'train-data{n}/manifest.json').read_text(encoding='utf-8')) for n in (4, 5)}
    lookup = {n: {r['image']: r for r in m['records']} for n, m in manifests.items()}
    # Old manifests hashed pixels before EXIF normalization. Recompute these
    # conflict identities in the same upright coordinate system used here.
    conflicts = set()
    for r in manifests[4]['records']:
        if r.get('conflicting_duplicate'):
            with Image.open(ROOT/'train-data4/images'/r['image']) as im:
                conflicts.add(pixels(ImageOps.exif_transpose(im).convert('RGB')))
    previous_val_groups = set()
    previous_val_names = set()
    for n in (3, 4, 5):
        for task in ('helmet', 'plate'):
            for p in (ROOT/f'training/round{n}/datasets/{task}/images/val').iterdir():
                previous_val_groups.add(group(p.name))
                previous_val_names.add(p.stem)
    assert MONITOR_GROUPS.isdisjoint(previous_val_groups)
    inventory = {str(p.relative_to(ROOT)): sha(p) for base in (ROOT/'models', ROOT/'training') for p in base.rglob('*.pt')}
    records, exclusions, duplicate_decisions, source_hashes = [], [], [], {}
    # Newest reviewed crops take precedence over their older full-frame copies.
    seen = {}
    for n in (5, 4, 3):
        source = ROOT/f'train-data{n}'
        for p in sorted((source/'images').iterdir()):
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            label = source/'labels'/p.with_suffix('.txt').name
            assert label.exists(), p
            source_hashes[str(p.relative_to(ROOT))] = sha(p)
            source_hashes[str(label.relative_to(ROOT))] = sha(label)
            meta = lookup.get(n, {}).get(p.name, {})
            with Image.open(p) as original:
                orientation = original.getexif().get(274, 1)
                im = ImageOps.exif_transpose(original).convert('RGB')
                digest = pixels(im)
            original_digest = meta.get('original_pixel_sha256', digest)
            if digest in conflicts and n != 5:
                exclusions.append(dict(image=str(p), reason='Unresolved conflicting labels for identical pixels'))
                continue
            identity = original_digest
            if identity in seen:
                duplicate_decisions.append(dict(image=str(p), retained=seen[identity]))
                continue
            rows = []
            for line in label.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5, (label, line)
                cls = int(parts[0]); x, y, w, h = map(float, parts[1:])
                assert 0 <= cls < len(NAMES) and all(math.isfinite(v) for v in (x,y,w,h))
                assert w > 0 and h > 0 and min(x-w/2, y-h/2) >= -.001 and max(x+w/2, y+h/2) <= 1.001
                # Round tiny export boundary noise to valid clipped boxes.
                left, top, right, bottom = max(0,x-w/2), max(0,y-h/2), min(1,x+w/2), min(1,y+h/2)
                rows.append([cls,(left+right)/2,(top+bottom)/2,right-left,bottom-top])
            capture = group(meta.get('group', p.name))
            ignore_helmet = bool(meta.get('ignore_helmet')) or capture in {'IMG_7439', 'IMG_7278'}
            image = f'r{n}__{p.stem}.png'
            record = dict(source=f'train-data{n}', source_image=str(p), source_label=str(label),
                          image=image, group=capture, pixel_sha256=digest, original_pixel_sha256=original_digest,
                          orientation_normalized=orientation != 1, rows=rows, ignore_helmet=ignore_helmet,
                          former_validation=capture in previous_val_groups,
                          split='val' if capture in MONITOR_GROUPS else 'train')
            records.append(record)
            seen[identity] = str(p)
    assert all(r['split']=='train' for r in records if r['former_validation'])
    assert any('IMG_6712_f000300' in r['image'] and r['split']=='train' and sum(row[0]==0 for row in r['rows'])==2 for r in records)
    for key in ('group', 'pixel_sha256', 'original_pixel_sha256'):
        assert {r[key] for r in records if r['split']=='train'}.isdisjoint({r[key] for r in records if r['split']=='val'})
    OUT.mkdir(parents=True)
    for sub in ('images','labels'):
        (DEST/sub).mkdir(parents=True)
    (DEST/'classes.txt').write_text('\n'.join(NAMES)+'\n')
    for r in records:
        with Image.open(r['source_image']) as im:
            ImageOps.exif_transpose(im).convert('RGB').save(DEST/'images'/r['image'])
        (DEST/'labels'/Path(r['image']).with_suffix('.txt')).write_text('\n'.join(str(row[0])+' '+' '.join(f'{v:.9f}' for v in row[1:]) for row in r['rows'])+'\n')
    counts = {}
    for task, mapping, names in [('helmet',{0:0,3:1},['with helmet','without helmet']), ('plate',{1:0},['license plate'])]:
        folder = OUT/'datasets'/task
        stats = defaultdict(Counter)
        for split in ('train','val'):
            for sub in ('images','labels'):
                (folder/sub/split).mkdir(parents=True)
        for r in records:
            if task == 'helmet' and r['ignore_helmet']:
                continue
            split = r['split']
            shutil.copy2(DEST/'images'/r['image'],folder/'images'/split/r['image'])
            rows = [str(mapping[row[0]])+' '+' '.join(f'{v:.9f}' for v in row[1:]) for row in r['rows'] if row[0] in mapping]
            (folder/'labels'/split/Path(r['image']).with_suffix('.txt')).write_text('\n'.join(rows)+('\n' if rows else ''))
            stats[split]['images'] += 1
            stats[split]['former_validation_images'] += r['former_validation']
            for row in r['rows']:
                if row[0] in mapping:
                    stats[split][names[mapping[row[0]]]] += 1
        assert all(stats[split][name] for split in ('train','val') for name in names)
        (folder/'dataset.yaml').write_text(f'path: {folder.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(names)))
        counts[task] = dict(stats)
    audit = dict(initialization='v5', street_level_only=True,
                 excluded_view_sources={'train-data':111,'train-data2':214},
                 visual_review='All 741 source thumbnails reviewed in analysis/round6-audit; rounds 1-2 overhead, rounds 3-5 street-level.',
                 monitoring_groups=sorted(MONITOR_GROUPS),
                 evaluation_limit='Monitoring clips were in earlier training. No independent unseen accuracy estimate; former validation is now training.',
                 counts=counts, unique_images=len(records), exclusions=exclusions, duplicate_decisions=duplicate_decisions, records=records)
    (OUT/'prior-model-inventory.json').write_text(json.dumps(inventory,indent=2))
    (OUT/'source-inventory.json').write_text(json.dumps(source_hashes,indent=2))
    (OUT/'app-env-sha256.txt').write_text(sha(ROOT/'.env'))
    for path in (OUT/'data-audit.json',DEST/'manifest.json'):
        path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('records','exclusions','duplicate_decisions')},indent=2))
    print('Excluded conflicts:',len(exclusions),'Duplicate copies:',len(duplicate_decisions),flush=True)


if __name__ == '__main__':
    main()
