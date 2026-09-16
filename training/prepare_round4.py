"""Validate and merge the three exports, then build clip-grouped v4 datasets."""
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\ADMIN\Downloads\more train")
DEST = ROOT / "train-data4"
OUT = ROOT / "training/round4"
NAMES = ["Helmet", "License Plate", "Motorcycle", "No Helmet", "No License Plate", "Person"]


def main():
    records, summaries, seen = [], {}, {}
    for folder in sorted(SOURCE.iterdir()):
        if not folder.is_dir():
            continue
        names = (folder / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
        assert [n.lower().replace(" ", "") for n in names] == [n.lower().replace(" ", "") for n in NAMES[:len(names)]], folder
        provenance = {x["image"]: x for x in json.loads((folder / "manifest.json").read_text())} if (folder / "manifest.json").exists() else {}
        images = {p.stem: p for p in (folder / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        labels = {p.stem: p for p in (folder / "labels").glob("*.txt")}
        assert images.keys() == labels.keys(), (folder, "unmatched files", images.keys() - labels.keys(), labels.keys() - images.keys())
        counts = Counter()
        for stem, img in sorted(images.items()):
            with Image.open(img) as im:
                im.load()
                digest = hashlib.sha256(str(im.size).encode() + im.convert("RGB").tobytes()).hexdigest()
            rows = []
            for line in labels[stem].read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5, (labels[stem], line)
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                assert 0 <= cls < len(names) and all(math.isfinite(v) for v in (x, y, w, h)), (img, line)
                assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1, (img, line)
                assert x-w/2 >= -0.001 and y-h/2 >= -0.001 and x+w/2 <= 1.001 and y+h/2 <= 1.001, (img, line)
                rows.append(" ".join(parts))
                counts[NAMES[cls]] += 1
            if digest in seen and sorted(rows) == sorted(seen[digest]["rows"]):
                continue
            meta = provenance.get(img.name, {})
            rec = dict(source=folder.name, original_image=str(img), original_label=str(labels[stem]), image=f"{folder.name}__{img.name}", rows=rows, pixel_sha256=digest, group=meta.get("source_clip", f"{folder.name}/{stem}"), provenance=meta)
            if digest in seen:
                rec["conflicting_duplicate"] = True
                for previous in records:
                    if previous["pixel_sha256"] == digest:
                        previous["conflicting_duplicate"] = True
            records.append(rec)
            seen[digest] = rec
        summaries[folder.name] = dict(images=len(images), boxes=dict(counts))
    # Check exact decoded-image overlap with every previous raw export.
    old_hashes = set()
    prior_val_hashes = set()
    for img in (ROOT / "training/round3/datasets/helmet/images/val").iterdir():
        with Image.open(img) as im:
            prior_val_hashes.add(hashlib.sha256(str(im.size).encode() + im.convert("RGB").tobytes()).hexdigest())
    for old in ("train-data", "train-data2", "train-data3"):
        for img in (ROOT / old / "images").iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            with Image.open(img) as im:
                old_hashes.add(hashlib.sha256(str(im.size).encode() + im.convert("RGB").tobytes()).hexdigest())
    # Hold out complete video clips, stratified by export folder.
    val_groups = set()
    for source in summaries:
        groups = defaultdict(list)
        for r in records:
            if r["source"] == source and not r.get("conflicting_duplicate"):
                groups[r["group"]].append(r)
        eligible = sorted(g for g, rs in groups.items() if all(r["pixel_sha256"] not in old_hashes for r in rs))
        random.Random(42).shuffle(eligible)
        target, total = round(sum(map(len, groups.values())) * .2), 0
        for group in eligible:
            if total >= target:
                break
            val_groups.add(group)
            total += len(groups[group])
    assert not DEST.exists() and not (OUT / "datasets").exists(), "Output already exists; refusing overwrite"
    (DEST / "images").mkdir(parents=True)
    (DEST / "labels").mkdir()
    (DEST / "classes.txt").write_text("\n".join(NAMES) + "\n")
    (DEST / "notes.json").write_text(json.dumps({"categories": [dict(id=i, name=n) for i,n in enumerate(NAMES)], "info": {"sources": summaries}}, indent=2))
    for folder in SOURCE.iterdir():
        if folder.is_dir():
            meta_out = DEST / "source_metadata" / folder.name
            meta_out.mkdir(parents=True)
            for p in folder.iterdir():
                if p.is_file() and p.suffix in {".json", ".txt"}:
                    shutil.copy2(p, meta_out / p.name)
    splits = Counter()
    for r in records:
        r["split"] = "excluded" if r.get("conflicting_duplicate") or r["pixel_sha256"] in prior_val_hashes else ("val" if r["group"] in val_groups else "train")
        if r["pixel_sha256"] in prior_val_hashes:
            r["prior_validation_overlap"] = True
        splits[r["split"]] += 1
        shutil.copy2(r["original_image"], DEST / "images" / r["image"])
        (DEST / "labels" / (Path(r["image"]).stem + ".txt")).write_text("\n".join(r["rows"]) + "\n")
    for task, mapping, names in [("helmet", {0:0, 3:1}, ["with helmet", "without helmet"]), ("plate", {1:0}, ["license plate"])]:
        out = OUT / "datasets" / task
        for r in records:
            if r["split"] == "excluded":
                continue
            for sub in ("images", "labels"):
                (out / sub / r["split"]).mkdir(parents=True, exist_ok=True)
            shutil.copy2(DEST / "images" / r["image"], out / "images" / r["split"] / r["image"])
            rows = [str(mapping[int(row.split()[0])]) + " " + " ".join(row.split()[1:]) for row in r["rows"] if int(row.split()[0]) in mapping]
            (out / "labels" / r["split"] / (Path(r["image"]).stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
        (out / "dataset.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i,n in enumerate(names)))
    report = dict(sources=summaries, merged_images=len(records), duplicates_removed=sum(s["images"] for s in summaries.values())-len(records), prior_exact_overlap=sum(r["pixel_sha256"] in old_hashes for r in records), splits=dict(splits), validation_groups=sorted(val_groups), records=records)
    (DEST / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
