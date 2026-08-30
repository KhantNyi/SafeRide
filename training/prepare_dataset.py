"""Build per-model training datasets from the Label Studio YOLO export.

Everything is written inside training/datasets/ by default (or the directory
passed via --output-dir) - nothing in the main project is read or modified
except the export folder you point --src at.

The export has 5 classes (classes.txt):
    0 Helmet, 1 License Plate, 2 Motorcycle, 3 No Helmet, 4 No License Plate

Two datasets are produced, matching the two models worth fine-tuning:

  datasets/helmet/  - classes: 0 "with helmet" (from Helmet),
                               1 "without helmet" (from No Helmet)
                      Class names match what the pipeline expects
                      (pipeline.py WITH_HELMET_LABEL / NO_HELMET_LABEL),
                      so trained weights are a drop-in swap.
  datasets/plate/   - class:   0 "license plate" (from License Plate)

Every image appears in both datasets; images without that model's classes
get an empty label file and act as background (teaches the model what NOT
to detect). The train/val split is decided once per image so both datasets
hold out the same frames.

Usage (one or more export folders; filenames must be unique across them):
    python training/prepare_dataset.py --src train-data train-data2
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAINING_DIR = Path(__file__).resolve().parent

# subset name -> (source class id -> (new class id, new class name))
SUBSETS = {
    "helmet": {0: (0, "with helmet"), 3: (1, "without helmet")},
    "plate": {1: (0, "license plate")},
}


def remap_label_lines(label_path: Path, mapping: dict) -> list[str]:
    lines = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        source_class = int(parts[0])
        if source_class not in mapping:
            continue
        new_class = mapping[source_class][0]
        lines.append(" ".join([str(new_class)] + parts[1:]))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="Label Studio export dir(s) (images/, labels/, classes.txt)")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=TRAINING_DIR / "datasets",
        help="Destination for the helmet/ and plate/ datasets (default: training/datasets)",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for src_dir in args.src:
        src = Path(src_dir)
        for p in (src / "images").iterdir():
            if p.suffix.lower() in IMG_EXTS:
                if p.stem in images:
                    raise SystemExit(f"Duplicate image stem across exports: {p.stem} ({images[p.stem]} vs {p})")
                images[p.stem] = p
        for p in (src / "labels").glob("*.txt"):
            if p.stem in labels:
                raise SystemExit(f"Duplicate label stem across exports: {p.stem} ({labels[p.stem]} vs {p})")
            labels[p.stem] = p

    paired = sorted(set(images) & set(labels))
    missing_img = sorted(set(labels) - set(images))
    if missing_img:
        print(f"WARNING: {len(missing_img)} labels have no matching image (skipped), e.g. {missing_img[:3]}")
    if not paired:
        raise SystemExit("No image/label pairs found - is the export's images folder populated?")

    # One split shared by every subset so val frames are held out consistently.
    random.Random(args.seed).shuffle(paired)
    n_val = max(1, round(len(paired) * args.val_ratio))
    splits = {"val": paired[:n_val], "train": paired[n_val:]}

    for subset, mapping in SUBSETS.items():
        out = args.output_dir.resolve() / subset
        if out.exists():
            shutil.rmtree(out)
        object_counts = {name: 0 for _, name in mapping.values()}
        background_count = 0

        for split, stems in splits.items():
            img_dir = out / "images" / split
            lbl_dir = out / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                lines = remap_label_lines(labels[stem], mapping)
                shutil.copy2(images[stem], img_dir / images[stem].name)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                if lines:
                    for line in lines:
                        new_id = int(line.split()[0])
                        name = next(n for i, n in mapping.values() if i == new_id)
                        object_counts[name] += 1
                else:
                    background_count += 1

        names = "\n".join(
            f"  {new_id}: {name}" for new_id, name in sorted(mapping.values())
        )
        (out / "dataset.yaml").write_text(
            f"path: {out.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n{names}\n",
            encoding="utf-8",
        )
        counts = ", ".join(f"{count} x {name}" for name, count in object_counts.items())
        print(f"[{subset}] {len(splits['train'])} train / {len(splits['val'])} val images | {counts} | {background_count} background images")
        print(f"[{subset}] config: {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
