# Model Training — Fine-Tuning Round 2

**Date:** July 15, 2026
**Hardware:** NVIDIA GeForce RTX 4070 SUPER (12 GB), CUDA, Ultralytics 8.3.52, PyTorch 2.12
**Staged weights:** `training/runs/helmet-v2/weights/best.pt`, `training/runs/plate-v2/weights/best.pt`
**Round 1 report:** [model-training.md](model-training.md)

## Why round 2

Round 1 proved fine-tuning works but used only 111 labeled frames, with a
22-image validation set too small to trust. A second Label Studio export
(`train-data2/`, 214 new frames — filenames confirmed disjoint from round 1's
`train-data/`) nearly tripled the labeled data, so both specialist models were
retrained from the same baseline weights on the combined set.

## What we did

### Dataset

- **Sources:** `train-data/` (111 frames) + `train-data2/` (214 frames) =
  **325 frames**, same 5-class Label Studio YOLO export format.
- `training/prepare_dataset.py` now accepts **multiple `--src` folders** and
  fails loudly on duplicate filenames across exports:

  ```
  python training/prepare_dataset.py --src train-data train-data2
  ```

- Combined split: **260 train / 65 val** (same seeded 80/20, shared across both
  per-model datasets):
  - `helmet` — 547 "with helmet" + 246 "without helmet" boxes, 4 background images.
  - `plate` — 587 "license plate" boxes, 5 background images.

### Training

Same recipe as round 1 — retrained **from the original baseline weights**
(`models/helmet-yolov8n.pt`, `models/license-plate-yolo11n.pt`), not from the
v1 checkpoints, to avoid compounding any round-1 overfit:

```
yolo detect train model=models/helmet-yolov8n.pt data=training/datasets/helmet/dataset.yaml epochs=100 imgsz=640 batch=-1 project=training/runs name=helmet-v2
yolo detect train model=models/license-plate-yolo11n.pt data=training/datasets/plate/dataset.yaml epochs=100 imgsz=640 batch=-1 project=training/runs name=plate-v2
```

- 100 epochs each, imgsz 640, AutoBatch, AMP on. Wall time: **~6.5 min (helmet)
  + ~7.5 min (plate)**.
- Round 1's `DataLoader worker exited unexpectedly` crash did **not** recur even
  though these runs used the default workers — treat `workers=0` as the fallback
  if it ever reappears, not a mandatory flag.
- **New Windows incident:** a custom eval script calling `model.val()` without an
  `if __name__ == "__main__":` guard silently re-spawned itself in a loop
  (Windows `spawn` multiprocessing re-imports the main module). Any script using
  ultralytics on this machine needs the main guard.

## Results

### v2 on its own validation set (65 held-out images)

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| with helmet | 0.83 | 0.84 | 0.84 |
| without helmet | 0.86 | 0.80 | 0.89 |
| helmet model overall | 0.85 | 0.82 | 0.87 |
| license plate | 0.93 | 0.83 | 0.94 |

Round 1 vs round 2 headline numbers look similar, but they are **not
comparable**: the round-2 val set is 3× larger (65 vs 22 images) and contains
the new footage, so these numbers are far more trustworthy.

### Production baseline vs v2, same data (the finding that matters)

Both the currently deployed models and the v2 weights were validated on the
identical dataset. Fair comparison is the 65-image held-out split — neither
model trained on those frames:

| Model | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet **production** | 0.52 | 0.46 | 0.39 | 0.15 |
| helmet **v2** | **0.84** | **0.82** | **0.86** | **0.38** |
| plate **production** | 0.58 | 0.70 | 0.56 | 0.26 |
| plate **v2** | **0.93** | **0.83** | **0.94** | **0.50** |

- **Helmet quality more than doubles** (mAP50 0.39 → 0.86). Production is
  weakest on "Without Helmet" (AP50 0.33 → 0.89) — the class that triggers
  violations.
- **Plate mAP50 0.56 → 0.94**, precision 0.58 → 0.93: far fewer junk plate
  crops entering OCR.
- On all 325 frames the gap widens further (helmet 0.41 → 0.95, plate
  0.62 → 0.98 mAP50), but v2 trained on 260 of those frames — upper bound only.

**Caveats:** the val set is drawn from our own footage — that's the deployment
distribution, so favoring the fine-tuned model is the point, but conditions not
in the dataset (night, rain, new angles) remain unproven. Detector metrics
still say nothing about end-to-end violation events, duplicates, or OCR
accuracy.

## What to do next

Unchanged from round 1 — the promotion gate is the end-to-end A/B:

1. Run `scripts/evaluate.py` with baseline weights, then with
   `HELMET_MODEL_PATH` / `PLATE_MODEL_PATH` pointed at the v2 `best.pt` files;
   compare event-level precision/recall, plate capture rate, OCR accuracy.
2. Promote by copying the v2 weights into `models/` under versioned names and
   updating the two paths in `backend/app/core/config.py` (or `backend/.env`);
   restart the backend (models load once at startup). Rollback = revert the paths.
3. Re-check `HELMET_CONFIDENCE` / `PLATE_CONFIDENCE` — at 2× detector mAP the
   confidence distributions have shifted.
4. Round 3 labeling: night, rain, and new angles are still the blind spots.
