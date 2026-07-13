# Model Training — Fine-Tuning Round 1

**Date:** July 11, 2026
**Hardware:** NVIDIA GeForce RTX 4070 SUPER (12 GB), CUDA, Ultralytics 8.3.52, PyTorch 2.12
**Staged weights:** `training/runs/helmet-v1/weights/best.pt`, `training/runs/plate-v1/weights/best.pt`

## Why we trained

The baseline helmet and plate models (`models/helmet-yolov8n.pt`, `models/license-plate-yolo11n.pt`)
are generic pretrained weights from Hugging Face — they were never trained on Thai
traffic footage, our camera angles, or Thai near-square motorcycle plates. The
observable symptoms in the pipeline:

- Distant no-helmet heads missed (violation recall capped).
- Helmet / no-helmet label flicker between frames, which the track-level voting
  system has to spend effort suppressing.
- Loose plate boxes producing poor crops, which caps OCR accuracy downstream.

Fine-tuning the two specialist models on our own labeled frames is the
highest-leverage accuracy work available; the association logic, tracker, and
voting were left untouched.

## What we did

### Dataset

- **Source:** Label Studio YOLO export at `train-data/` — 111 labeled frames,
  5 classes (Helmet, License Plate, Motorcycle, No Helmet, No License Plate).
- **Prep:** `training/prepare_dataset.py` splits the export into two per-model
  datasets under `training/datasets/` with remapped class IDs and a shared
  80/20 train/val split (89 train / 22 val):
  - `helmet` — 182 "with helmet" + 99 "without helmet" boxes. Class names are
    deliberately `with helmet` / `without helmet` so the trained weights are a
    drop-in swap for the pipeline's label matching (no code changes).
  - `plate` — 202 "license plate" boxes.
  - Frames without a model's classes are kept as **background images** (empty
    label files) to teach the model what not to detect.
- **Unused labels (by design):** Motorcycle (COCO YOLO11s already handles it;
  the object model also supplies person/car/bus/truck, which our dataset lacks —
  this is why a single unified 5-class model cannot replace the 3-model
  structure). No License Plate (no pipeline feature for it yet).

### Training

Everything ran in the isolated `training/` folder — no changes to `backend/`,
`models/`, or config. Commands (see `training/README.md` for the full workflow):

```
python training/prepare_dataset.py --src train-data
yolo detect train model=models/helmet-yolov8n.pt data=training/datasets/helmet/dataset.yaml epochs=100 imgsz=640 batch=-1 workers=0 project=training/runs name=helmet-v1
yolo detect train model=models/license-plate-yolo11n.pt data=training/datasets/plate/dataset.yaml epochs=100 imgsz=640 batch=-1 workers=0 project=training/runs name=plate-v1
```

- 100 epochs each, imgsz 640, AutoBatch chose batch 51 (~6.7 GB VRAM), AMP on.
- Total wall time for both models: **~15 minutes**.

**Incident worth remembering:** the first run crashed mid-epoch with
`RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly` — a known
PyTorch-on-Windows multiprocessing failure with `workers=8` (the default).
**Fix: `workers=0`** (load data in the main process). On a dataset this small
the speed cost is negligible. Keep this flag for all future training runs on
this machine.

## Results (22 held-out validation images)

| Class | Boxes | Precision | Recall | mAP50 |
|---|---|---|---|---|
| with helmet | 36 | 0.81 | 0.69 | 0.82 |
| **without helmet** | 21 | **1.00** | **0.90** | **0.92** |
| helmet model overall | 57 | 0.91 | 0.80 | 0.87 |
| license plate | 44 | 0.93 | 0.85 | 0.96 |

Highlights:

- **Without-helmet: 100% precision, 90% recall** on held-out frames — the class
  that actually triggers violations is the strongest one.
- **Plate mAP50 0.96** — Thai near-square plates are now well learned.
- Losses fell smoothly with no divergence (see `training/runs/*/results.png`);
  prediction grids (`val_batch0_pred.jpg`) look clean against the label grids.

**Caveats:** 22 validation images is a small sample — treat these numbers as
encouraging, not definitive. The models learned *our footage*; conditions absent
from the dataset (night, rain, new angles) may not have improved. The metrics
say nothing about end-to-end violation detection — that's what the A/B below is for.

A slide deck version of this report lives at `presentation/training-deck.html`.

## What to do next

1. **End-to-end A/B (the promotion gate).** Build a small eval labels JSON
   (frame ranges of real violations in 2–3 clips, per
   `scripts/eval-labels.example.json`) and run `scripts/evaluate.py` twice —
   once with the current weights, once with the new ones via env vars only:

   ```powershell
   $env:HELMET_MODEL_PATH = "training/runs/helmet-v1/weights/best.pt"
   $env:PLATE_MODEL_PATH  = "training/runs/plate-v1/weights/best.pt"
   python scripts/evaluate.py <labels.json> --json training/eval-new.json
   ```

   Compare event-level precision/recall, duplicate rate, plate capture rate,
   and OCR accuracy.

2. **Promote only if better.** Copy the winning `best.pt` files into `models/`
   under versioned names and update the two paths in
   `backend/app/core/config.py` (or set them in `backend/.env`). If the new
   weights don't win end-to-end, keep the old ones and label more data instead.

3. **Re-tune confidence thresholds.** `HELMET_CONFIDENCE` and
   `PLATE_CONFIDENCE` were tuned for the old models; a fine-tuned model is
   typically more confident on in-domain footage, so re-check these during the
   A/B run.

4. **Round 2 of labeling.** Run the new weights on fresh footage, collect the
   frames they still fail on (especially night footage, rain, and the
   with-helmet class, which is the current weak spot at 0.69 recall), label
   50–100 of them, re-export, and rerun the same workflow. Gains compound
   across rounds — this loop matters more than any hyperparameter.

5. **Future idea:** the No License Plate labels could power a "rider with no
   plate" violation type — a pipeline feature, not a training task.
