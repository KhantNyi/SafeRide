# SafeRide Model Training (isolated)

Everything training-related lives in this folder: dataset prep, generated
datasets, and run outputs. Nothing here modifies the main project — the
backend, `models/`, `scripts/`, and config stay untouched. Generated
datasets, runs, and weights are git-ignored via `training/.gitignore`.

## Workflow

**0. Prerequisite** — the Label Studio export at
`C:/Users/ADMIN/Downloads/train-data` must have its `images/` folder
populated (re-export from Label Studio with images included; filenames must
match the label files).

**1. Build the per-model datasets** (from the repo root):

```
python training/prepare_dataset.py --src "C:/Users/ADMIN/Downloads/train-data"
```

Creates `training/datasets/helmet` (with helmet / without helmet) and
`training/datasets/plate` (license plate), 80/20 split, remapped class IDs.

**2. Train** (outputs go to `training/runs/`, not the default `runs/`):

```
yolo detect train model=models/helmet-yolov8n.pt data=training/datasets/helmet/dataset.yaml epochs=100 imgsz=640 batch=-1 project=training/runs name=helmet-v1
yolo detect train model=models/license-plate-yolo11n.pt data=training/datasets/plate/dataset.yaml epochs=100 imgsz=640 batch=-1 project=training/runs name=plate-v1
```

Best weights land at `training/runs/helmet-v1/weights/best.pt` and
`training/runs/plate-v1/weights/best.pt`.

**3. A/B test without changing any project files** — the backend reads model
paths from environment variables (pydantic settings), so point a test run at
the new weights via env vars only:

```powershell
$env:HELMET_MODEL_PATH = "training/runs/helmet-v1/weights/best.pt"
$env:PLATE_MODEL_PATH  = "training/runs/plate-v1/weights/best.pt"
python scripts/evaluate.py <labels.json> --json training/eval-new.json
```

Run `evaluate.py` once *without* the env vars first (baseline), then with
them, and compare event-level precision/recall, plate capture rate, and OCR
accuracy.

**4. Promote only if better** — copy the winning `best.pt` files into
`models/` under new names and update the two paths in
`backend/app/core/config.py` (or set the env vars in `backend/.env`). Until
you do this, the running app is completely unaffected by anything in this
folder.

## Notes

- Class names in the helmet dataset are deliberately `with helmet` /
  `without helmet` so the pipeline's label matching works without code
  changes.
- Motorcycle and No License Plate labels from the export are intentionally
  unused (see models/README.md discussion — motorcycle detection stays on
  COCO yolo11s; "no plate" has no pipeline feature yet).
- Retraining round two: label the frames the new model still fails on,
  re-export, rerun step 1 (it rebuilds datasets from scratch each time).
