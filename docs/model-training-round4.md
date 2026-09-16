# SafeRide v4 training report

## Outcome

Both v3 specialist detectors were fine-tuned and exported. Plate v4 improves the new-data metrics and retains previous-set performance. Helmet v4 improves the new data but regresses on the previous validation set; it is not an unconditional upgrade. Application model settings were not changed.

## Data audit

- Sources: train-5 (86), train-6 (68), train-video-data (73): 227 image-label pairs merged into `train-data4/`.
- Class IDs were compatible; names were normalized. The sixth Person class is preserved but not trained by the helmet/plate detectors.
- 86 images already occur in previous raw exports. All new validation images are free of exact decoded-pixel overlap with those exports.
- 26 entries have conflicting annotations for identical pixels; 22 overlap previous v3 validation pixels. Their union is 38 excluded entries (10 overlap both categories). All remain preserved in the merged raw export.
- Final split: 159 train / 30 validation. Video frames are grouped by source clip. Still exports lack clip provenance, so unknown near-duplicate scenes remain a limitation.
- All pairs passed decoding, class, finite coordinate, positive box size, and normalized boundary checks. No identical image or known clip appears across the new train/validation split.
- Box counts: training has 107 helmet / 97 no-helmet / 143 plate; validation has 20 helmet / 19 no-helmet / 29 plate.
- Boxed preview images are excluded. Original metadata and per-image provenance are preserved in `train-data4/source_metadata/` and `train-data4/manifest.json`.

## Training

- Initialized directly from `training/round3/models/helmet-v3-best.pt` and `license-plate-v3-best.pt`.
- Fine-tuned on the selected merged data; no other old datasets were appended. Existing overlap supplies some old examples.
- RTX 4070 SUPER 12 GB; PyTorch 2.12.0+cu130; Ultralytics 8.3.52.
- 100 epoch limit, patience 30, image size 640, batch 16, workers 0, seed 0, deterministic mode, automatic optimizer.
- Helmet: early stopped at epoch 41; best checkpoint epoch 11. Plate: early stopped at epoch 55; best checkpoint epoch 25.

## Matched validation results

| Dataset | Detector | Version | Precision | Recall | mAP50 | mAP50?95 |
|---|---|---|---:|---:|---:|---:|
| new_holdout | helmet | v3 | 0.7887 | 0.5632 | 0.6427 | 0.2198 |
| new_holdout | helmet | v4 | 0.8828 | 0.7145 | 0.8453 | 0.4515 |
| new_holdout | plate | v3 | 0.8560 | 0.6897 | 0.6471 | 0.1661 |
| new_holdout | plate | v4 | 0.9849 | 0.8621 | 0.9542 | 0.6927 |
| round3_holdout | helmet | v3 | 0.8614 | 0.8238 | 0.8139 | 0.4134 |
| round3_holdout | helmet | v4 | 0.8048 | 0.6619 | 0.7283 | 0.3159 |
| round3_holdout | plate | v3 | 0.9381 | 1.0000 | 0.9922 | 0.4125 |
| round3_holdout | plate | v4 | 0.9427 | 1.0000 | 0.9922 | 0.4331 |

The new holdout has 30 images; the round3 holdout has 17. The new set selected the best checkpoints, so these are validation results, not an independent test. The previous set was used as a regression check, and overlapping pixels were excluded from v4 training. Results do not measure tracking, violation-event accuracy, or OCR. Additional video annotations are model-assisted and visually reviewed according to their source metadata.

## Artifacts

- `training/round4/models/helmet-v4-best.pt`
- `training/round4/models/license-plate-v4-best.pt`
- `training/round4/metrics.json`: full matched metrics.
- `training/round4/artifact-checks.json`: SHA-256 checks, class mappings, and exported-weight inference checks.
- `training/round4/runs/`: best/last checkpoints, training arguments, CSV histories, and plots.
- `training/round4/evaluation/`: predictions, confusion matrices, and precision-recall plots.

Both exported files match the respective best checkpoints byte-for-byte and successfully load and predict on a held-out image.

## Reproduction

From the project root, using fresh output directories:

```powershell
.\.venv\Scripts\python.exe training\prepare_round4.py
.\.venv\Scripts\python.exe training/train_round4.py
```

Preparation deliberately refuses to overwrite the existing merged export or generated datasets. The scripts use the original Downloads source path. Training logs include a nonfatal package version-check network error; training and both final evaluations completed. Future runs disable that unnecessary version check.
