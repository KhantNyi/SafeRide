# SafeRide v5 training report

## Outcome

Both specialist detectors were fine-tuned from v4 and exported as v5. Helmet detection and plate-box localization improved on the new footage, but both detectors regressed in mAP50 on the older validation sets, especially the plate detector. V5 is not a general replacement for v4 based on these results. The app remains configured to v4; its PaddleOCR selection is unchanged.

## Data preparation

- Inputs: `C:/Users/ADMIN/Downloads/train-video-footage` (53 pairs) and `C:/Users/ADMIN/Downloads/train-data5` (54 pairs). Original folders were preserved.
- Merged destination: `train-data5/`, with 103 unique pairs and 480 boxes (73 Helmet, 108 License Plate, 116 Motorcycle, 88 No Helmet, 9 No License Plate, 86 Person).
- Four duplicate copies removed. For IMG_7438, the annotation containing the visible plate was retained over the conflicting No License Plate annotation. IMG_7439 duplicates were consolidated; its ambiguous headwear is excluded from helmet training. The redundant IMG_7440 annotation was also removed.
- Fourteen EXIF-rotated photos normalized to upright pixels. Fifty retained still images cropped around all labeled objects with context; boxes transformed to match. PNG output avoids further lossy compression. Footage crops remain as previously reviewed.
- All pairs passed decoding, finite-coordinate, positive-size, class-ID and boundary checks. All 54 new images were visually inspected using contact sheets, with closer inspection of duplicate/headwear cases. All 53 footage file hashes match the earlier reviewed export.
- Fourteen inputs overlap earlier raw exports exactly after orientation normalization. IMG_7272 overlaps an earlier validation capture and is excluded from both specialist datasets.
- IMG_7439 and IMG_7278 have ambiguous fabric-covered heads and are additionally excluded from the helmet dataset. Their plate labels remain usable.
- Helmet dataset: 86 train images (61 Helmet, 73 No Helmet boxes), 14 validation images (10 Helmet, 12 No Helmet boxes).
- Plate dataset: 88 train images (94 plate boxes), 14 validation images (13 plate boxes).
- Validation clips: IMG_6712, IMG_6774, IMG_6779, IMG_6788, IMG_6805. Whole clips stay together. All new still images from the same road/capture session stay in training. No exact decoded image or known capture ID crosses the new split.
- The six raw class IDs are preserved, but only Helmet/No Helmet and License Plate are remapped into the two specialists. The new still export lacks Person annotations; this is not a fully labeled Person dataset. General object detection and OCR were not trained.

## Training

- Initialized from `training/round4/models/helmet-v4-best.pt` and `training/round4/models/license-plate-v4-best.pt`.
- RTX 4070 SUPER 12 GB; PyTorch 2.12.0+cu130; Ultralytics 8.3.52.
- 100 epoch limit, patience 30, image size 640, batch 16, workers 0, seed 0, deterministic mode, mixed precision. AdamW, initial learning rate 0.001, final multiplier 0.01.
- Fine-tuned on the prepared merged set; no separate historical replay dataset was appended.
- Helmet: 67 epochs completed; best epoch by logged validation fitness: 37.
- Plate: 80 epochs completed; best epoch by logged validation fitness: 50.

## Matched validation results

| Set | Detector | Version | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| new_holdout | helmet | v4 | 0.9288 | 0.6207 | 0.6895 | 0.5023 |
| new_holdout | helmet | v5 | 0.9003 | 0.7741 | 0.8887 | 0.6211 |
| new_holdout | plate | v4 | 0.9956 | 1.0000 | 0.9950 | 0.5471 |
| new_holdout | plate | v5 | 0.9663 | 1.0000 | 0.9950 | 0.8692 |
| round4_holdout | helmet | v4 | 0.8828 | 0.7145 | 0.8453 | 0.4515 |
| round4_holdout | helmet | v5 | 0.7519 | 0.7895 | 0.8342 | 0.3918 |
| round4_holdout | plate | v4 | 0.9849 | 0.8621 | 0.9542 | 0.6927 |
| round4_holdout | plate | v5 | 1.0000 | 0.8252 | 0.8621 | 0.3944 |
| round3_holdout | helmet | v4 | 0.8048 | 0.6619 | 0.7283 | 0.3159 |
| round3_holdout | helmet | v5 | 0.7292 | 0.6329 | 0.6802 | 0.3446 |
| round3_holdout | plate | v4 | 0.9427 | 1.0000 | 0.9922 | 0.4331 |
| round3_holdout | plate | v5 | 0.8176 | 0.8235 | 0.8170 | 0.2629 |

## Interpretation and limits

- Helmet mAP50: new_holdout: 68.95% to 88.87% (+19.92 percentage points); round4_holdout: 84.53% to 83.42% (-1.10 percentage points); round3_holdout: 72.83% to 68.02% (-4.82 percentage points).
- Plate mAP50: new_holdout: 99.50% to 99.50% (+0.00 percentage points); round4_holdout: 95.42% to 86.21% (-9.21 percentage points); round3_holdout: 99.22% to 81.70% (-17.52 percentage points).
- The new holdout contains only 14 images from five clips and selects the best checkpoint. It is validation, not an independent test. Older checks contain 30 and 17 images. Known old-validation overlaps were excluded from new training.
- Footage labels are model-assisted and AI-visually-reviewed, not independently human-verified. Unknown similar scenes/captures remain a limitation of still-photo provenance.
- The Sp vd collection was used in earlier benchmarking and now partly in training. It must not be presented as an independent v5 test. These metrics do not measure tracking, violation-event accuracy or OCR character recognition.

## Artifacts and preservation

- SHA-256 verification confirms all 24 earlier checkpoint files under `models/` and `training/` are unchanged.
- Source image and label hashes were verified unchanged. Removed duplicates remain in the source folder.
- `training/round5/models/helmet-v5-best.pt`: `e3f75fdd5899c4c4e8584f036f1d0f9f0e2ca5326795020e25f7ddc8af1c74f2`. Matches its best checkpoint and successfully loads and predicts on a validation image.
- `training/round5/models/license-plate-v5-best.pt`: `c0b8852a7a9eea4b5de3207d12b18fb606050ec5d65cc157dd6e4800674babf9`. Matches its best checkpoint and successfully loads and predicts on a validation image.
- `training/round5/metrics.json`, `COMPLETE.json`: final results.
- `training/round5/data-audit.json`, `train-data5/manifest.json`: audit, provenance and split decisions.
- `training/round5/runs/`: best/last checkpoints, arguments, CSV history and plots.
- `training/round5/evaluation/`: matched evaluations and diagnostic plots.
- `training/round5/training.log`: complete training log.
- The PowerShell log marks redirected Ultralytics stderr output as `NativeCommandError`. Both training runs, final evaluations, export/inference checks and the completion marker finished successfully; no Python traceback occurred.
- `analysis/round5-audit/`: visual-review sheets and headwear close-ups.

## Reproduction

Using fresh output directories (scripts refuse to overwrite existing runs):

```powershell
.\.venv\Scripts\python.exe training\prepare_round5.py
.\.venv\Scripts\python.exe training\train_round5.py
.\.venv\Scripts\python.exe training\report_round5.py
```
