# SafeRide round 6: street-level continuation from v5

## Outcome

Both specialist detectors were fine-tuned directly from v5 and exported as v6. Earlier checkpoints and source datasets were preserved. The training script left application configuration unchanged at completion. On resuming the final checks, the current configuration already selected both v6 specialists; it was preserved.

**This is the requested reuse experiment, not an independent generalization test.** Former street-level validation examples now contribute to training. The monitoring clips are excluded from round-6 gradient updates, but were seen by earlier models. Historical validation metrics are now overlapping/training diagnostics, not holdout scores.

**Verdict:** keep v6 as an experimental checkpoint. The familiar-video checks still produce the pink-helmet false violation, and IMG_6665 gains a duplicate record with the wrong motorcycle's plate. Higher diagnostic mAP does not establish a safer end-to-end replacement. Both training jobs had already finished in the background before the user offered to skip plate fine-tuning; no additional plate training was started on resumption.

## Dataset

- Inputs: street-level `train-data3`, `train-data4`, and `train-data5`, visually checked using contact sheets.
- Excluded: all 325 overhead-view images in `train-data` and `train-data2`. The inherited v5 weights still contain earlier learning; this filter applies to round-6 input data.
- Merged output: `train-data6`, 290 unique image-label pairs.
- Removed duplicate copies: 74. Entries excluded for unresolved conflicting labels: 52.
- Newer reviewed crops take precedence over their older full-frame copies. IMG_7439 and IMG_7278 remain excluded from helmet training because their headwear labels are ambiguous; their plate labels may still train the plate detector.
- Both helmets in `IMG_6712_f000300` are class Helmet and now belong to training.
- Added usable former validation examples to existing street-level training data; did not train exclusively on old validation.
- Monitoring clips (previously training): IMG_7080, IMG_7082, IMG_7086, IMG_7094.
- Entire known clips remain together; exact pixel hashes and capture IDs do not cross the round-6 split. Monitoring footage shares the street/capture domain and is not a fresh recording session.
- Person/motorcycle detection and OCR were not trained. Sources include previously model-assisted, AI-reviewed annotations; no independent human label audit is claimed.

| Detector | Training images | Former-validation images in training | Monitoring images | Training boxes |
|---|---:|---:|---:|---|
| helmet | 268 | 56 | 20 | without helmet: 186, with helmet: 190 |
| plate | 270 | 56 | 20 | license plate: 266 |

## Training

- Initialization: `training/round5/models/helmet-v5-best.pt` and `license-plate-v5-best.pt`.
- 100-epoch cap, patience 25, image size 960, batch 8, CUDA device 0, workers 0, seed 0, deterministic mode, AMP, AdamW, initial learning rate 0.0005, final multiplier 0.01, mosaic probability 0.5, scale augmentation 0.35, close mosaic for last 10 scheduled epochs.
- Resized training images cached in RAM to avoid repeatedly decoding large PNGs. An initial uncached attempt was stopped early and preserved under `analysis/round6-audit/uncached-initial-*`; the final run restarted from the same v5 checkpoints. RAM caching can limit strict determinism.
- Training size changed from 640 in round 5 to 960 to match application inference. Both versions are evaluated at 960 here; these numbers are not directly comparable to the older 640-pixel report.
- Training/evaluation wall time: 18.3 minutes.
- helmet: 53 epochs completed; best and last checkpoints retained.
- plate: 44 epochs completed; best and last checkpoints retained.

## Matched diagnostic metrics

**No row below is an independent unseen-test accuracy estimate.**

| Detector | Dataset | Version | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| helmet | monitor_seen_by_base | v5 | 0.7538 | 0.8519 | 0.8817 | 0.4928 |
| helmet | former_round3_val_NOW_TRAINING | v5 | 0.6926 | 0.8007 | 0.8158 | 0.3618 |
| helmet | former_round4_val_NOW_TRAINING | v5 | 0.6913 | 0.8184 | 0.8401 | 0.3969 |
| helmet | former_round5_val_NOW_TRAINING | v5 | 0.8097 | 0.7167 | 0.7881 | 0.5079 |
| helmet | monitor_seen_by_base | v6 | 0.9934 | 0.9202 | 0.9937 | 0.6686 |
| helmet | former_round3_val_NOW_TRAINING | v6 | 0.8039 | 0.9161 | 0.9344 | 0.5262 |
| helmet | former_round4_val_NOW_TRAINING | v6 | 0.9314 | 0.9391 | 0.9636 | 0.5973 |
| helmet | former_round5_val_NOW_TRAINING | v6 | 0.9886 | 1.0000 | 0.9950 | 0.9114 |
| plate | monitor_seen_by_base | v5 | 0.9266 | 1.0000 | 0.9900 | 0.7030 |
| plate | former_round3_val_NOW_TRAINING | v5 | 0.6081 | 0.6471 | 0.6207 | 0.1485 |
| plate | former_round4_val_NOW_TRAINING | v5 | 0.9949 | 0.8621 | 0.9278 | 0.4759 |
| plate | former_round5_val_NOW_TRAINING | v5 | 0.9927 | 1.0000 | 0.9950 | 0.8345 |
| plate | monitor_seen_by_base | v6 | 0.9236 | 1.0000 | 0.9900 | 0.8871 |
| plate | former_round3_val_NOW_TRAINING | v6 | 0.9798 | 1.0000 | 0.9950 | 0.3909 |
| plate | former_round4_val_NOW_TRAINING | v6 | 0.9626 | 1.0000 | 0.9930 | 0.7275 |
| plate | former_round5_val_NOW_TRAINING | v6 | 0.9963 | 1.0000 | 0.9950 | 0.9306 |

## Familiar-video diagnostic

Full-frame inference, 0.5-second base interval, adaptive sampling on, OCR disabled. Existing tracking/association code is identical for both versions. These videos now overlap training; record counts alone do not establish correctness. Evidence and detection metadata are preserved.

| Video | Version | Records | Analyzed frames | Status |
|---|---|---:|---:|---|
| IMG_6712.MOV | v5 | 2 | 50 | completed |
| IMG_6665.MOV | v5 | 2 | 60 | completed |
| IMG_6805.MOV | v5 | 3 | 75 | completed |
| IMG_6712.MOV | v6 | 2 | 50 | completed |
| IMG_6665.MOV | v6 | 3 | 60 | completed |
| IMG_6805.MOV | v6 | 3 | 75 | completed |

Visual review: the v6 IMG_6712 record at frame 330 flags the pink-helmet passenger. In IMG_6665, v6 frames 522 and 618 flag the same first motorcycle twice; frame 522 saves its plate 3399, while frame 618 saves plate 3872 from the helmeted motorcycle behind it. V5 produces two records in that comparison. IMG_6805 keeps three records for both versions. These observations do not constitute exhaustive ground-truth scoring.

### Pink helmet: fixed-frame check

The additional near-view training example did **not** resolve the distant pink-helmet error. Both versions recognize the helmet at frame 300 but predict no helmet at frames 318, 342 and 366. This test predicts on full frames at 960 pixels and confidence 0.35. It isolates detector behavior; OCR and tracking are not involved. Approximate head regions come from the inspected earlier-run metadata.

| Frame | Version | Pink-head detections |
|---:|---|---|
| 300 | v5 | with helmet (66.2%) |
| 318 | v5 | without helmet (38.3%) |
| 342 | v5 | without helmet (51.2%) |
| 366 | v5 | without helmet (44.5%) |
| 300 | v6 | with helmet (51.8%) |
| 318 | v6 | without helmet (37.9%) |
| 342 | v6 | without helmet (50.1%) |
| 366 | v6 | without helmet (49.8%) |

## Artifacts and preservation

- `training/round6/models/helmet-v6-best.pt`
- `training/round6/models/license-plate-v6-best.pt`
- Matching `*-v6-last.pt` exports and complete training run checkpoints are also retained.
- At training completion, verified unchanged: 30 previous weight files, 832 source image/label files, and `.env`.
- Resumption checks again verified the earlier weights and v6 export hashes. Configuration had changed since training and now selects v6; the resumption did not overwrite it.
- `training/round6/data-audit.json`: per-image provenance, split, exclusions and duplicate decisions.
- `training/round6/metrics.json`, `artifact-checks.json`, `COMPLETE.json`: metrics and SHA-256 checks.
- `training/round6/training.log`, `runs/`, `evaluation/`: logs, history, plots and checkpoints.
- `training/round6/video-checks/results.json`: diagnostic jobs and evidence paths.
- `training/round6/pink-helmet-check/`: fixed-frame predictions and images.

## Reproduction

Scripts refuse to overwrite datasets or runs. From a fresh output location:

```powershell
.venv/Scripts/python.exe training/prepare_round6.py
.venv/Scripts/python.exe training/train_round6.py
.venv/Scripts/python.exe training/check_round6_videos.py
.venv/Scripts/python.exe training/check_round6_helmet_views.py
.venv/Scripts/python.exe training/report_round6.py
```
