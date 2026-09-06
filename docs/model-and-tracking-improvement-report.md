# SafeRide improvement report: training rounds 1–3, sampling, and tracking

Report date: 6 September 2026. This report consolidates existing training artifacts and September video evaluations; no new inference benchmark was run to prepare it.

## 1. Main findings

Fine-tuning for street-level footage produced the largest measured recall improvement. On the same 20 September clips, changing from v2 to v3 at a 1-second sampling interval increased detected motorcycle violation events from **8/25 (32%) to 19/25 (76%)**. Subsequent tracking refinement increased that to **21/25 (84%)**.

At a 0.25-second interval, v2 detected **17/25 events (68%)**, the original v3 pipeline detected **21/25 (84%)**, and v3 with refined tracking detected **22/25 (88%)**. Tracking refinement reduced extra duplicate records at this interval from **7 to 1** while recovering one additional event.

These are event-level results on a small, repeatedly inspected development benchmark. They are not a universal accuracy percentage, an independent production acceptance test, or a measurement of license plate transcription accuracy.

## 2. Model versions and fine-tuning history

SafeRide uses separate detectors for general objects, helmet status, and license plates. The helmet and plate specialists were fine-tuned; motorcycle/person/vehicle context continues to come from the general object detector. The exported Motorcycle and No License Plate labels were not used to train those two specialists.

| Round | Output | Training source | Initialization | Training recipe |
|---|---|---|---|---|
| 1 | helmet-v1, plate-v1 | `train-data`: 111 labeled frames | Original pretrained helmet and plate models | 100 epochs, 640-pixel input, AutoBatch; documented Windows run used workers=0 |
| 2 | helmet-v2, plate-v2 | `train-data` + `train-data2`: 111 + 214 = 325 frames | Original pretrained models again, rather than v1 | 100 epochs, 640-pixel input, AutoBatch |
| 3 | helmet-v3, plate-v3 | `train-data3`: 86 street-level frames | Corresponding v2 best checkpoints | 100 epochs, 640-pixel input, batch 16, workers=0, GPU device 0, AMP |

Rounds 1 and 2 represent the earlier footage collection, described by the project owner as above-ground footage. Round 3 adapts those learned weights to the newer street-level viewpoint. The exact round-3 count is **86 images**, rather than the approximate “100 images” used in conversation. Both prepared round-3 datasets contain exactly the same image stems as `train-data3/images`.

Round 3 did **not** directly retrain on a combined 411-image dataset. It trained on 86 images while inheriting v2 weights previously trained on the round-2 training subset. A total of 411 source images across exports should not be described as 411 round-3 training images.

Round-3 logs contain 100 epochs for each model, with approximately 363.8 seconds for helmet training and 415.6 seconds for plate training (about 13 minutes combined). Earlier reports document an RTX 4070 SUPER with 12 GB VRAM; the September benchmark also used this GPU.

## 3. Train, validation, and test ratios

Here, **validation** means the held-out images used during training to evaluate checkpoints. A separate **test** split would be reserved for final assessment after model selection. The stored preparation script and dataset YAML files define train and validation sets only.

| Round | Total images | Train | Validation | Separate image test | Train / validation / test ratio |
|---|---:|---:|---:|---:|---|
| 1 | 111 | 89 | 22 | 0 | 80.18% / 19.82% / 0% |
| 2 | 325 | 260 | 65 | 0 | 80.00% / 20.00% / 0% |
| 3 | 86 | 69 | 17 | 0 | 80.23% / 19.77% / 0% |

The intended split is **80/20**, with rounding accounting for the slight differences. If expressed in the requested order **train / test / evaluation**, and “evaluation” means validation, these are approximately **80 / 0 / 20**, not 80/10/10 or 70/15/15.

The preparation script defaults to seed 42, splits at image level, and shares the split between helmet and plate datasets. Images without relevant boxes remain as background examples. The surviving round-3 folders verify the 69/17 counts; they do not by themselves prove which preparation command or seed was used.

The September evaluation is a separate **20-video pipeline benchmark**, not a percentage carved out of these training image splits. The clips are from a later supplied footage folder, but complete source-video independence from training has not been audited. Nearby frames from one source video can also appear in both sides of an image-level split. Future evaluation should split by source clip/session and reserve a final test collection that is not used for development.

## 4. Detector validation results

### Rounds 1 and 2

The existing training reports record the following rounded metrics on each round's own validation split:

| Round | Detector/class | Precision | Recall | mAP50 |
|---|---|---:|---:|---:|
| 1 | With helmet | 81% | 69% | 82% |
| 1 | Without helmet | 100% | 90% | 92% |
| 1 | Helmet model overall | 91% | 80% | 87% |
| 1 | License plate | 93% | 85% | 96% |
| 2 | With helmet | 83% | 84% | 84% |
| 2 | Without helmet | 86% | 80% | 89% |
| 2 | Helmet model overall | 85% | 82% | 87% |
| 2 | License plate | 93% | 83% | 94% |

These tables use **different validation images**: 22 in round 1 and 65 in round 2. They do not establish that v1 is better than v2 simply because some v1 percentages are higher.

A separate baseline-versus-v2 comparison in the round-2 report used the same 65-image validation split:

| Detector | Original pretrained mAP50 | v2 mAP50 | Change |
|---|---:|---:|---:|
| Helmet | 39% | 86% | +47 percentage points |
| Plate | 56% | 94% | +38 percentage points |

The separate v2 comparison reports helmet precision/recall of 84%/82%, versus 52%/46% for the original model; plate precision/recall improved from 58%/70% to 93%/83%. Small differences from the round-2 training table reflect separately reported validation results and should not be silently combined.

### Round 3

The following rows were extracted from the saved training CSVs at the epoch with the **highest logged mAP50–95** for each model. They are training-validation observations on 17 images, not a fresh evaluation of the exported weights or a common-set comparison against v2.

| Model | Selected CSV epoch | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Helmet v3, overall | 26 | 86.15% | 82.38% | 81.39% | 41.22% |
| Plate v3 | 46 | 93.81% | 100.00% | 99.22% | 43.87% |

mAP50 measures object detection at an overlap threshold of 0.5; mAP50–95 averages over stricter overlap thresholds as well. Neither is the proportion of real motorcycle violations caught by the application. The 100% plate recall here is based on a very small validation set and does not imply perfect real-world plate detection or OCR.

## 5. September video evaluation setup

- Source: `C:/Users/ADMIN/Downloads/Sp data 2 september`, clips IMG_7075 through IMG_7094.
- Size: **20 clips, 326.44 seconds total** (5 minutes 26 seconds); mean 16.32 seconds, range 3.70–41.57 seconds.
- Video: encoded 3840×2160 at approximately 30 FPS; orientation metadata produces portrait frames for analysis.
- Labels: **25 confirmed motorcycle-passage violation events**. A motorcycle with two unhelmeted occupants counts as one event. One ambiguous head-cloth case is excluded.
- A correction during v3 review added a gray-haired delivery rider previously mistaken for helmeted, changing the denominator from 24 to 25. All figures below use 25.
- Labels and record-to-event assignments were produced by AI visual review, using source contact sheets around 0.5-second spacing and targeted full-resolution frames. They were not independently human-validated or exhaustively annotated frame by frame.
- Matched v2/v3 benchmarks used full-frame helmet inference, object/helmet confidence 0.35, plate confidence 0.30, adaptive sampling, and CUDA/FP16. Both specialist models changed together.

Historical benchmark settings matter: the current code defaults include rider-crop helmet inference. These results should not be presented as a fresh benchmark of every current application setting.

### Sampling terminology

The tested values are **seconds between base samples**, not FPS:

| Sampling interval | Nominal base rate |
|---|---:|
| 1.0 second | 1 FPS |
| 0.5 second | 2 FPS |
| 0.25 second | 4 FPS |

A smaller interval means more frequent analysis. Adaptive sampling adds extra frames after recent no-helmet detections, so actual analyzed frame counts exceed the nominal base rate. Integer frame-step rounding also affects cadence.

### Metric definitions

- **Recall:** unique matched violation events / 25.
- **Event precision:** unique matched violation events / saved records. Additional records for an already matched event count against precision.
- **Duplicate records:** extra saved records for the same motorcycle passage.
- **False helmet alerts:** saved records matched to a nonviolating or unrelated motorcycle.

No ordinary “overall accuracy” is calculated because there is no defined count of true-negative motorcycle events. Zero false helmet alerts does not establish correct plate ownership or OCR text.

## 6. v2 versus v3 at different sampling intervals

OCR was disabled in this primary comparison. These results precede the subsequent tracking refinement.

| Interval | v2 found | v2 recall | v2 duplicates | v2 event precision | v3 found | v3 recall | v3 duplicates | v3 event precision | Recall gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0s | 8/25 | 32% | 0 | 100% | 19/25 | 76% | 1 | 95.0% | +44 pp |
| 0.5s | 13/25 | 52% | 0 | 100% | 20/25 | 80% | 4 | 83.3% | +28 pp |
| 0.25s | 17/25 | 68% | 2 | 89.5% | 21/25 | 84% | 7 | 75.0% | +16 pp |

No saved record in these runs was visually classified as a false helmet alert. Falling event precision at higher sampling was attributable to duplicates. V3 improved overall recall but did not retain every individual event that v2 detected; the saved comparison documents regressions for some passengers and one scooter rider.

### Runtime and OCR observations

Times below are total processing times for all 326.44 seconds of footage.

| Interval | OCR | v2 runtime | v3 runtime |
|---|---|---:|---:|
| 1.0s | Off | 120.4s | 157.3s |
| 0.5s | Off | 138.7s | 209.3s |
| 0.25s | Off | 335.7s | 578.0s |
| 0.5s | On | 145.6s | 216.4s |

For original v3, moving from 1s to 0.5s added one event and 52 seconds; moving from 0.5s to 0.25s added another event and 368.7 seconds. Detection-triggered adaptive sampling and downstream processing mean runtime does not scale solely with nominal FPS.

Enabling OCR at 0.5s preserved the same event counts: 13 for v2 and 20 for original v3. It added approximately 5.0% and 3.4% runtime respectively in these passes. V3 produced 17 plate crops and 12 nonempty OCR outputs among 24 records with OCR enabled. These counts include duplicate records and **are not OCR accuracy**. The v2 review found both transcription errors and a plate associated with the wrong motorcycle.

Timing runs were sequential single passes. Initial runs included cold model loading; later runs reused models. These are observations, not repeated controlled performance estimates or live-stream latency guarantees.

## 7. Tracking mechanism and refinements

### Existing tracking and confirmation

The pipeline tracks motorcycle boxes using a local ByteTrack-style tracker, including motion prediction with a Kalman filter that accounts for the actual frame gap between samples. Rider, head, and plate associations attach evidence to the motorcycle identity. Helmet votes accumulate across frames, and qualifying no-helmet evidence enters a pending event after the confirmation checks pass.

Plate candidates are buffered across the track, including before violation confirmation. Collection continues until its time window expires or the track ends; remaining pending events are finalized when the clip ends. Ranked plate crops are processed by OCR after collection, with repeated-sighting checks and text voting. This allows the clearest plate and no-helmet evidence to occur at different moments.

### Duplicate failure mode

The detector could produce both a whole-bike box and a nested rear-bike box for one physical motorcycle. Separate raw track IDs could then split the evidence and create multiple saved violation records. Track loss and reacquisition also created duplicate identities. More sampled frames exposed more opportunities for these failures.

### General refinement implemented

1. Alias a newly created overlapping raw track to an existing event identity when their motorcycle boxes overlap sufficiently and both associate with the same nearest person. The current overlap threshold is IoU ≥ 0.5.
2. Preserve both detector boxes for rider and plate association; unify their violation identity rather than deleting useful boxes.
3. Avoid merging already established tracks merely because their boxes overlap.
4. Keep a pending event active while any raw track aliased to that identity remains active.
5. Remember simultaneously observed separate motorcycles so later positional duplicate checks do not suppress a different rider using the same road position.

These are general association rules; production logic does not branch on September clip names or specific frame numbers. Regression tests include an observed duplicate-box example, alongside nearby distinct motorcycles, established overlapping tracks, alias lifetime, and separate passages through a previous rider's position.

### Full-clip rerun after refinement

All 20 clips were processed again with v3 at each interval, OCR disabled. These were fresh inference runs, not post-filtered original results.

| Interval | Events before → after | Duplicates before → after | Refined recall | Refined records | Refined event precision | Refined runtime |
|---|---|---|---:|---:|---:|---:|
| 1.0s | 19 → 21 | 1 → 1 | 84% | 22 | 95.5% | 113.9s |
| 0.5s | 20 → 21 | 4 → 1 | 84% | 22 | 95.5% | 182.6s |
| 0.25s | 21 → 22 | 7 → 1 | 88% | 23 | 95.7% | 470.4s |

Event precision in this table is calculated from the saved unique-event and record counts. There were zero visually identified false helmet alerts. No previously matched events were lost in this rerun. At 0.5s duplicate count fell 75%; at 0.25s it fell approximately 85.7%.

The 1s duplicate count stayed at one, but the duplicate's identity changed: suppressing one failure and protecting a previously suppressed distinct passage exposed another. One duplicate remained at each rate: IMG_7082 at 1s, IMG_7090 at 0.5s, and IMG_7080 at 0.25s. Tracking gaps remain a limitation.

The refined runs used a different execution order (0.5s, 0.25s, then 1s), so their lower runtimes do not establish a controlled tracking speedup. On these clips, refined 1s and 0.5s found the same number of events; 0.25s found one more at substantially greater processing cost. This does not guarantee equivalent recall or constant duplicate counts on other footage.

## 8. Subsequent evidence and review improvements

After the tracking benchmark, screenshot selection was changed because the previous implementation repeatedly overwrote an earlier clear rider view with the latest qualifying frame, often near the image edge.

The pipeline now ranks qualifying rider screenshots using boundary clearance, head size, head-crop sharpness, and no-helmet confidence. It retains the better image and its matching frame number and rider boxes. Plate collection remains independent, while duplicate suppression retains the latest rider position. This requires no additional model inference.

All **13 backend regression tests passed**, including five screenshot-selection tests and the existing tracking and plate-collection tests. However, the September benchmark has **not** been rerun with this screenshot-selection change. No measured improvement in screenshot acceptability, runtime, or event recall is claimed for it yet. Existing stored screenshots require reprocessing to change.

Related UI work added video thumbnails to the upload and job-history views, and separated detected violations from completed jobs with no saved violations. These changes improve review usability; they are not detector accuracy improvements.

Implementation checkpoints: `f91c3e9` contains tracking and review UI improvements; `adef216` contains clearer rider evidence selection. Both were pushed to `main` during this work.

## 9. Conclusions and remaining validation

The strongest supported progression is **better domain-specific detection from v3**, followed by **fewer duplicate records and fewer wrongly suppressed passages from tracking refinement**. At 1s, measured recall progressed 32% → 76% → 84%; at 0.25s, it progressed 68% → 84% → 88%.

Higher sampling improved recall in the original v2 and v3 runs, but with substantial runtime cost and more duplicates before refinement. The refined results make 1s a reasonable efficiency candidate for this particular collection; they do not establish a universal optimal setting.

Before making production claims, obtain independent human event labels, annotate plate ownership and exact plate text, evaluate unseen camera sessions and opposite street angles, and repeat timing measurements under matched conditions. The September clips have already guided development, so a new untouched test collection is needed. Review the new screenshot selector on real footage separately from detector recall.

## 10. Evidence sources

Tracked documentation and implementation:

- [Round-1 training report](model-training.md)
- [Round-2 training report](model-training-round2.md)
- [Dataset preparation script](../training/prepare_dataset.py)
- [Detection and tracking pipeline](../backend/app/services/pipeline.py)
- [ByteTrack-style tracker](../backend/app/services/byte_tracker.py)
- [Tracking regression tests](../backend/tests/test_motorcycle_deduplication.py)
- [Plate collection tests](../backend/tests/test_plate_collection.py)
- [Rider screenshot tests](../backend/tests/test_rider_evidence.py)

Local artifacts used for verification (generated or untracked; these links may not resolve in a fresh clone):

- [Round-3 helmet training arguments](../training/round3/runs/helmet-v3/args.yaml), [plate training arguments](../training/round3/runs/plate-v3/args.yaml)
- [Round-3 helmet CSV](../training/round3/runs/helmet-v3/results.csv), [plate CSV](../training/round3/runs/plate-v3/results.csv)
- Round-3 image counts: `training/round3/datasets/{helmet,plate}/images/{train,val}`; source stem comparison against `train-data3/images`.
- [v2 timing report](../analysis/september-evaluation/report.md), [v2 event evaluation](../analysis/september-evaluation/accuracy-report.md)
- [Original v3 comparison](../analysis/september-evaluation/v3/report.md)
- [Tracking refinement evaluation](../analysis/september-evaluation/v3-dedup-final/report.md), [scored results and runtimes](../analysis/september-evaluation/v3-dedup-final/accuracy.json)

The tables above retain the essential findings in this report even when the local artifacts are unavailable.
