# Street-level tripod evaluation

Date: 2026-08-25

## Outcome

The best balanced setting on this set is the v2 helmet model with **full-frame inference**, helmet confidence **0.35**, object confidence **0.35**, and a **0.5-second sample interval**. It detected 6 of 8 manually identified no-helmet events (75% recall), produced 1 false-positive record, and completed the nine clips in 58.5 seconds including model startup.

For maximum detection, changing only the sample interval to **0.25 seconds** detected 7 of 8 events (87.5% recall). The cost was 125.4 seconds of processing, 2 false-positive records, and 1 duplicate record. This is a useful high-sensitivity mode, but it is less attractive as the default.

The live application was not changed by these tests. It remains on v2, full-frame inference, 0.35 helmet confidence, 0.35 object confidence, and a 1-second sample interval.

## Test set

The supplied folder contained 9 clips, `IMG_6834.MOV` through `IMG_6842.MOV`, totaling about 3.9 minutes. Ten clips were expected, so one clip appears to be missing.

Manual review identified 8 visible no-helmet events across 6 clips:

| Clip | Expected events | Notes |
|---|---:|---|
| IMG_6834 | 0 | Helmeted riders only |
| IMG_6835 | 2 | Two separate no-helmet motorcycle events |
| IMG_6836 | 2 | Two separate no-helmet events |
| IMG_6837 | 1 | Red-hoodie rider without a helmet |
| IMG_6838 | 1 | Passenger without a helmet; driver helmeted |
| IMG_6839 | 0 | Helmeted rider only |
| IMG_6840 | 1 | No-helmet rider near another helmeted rider |
| IMG_6841 | 0 | Helmeted riders only |
| IMG_6842 | 1 | Passenger without a helmet; driver helmeted |

## Controlled results

All runs used the same event windows and ±15-frame matching tolerance. OCR was disabled to isolate detection performance. The plate model and all non-listed pipeline settings were held constant.

| Helmet weights / settings | Detected | Recall | False positives | Precision | Duplicates | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| **v2 full, H=.35, O=.35, sample=.25s** | **7/8** | **87.5%** | 2 | 80.0% | 1 | 125.4s |
| **v2 full, H=.35, O=.35, sample=.50s** | **6/8** | **75.0%** | **1** | **85.7%** | **0** | 58.5s |
| v2 full, H=.35, O=.25, sample=.50s | 6/8 | 75.0% | 2 | 75.0% | 0 | 56.8s |
| v2 full, H=.35, O=.35, sample=1.00s | 5/8 | 62.5% | 1 | 83.3% | 0 | 42.0s |
| v2 full, H=.20, O=.35, sample=1.00s | 5/8 | 62.5% | 1 | 83.3% | 0 | 44.4s |
| v2 full, H=.35, O=.25, sample=1.00s | 5/8 | 62.5% | 1 | 83.3% | 0 | 41.0s |
| original full, H=.35, O=.35, sample=1.00s | 5/8 | 62.5% | 2 | 71.4% | 0 | 43.2s |
| v2 crop, H=.20, O=.35, sample=1.00s | 4/8 | 50.0% | 0 | 100% | 0 | 40.1s |
| original crop, H=.35, O=.35, sample=1.00s | 4/8 | 50.0% | 1 | 80.0% | 0 | 43.6s |
| v2 crop, H=.35, O=.35, sample=1.00s | 3/8 | 37.5% | 0 | 100% | 0 | 38.0s |

## External YOLO11m candidate

After the initial evaluation, a publicly available YOLO11m motorcycle helmet checkpoint (`nnsohamnn/helmet-detection-yolo11`) was downloaded as an inactive candidate and tested with the same labels. Its classes are compatible with SafeRide (`With Helmet` and `Without Helmet`), but it generalized poorly to these clips.

| Candidate settings | Detected | Recall | False positives | Precision | Duplicates | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11m full, H=.35, O=.35, sample=1.00s | 1/8 | 12.5% | 0 | 100% | 1 | 42.1s |
| YOLO11m full, H=.35, O=.35, sample=.50s | 1/8 | 12.5% | 3 | 25% | 0 | 58.2s |
| YOLO11m crop, H=.35, O=.35, sample=1.00s | 1/8 | 12.5% | 1 | 50% | 0 | 58.9s |

This candidate is not an upgrade. SafeRide helmet-v2 remains active. The outcome demonstrates that a larger/newer architecture does not provide a useful improvement when its learned training domain does not match the target footage.

`H` is helmet confidence and `O` is object confidence. Runtime includes model cold-start overhead and is intended for relative comparison on this machine.

## Per-clip findings

- `IMG_6834`, `IMG_6839`, and `IMG_6841` remained clean in every configuration; none produced a false violation.
- `IMG_6836` improved from 1/2 to 2/2 when sampling changed from 1 second to 0.5 seconds.
- `IMG_6842` was recovered only at the 0.25-second interval. This is strong evidence of a temporal sampling miss rather than a confidence-threshold miss.
- One of the two `IMG_6835` events was missed by every v2 full-frame configuration, including 0.25-second sampling. This is the clearest remaining model/domain-generalization failure in the set.
- `IMG_6837` was detected by v2 full-frame inference, but it also generated the false-positive records seen in the higher-recall configurations. This clip needs more labeled examples to separate the true rider from confusing nearby subjects.
- Full-frame inference materially outperformed rider crops on this street-level view. At otherwise identical v2 defaults, recall rose from 37.5% to 62.5%.
- Lowering helmet confidence from 0.35 to 0.20 did not change full-frame results. Lowering object confidence from 0.35 to 0.25 also did not recover an event and increased false positives in the denser-sampling run.
- The original and v2 full-frame models had the same 62.5% recall at the 1-second interval, but v2 produced one false alert instead of two. V2 therefore remains preferable on this test.

## Recommendation

Use this street-level preset first:

- Helmet inference: **Full frame**
- Helmet confidence: **0.35**
- Object confidence: **0.35**
- Sample interval: **0.50 seconds**

Use a 0.25-second sample interval when missing a violation is more costly than the additional runtime, false positives, and occasional duplicate. Do not lower either confidence slider based on this test; neither change improved recall.

The durable model fix is to add labeled street-level frames—especially examples like the consistently missed `IMG_6835` event and the confusing `IMG_6837` scene—to the training set while retaining elevated CCTV examples. A mixed-view validation split should be used so improvements on street footage do not silently reduce CCTV performance.

## Limitations

- This is a small, single-location set with 8 positive events, so one event changes recall by 12.5 percentage points.
- Ground truth was manually established from the supplied clips and has not been independently reviewed.
- The evaluation measures event detection and false records, not plate OCR accuracy; OCR was intentionally disabled.
- CCTV performance was not retested here, so no claim is made that a street-level setting or future retraining preserves the older footage performance until both sets are evaluated together.

Raw evaluator outputs and the event labels are stored beside this report.
