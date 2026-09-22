# September benchmark: v3 versus v2

All 20 clips (326.44 seconds) were rerun with helmet-v3 and plate-v3. The same 25 confirmed motorcycle-passage labels are used for both versions; one ambiguous head-cloth case is excluded. Labels are AI visual annotations from source review at approximately 0.5s spacing and targeted full-resolution inspection, not independently human-validated.

| Interval | OCR | V2 found / 25 | V3 found / 25 | V3 recall | V3 precision* | V3 false alerts | V3 duplicates | V2 runtime | V3 runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0s | False | 8/25 | 19/25 | 76.0% | 95.0% | 0 | 1 | 120.4s | 157.3s |
| 0.5s | False | 13/25 | 20/25 | 80.0% | 83.3% | 0 | 4 | 138.7s | 209.3s |
| 0.25s | False | 17/25 | 21/25 | 84.0% | 75.0% | 0 | 7 | 335.7s | 578.0s |
| 0.5s | True | 13/25 | 20/25 | 80.0% | 83.3% | 0 | 4 | 145.6s | 216.4s |

*Precision counts only one true-positive alert per event, penalizing duplicate records. Helmet event correctness does not establish plate association or OCR correctness.

V3 materially improves recall at every interval. On this sample, 1s offers the strongest speed/duplicate tradeoff: 19 events in 157.3s with one duplicate. At 0.5s, one more event costs 52.0s and increases duplicates to four. At 0.25s, another event costs 368.7s and increases duplicates to seven. The 0.5s OCR pass catches the same 20 events in 216.4s. These observations do not change application defaults.

Label correction: the earlier report used 24 events. Full-resolution source frame 270 in IMG_7094 confirms gray hair on the first delivery rider previously mistaken for a helmet. Adding that event gives 25. V2 and v3 are both scored against this corrected denominator; the original v2 detections are unchanged. [Source close-up](review/7094-source270.jpg).

## Per-clip v3 detections

| Clip | Ground truth | 1s | 0.5s | 0.25s | 0.5s + OCR |
|---|---:|---:|---:|---:|---:|
| IMG_7075.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7076.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7077.MOV | 0 | 0 | 0 | 0 | 0 |
| IMG_7078.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7079.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7080.MOV | 4 | 3 | 3 | 3 | 3 |
| IMG_7081.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7082.MOV | 2 | 2 | 2 | 2 | 2 |
| IMG_7083.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7084.MOV | 2 | 1 | 1 | 2 | 1 |
| IMG_7085.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7086.MOV | 2 | 1 | 2 | 2 | 2 |
| IMG_7087.MOV | 1 | 0 | 0 | 0 | 0 |
| IMG_7088.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7089.MOV | 2 | 1 | 1 | 1 | 1 |
| IMG_7090.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7091.MOV | 1 | 1 | 1 | 1 | 1 |
| IMG_7092.MOV | 0 | 0 | 0 | 0 | 0 |
| IMG_7093.MOV | 0 | 0 | 0 | 0 | 0 |
| IMG_7094.MOV | 2 | 1 | 1 | 1 | 1 |

## What remains at 0.25 seconds

- 7080-E2, around 17s: Bareheaded black long-sleeve rider on black/green scooter, plate digits 5528.
- 7087-E1, around 6s: Bareheaded blonde passenger in white T-shirt and black skirt holding drink.
- 7089-E2, around 23s: Bareheaded brown-haired passenger in black hoodie with hood down.
- 7094-E1, around 11s: Bareheaded female passenger in black top and white trousers, motorcycle plate digits 1380.

V3 is better overall but does not recover every v2 detection. At 0.5s and 0.25s, v2 caught the black/green scooter rider in IMG_7080 (E2) and the female passenger in IMG_7094 (E1); v3 misses both. V3 detects the separate gray-haired delivery rider in IMG_7094 instead.

## Plate output coverage

| Interval | OCR | Saved records | Plate crops | Nonempty OCR |
|---|---|---:|---:|---:|
| 1.0s | False | 20 | 15 | 0 |
| 0.5s | False | 24 | 17 | 0 |
| 0.25s | False | 28 | 23 | 0 |
| 0.5s | True | 24 | 17 | 12 |

Plate coverage counts include duplicate records and do not measure transcription accuracy.

## Method and limitations

- Both specialist weights changed together; this measures the combined v3 pipeline, not the isolated contribution of either model.
- Weights: training/round3/models/helmet-v3-best.pt and training/round3/models/license-plate-v3-best.pt. Both were verified to match the corresponding round3 training-run best.pt files by SHA-256 before this rerun.
- All other resolved inference settings match v2: full-frame helmet inference, confidence 0.35, plate confidence 0.30, adaptive sampling enabled. Base rates are 1, 2, and 4 FPS; adaptive sampling analyzes extra frames.
- Separate benchmark database and media directory; application defaults and normal database were not modified.
- Single sequential timing passes. The first run in each model benchmark includes cold detector startup; timings are not repeated controlled estimates.
- Every saved v3 record was visually matched by motorcycle identity. Overlapping time windows alone were not used to assign identity.
- Small, single-location sample; no claim of general production accuracy or complete OCR accuracy.

Artifacts: [raw v3 results](results.json), [scored assignments](accuracy.json), [visual record labels](record-labels.json), [v3 evidence browser](evidence.html), [shared source labels](../labels.json), [v2 accuracy report](../accuracy-report.md).

Reproduce: `.venv/Scripts/python.exe analysis/september-evaluation/run_benchmark.py --v3`, then visually label the new records before running `python analysis/september-evaluation/score_v3.py`. Record IDs depend on inference output; the scorer rejects unreviewed records.
