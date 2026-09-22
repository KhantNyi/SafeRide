# SafeRide September video benchmark

Test date: 2026-09-05. Source: `C:/Users/ADMIN/Downloads/Sp data 2 september`.

Processed all 20 clips (326.44 seconds total; about 5 minutes 26 seconds), at four configurations: 80 successful jobs, no failed jobs. Clips range from 3.70 to 41.57 seconds and average 16.32 seconds. Encoded resolution is 3840×2160 at approximately 30 FPS; automatic orientation produces portrait frames.

Hardware: NVIDIA GeForce RTX 4070 SUPER, CUDA inference with FP16. Used the existing helmet-v2 model with full-frame helmet inference, object and helmet confidence 0.35, plate confidence 0.30, adaptive sampling enabled, and the existing plate/tracker quality gates. The complete resolved settings are in results.json. Runs were sequential using one process. Source videos, application configuration and the normal application database were not modified. Benchmark records/media are retained locally in this folder.

## Measured results

The table below shows model record counts. The completed [visual accuracy evaluation](accuracy-report.md) finds 25 confirmed events after a source-label correction during v3 review: recall is 32% at 1s, 52% at 0.5s, and 68% at 0.25s. These are single-reviewer AI visual labels, not independently human-validated. Plate crops can include duplicates or incorrect associations. Nonempty OCR is not a correct-registration metric.

| Interval | Base FPS | OCR | Total runtime | Records | Clips with records | Plate crops | Nonempty OCR | Sampled frames |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0s | 1 | False | 120.4s | 8 | 8 | 5 | 0 | 450 |
| 0.5s | 2 | False | 138.7s | 13 | 12 | 8 | 0 | 995 |
| 0.25s | 4 | False | 335.7s | 19 | 15 | 16 | 0 | 2784 |
| 0.5s | 2 | True | 145.6s | 13 | 12 | 8 | 8 | 995 |

Timing caveat: the first 1-second run includes detector cold startup; later runs reuse loaded detectors. OCR initialization occurs only in the OCR-enabled run. These are single passes, not repeated statistical benchmarks. Excluding IMG_7075 from both OCR-disabled 1s and 0.5s runs gives 99.9s versus 130.0s for the same remaining 19 clips: approximately 30% more processing time. This reduces startup bias but does not eliminate all cache/order effects.

The 0.25s run costs 2.42× the 0.5s run and takes slightly longer than the full 326.44s footage duration. This is offline throughput, not a live-camera latency guarantee. OCR at 0.5s adds approximately 5% runtime in this dataset. Adaptive sampling means actual analyzed frames exceed the nominal base FPS; integer frame-interval rounding also affects the exact cadence.

## Visual findings

- At 0.5s, newly recorded clips relative to 1s are IMG_7076, IMG_7078, IMG_7089 and IMG_7094; IMG_7086 also gains a separate motorcycle event.
- At 0.25s, additional apparent motorcycle events appear in IMG_7080, IMG_7081, IMG_7085 and IMG_7088.
- Two clear extra duplicate records at 0.25s: IMG_7086 frames 381/411, tracks 1/2; IMG_7090 frames 280/310, tracks 1/2. The same motorcycle/rider is visible in each pair. Removing these two extras leaves 17 apparent distinct motorcycle events among 19 records, not a count of individual people or a complete ground-truth total. Evidence crops are in review/.
- Confirmed miss at every tested rate: IMG_7082 around 8.5s, bareheaded driver. See [source detail](review/IMG_7082-detail.jpg).
- Confirmed miss at every tested rate: IMG_7087 around 6.5s, bareheaded passenger behind a helmeted driver. See [source detail](review/IMG_7087-detail.jpg).
- At 0.25s, metadata for both missed clips contains zero sampled no-helmet boxes and zero rider associations. This points to a detector/view/generalization limitation upstream of saved-violation confirmation, rather than simply duplicate suppression or plate gates.
- Follow-up source review at approximately 0.5s spacing found no confirmed violations in IMG_7077, IMG_7092 or IMG_7093. See the accuracy report for annotation limitations.
- OCR returned text for all 8 captured plates at 0.5s, but IMG_7076 returned only two Thai characters, IMG_7090 returned digits only, and IMG_7079 read the visible digit group 482 as 402. OCR requires human verification; no OCR accuracy percentage is established.

## Per-clip outputs

Each configuration cell is saved records / wall-clock seconds. All jobs completed.

| Clip | Duration | 1s, OCR off | 0.5s, OCR off | 0.25s, OCR off | 0.5s, OCR on |
|---|---:|---:|---:|---:|---:|
| IMG_7075.MOV | 14.90s | 1 / 20.3s | 1 / 8.5s | 1 / 20.1s | 1 / 13.5s |
| IMG_7076.MOV | 6.03s | 0 / 2.3s | 1 / 3.4s | 1 / 8.6s | 1 / 4.0s |
| IMG_7077.MOV | 19.20s | 0 / 6.4s | 0 / 6.5s | 0 / 9.0s | 0 / 6.6s |
| IMG_7078.MOV | 18.07s | 0 / 6.2s | 1 / 8.6s | 1 / 18.2s | 1 / 8.0s |
| IMG_7079.MOV | 3.70s | 1 / 1.7s | 1 / 2.5s | 1 / 5.1s | 1 / 2.7s |
| IMG_7080.MOV | 41.57s | 1 / 13.1s | 1 / 17.8s | 2 / 46.3s | 1 / 17.0s |
| IMG_7081.MOV | 16.73s | 0 / 4.7s | 0 / 5.0s | 1 / 18.8s | 0 / 5.0s |
| IMG_7082.MOV | 17.07s | 0 / 5.0s | 0 / 5.3s | 0 / 8.3s | 0 / 5.1s |
| IMG_7083.MOV | 5.87s | 1 / 2.3s | 1 / 2.7s | 1 / 7.2s | 1 / 3.1s |
| IMG_7084.MOV | 25.67s | 1 / 8.4s | 1 / 11.2s | 1 / 25.5s | 1 / 12.1s |
| IMG_7085.MOV | 4.53s | 0 / 1.5s | 0 / 1.7s | 1 / 9.3s | 0 / 1.7s |
| IMG_7086.MOV | 30.63s | 1 / 9.8s | 2 / 16.9s | 3 / 40.0s | 2 / 17.2s |
| IMG_7087.MOV | 11.40s | 0 / 3.2s | 0 / 3.4s | 0 / 5.6s | 0 / 3.6s |
| IMG_7088.MOV | 11.77s | 0 / 3.3s | 0 / 3.7s | 1 / 15.0s | 0 / 3.5s |
| IMG_7089.MOV | 33.10s | 0 / 9.3s | 1 / 12.4s | 1 / 28.1s | 1 / 12.1s |
| IMG_7090.MOV | 15.03s | 1 / 6.2s | 1 / 8.2s | 2 / 20.5s | 1 / 8.9s |
| IMG_7091.MOV | 14.37s | 1 / 5.4s | 1 / 7.4s | 1 / 18.0s | 1 / 7.3s |
| IMG_7092.MOV | 13.77s | 0 / 3.9s | 0 / 4.2s | 0 / 6.9s | 0 / 4.2s |
| IMG_7093.MOV | 9.00s | 0 / 2.6s | 0 / 2.7s | 0 / 4.4s | 0 / 2.8s |
| IMG_7094.MOV | 14.03s | 0 / 4.7s | 1 / 6.6s | 1 / 20.6s | 1 / 6.9s |

## Assessment

0.5-second sampling with OCR enabled is the most practical tested balance for supervised review: 13 records in 145.6 seconds for 326.44 seconds of footage. 0.25 seconds is useful for a slower second pass to recover additional events, but introduces duplicate records and still misses clearly visible violations. No application default was changed.

Follow-up visual labeling and precision/recall scoring are complete in [accuracy-report.md](accuracy-report.md), with labels and explicit record-to-event assignments retained. Eight confirmed events are missed even at 0.25s. Next priorities are detector improvements, tracker ID continuity, and plate association. An independent human review would strengthen the labels.

Artifacts: [raw results](results.json), [evidence browser](evidence.html), source/contact sheets in review/, isolated SQLite database benchmark.db, and reproducible runner run_benchmark.py. The evidence browser works from local files without starting the application.
