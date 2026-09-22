# Brief confidence-threshold test

Date: 2026-09-07. Twelve completed full-clip jobs: three clips at four configurations. Source duration per configuration: 63.70 seconds (IMG_7077: 19.2s; IMG_7087: 11.4s; IMG_7089: 33.1s).

Fixed: v3 helmet and plate weights, full-frame helmet inference, 0.5-second base sampling, adaptive sampling enabled, plate threshold 0.30, OCR disabled, current tracking and screenshot selection. Outputs and database are isolated; normal app settings and records were not changed.

Selected labels: no violation in IMG_7077; one previously missed bareheaded passenger in IMG_7087; one previously found bareheaded driver and one missed bareheaded passenger in IMG_7089. Three labeled events total, of which two are targeted misses. These are existing AI visual labels, not independently human-validated.

| Helmet threshold | Object threshold | Previously missed passengers recovered / 2 | Total labeled events found / 3 | False alerts | Duplicates | Runtime |
|---|---|---|---|---|---|---|
| 0.35 | 0.35 | 0 | 1 | 0 | 0 | 46.3s |
| 0.25 | 0.35 | 0 | 1 | 1 | 0 | 34.3s |
| 0.15 | 0.35 | 1 | 2 | 0 | 0 | 40.3s |
| 0.25 | 0.25 | 0 | 1 | 1 | 0 | 34.0s |

Saved-frame assignments: IMG_7089 frame 483 is the already detected black-shirt driver with gray backpack in all configurations. At helmet 0.15, IMG_7087 frame 231 captures the previously missed passenger in white. At helmet 0.25, IMG_7077 frame 306 incorrectly flags the white-helmeted rider; lowering object confidence to 0.25 retains that same false alert. Evidence for the baseline driver, newly recovered passenger, and false alert was visually inspected. The other outputs match those clip/frame identities.

Conclusion: helmet confidence materially changes event results. Lowering it to 0.15 recovered one target miss in this small test; lowering object confidence from 0.35 to 0.25 with helmet fixed at 0.25 provided no additional recovery. This does not show object confidence is irrelevant generally, nor establish 0.15 as a safe global default. Plate confidence was fixed, so no plate-threshold conclusion is drawn.

Threshold effects are not monotonic at the saved-event level: the false alert at 0.25 did not survive at 0.15. Helmet thresholds affect both helmet classes, adaptive sampling, association, and accumulated votes. The precise cause of this change was not isolated by this experiment. A saved screenshot can also have confidence above the original threshold even when lower-confidence earlier frames were necessary to build the event.

Timing is single-pass and ordered; baseline includes cold detector startup. Do not interpret the lower later runtimes as a threshold speedup. Repeat on more positive and negative clips before changing defaults.

Reproduce with ` .venv/Scripts/python.exe analysis/september-evaluation/test_confidence.py` from the repository root. Raw results, metadata, evidence, and benchmark databases are in the four sibling configuration directories. Existing output locations are reused by that script; preserve a copy before rerunning if required.
