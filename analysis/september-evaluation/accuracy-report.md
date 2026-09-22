# September video accuracy evaluation

Visual review found **25 confirmed violation events across 20 clips (326.44 seconds)**. One additional head-cloth case is uncertain and excluded. A violation event is one motorcycle passage with at least one unhelmeted occupant; two bareheaded people on one motorcycle count once.

| Sample interval | Base FPS | OCR | Found / 25 | Recall | Event precision* | F1* | Duplicate records | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0s | 1 | False | 8/25 | 32.0% | 100.0% | 48.5% | 0 | 120.4s |
| 0.5s | 2 | False | 13/25 | 52.0% | 100.0% | 68.4% | 0 | 138.7s |
| 0.25s | 4 | False | 17/25 | 68.0% | 89.5% | 77.3% | 2 | 335.7s |
| 0.5s | 2 | True | 13/25 | 52.0% | 100.0% | 68.4% | 0 | 145.6s |

*Event precision allows one true-positive record per ground-truth event; duplicate records count against precision. No saved helmet alert was visually judged to be an unrelated/nonviolating motorcycle. Consequently helmet-correct record precision is 100% in each run, but the 0.25s run has only 17 unique events among 19 records (89.5% event precision). This does not measure correct plate association or OCR.

Label correction: full-resolution review during the v3 comparison confirmed that the first IMG_7094 delivery rider has gray hair, not a helmet. This adds a 25th event and supersedes the previously reported 24-event denominator. V2 recall changes from 33.3/54.2/70.8% to 32/52/68%; detected counts remain 8/13/17.

## Interpretation

The 1s default interval catches 32% of confirmed violations. Moving to 0.5s catches five more events (+20 percentage points recall). Moving to 0.25s catches another four (+16 points), but still misses eight and adds two duplicates. OCR at 0.5s does not change event recall on these clips. No settings were changed.

If the uncertain IMG_7078 head-cloth case is a violation, the denominator becomes 26 and recall is 30.8%, 50%, and 65.4% at 1s, 0.5s, and 0.25s respectively.

These are single-reviewer AI visual annotations, not independently human-validated ground truth. Review covered source contact sheets at ~0.5s spacing across every clip, plus targeted full-resolution frames and detection evidence. This was not a blinded or every-frame annotation. Report event precision/recall rather than a generic accuracy percentage: an event detector has no natural count of true-negative events.

## Per-clip confirmed events

| Clip | Confirmed events | Found at 1s | Found at 0.5s | Found at 0.25s |
|---|---:|---:|---:|---:|
| IMG_7075.MOV | 1 | 1 | 1 | 1 |
| IMG_7076.MOV | 1 | 0 | 1 | 1 |
| IMG_7077.MOV | 0 | 0 | 0 | 0 |
| IMG_7078.MOV | 1 | 0 | 1 | 1 |
| IMG_7079.MOV | 1 | 1 | 1 | 1 |
| IMG_7080.MOV | 4 | 1 | 1 | 2 |
| IMG_7081.MOV | 1 | 0 | 0 | 1 |
| IMG_7082.MOV | 2 | 0 | 0 | 0 |
| IMG_7083.MOV | 1 | 1 | 1 | 1 |
| IMG_7084.MOV | 2 | 1 | 1 | 1 |
| IMG_7085.MOV | 1 | 0 | 0 | 1 |
| IMG_7086.MOV | 2 | 1 | 2 | 2 |
| IMG_7087.MOV | 1 | 0 | 0 | 0 |
| IMG_7088.MOV | 1 | 0 | 0 | 1 |
| IMG_7089.MOV | 2 | 0 | 1 | 1 |
| IMG_7090.MOV | 1 | 1 | 1 | 1 |
| IMG_7091.MOV | 1 | 1 | 1 | 1 |
| IMG_7092.MOV | 0 | 0 | 0 | 0 |
| IMG_7093.MOV | 0 | 0 | 0 | 0 |
| IMG_7094.MOV | 2 | 0 | 1 | 1 |

## Misses remaining at 0.25s

- 7080-E1, around 6–11.5s: Bareheaded black-shirt rider with graphic backpack on yellow/orange motorcycle.
- 7080-E3, around 18–22s: Bareheaded J&T courier in gray/white/red jacket with loaded side bags.
- 7082-E1, around 8–14.5s: Bareheaded dark-blue-shirt solo rider.
- 7082-E2, around 13–17.03s: Bareheaded white-shirt driver and black-jacket ponytail passenger on lime scooter.
- 7084-E2, around 24–25.63s: Bareheaded buzzcut Grab delivery rider, plate digits 7390.
- 7087-E1, around 6–8.5s: Bareheaded blonde passenger in white T-shirt and black skirt holding drink.
- 7089-E2, around 23–28s: Bareheaded brown-haired passenger in black hoodie with hood down.
- 7094-E2, around 8.5–13s: Bareheaded gray-haired green delivery rider, plate digits 1742; corrected after full-resolution source review during v3 evaluation.

## Evidence and performance limitations

- Duplicate pairs: IMG_7086 frames 381/411 and IMG_7090 frames 280/310.
- IMG_7094 at 0.5s with OCR associates plate digits 1742 from the preceding bareheaded delivery rider with the violating motorcycle; the latter shows digits 1380. A correct helmet alert can still have incorrect evidence association.
- OCR correctness has not been fully annotated. Nonempty OCR is not accuracy; observed errors include IMG_7079 digits 482 read as 402.
- Runtime comes from single sequential passes; the 1s run includes cold detector startup. Excluding the first clip, 0.5s costs about 30% more runtime than 1s. The complete 0.25s pass costs 2.42 times the 0.5s pass.

Artifacts: [labels and methodology](labels.json), [all record-to-event assignments and metrics](accuracy.json), [benchmark results](report.md), [detection evidence](evidence.html). Source review sheets are in review/labeling/. Run `python analysis/september-evaluation/score_saved.py` to reproduce these metrics from saved results and the visual annotations embedded in that script.
