# Duplicate violation fix: v3 validation

All 20 September clips were processed again with helmet-v3 and plate-v3 at 1s, 0.5s and 0.25s intervals. These are full inference runs, not filtered copies of previous results. OCR was disabled. The same corrected set of 25 visual event labels is used.

| Interval | Before: events | After: events | Before: duplicates | After: duplicates | After: recall | After: false helmet alerts |
|---|---:|---:|---:|---:|---:|---:|
| 1.0s | 19 | 21 | 1 | 1 | 84% | 0 |
| 0.5s | 20 | 21 | 4 | 1 | 84% | 0 |
| 0.25s | 21 | 22 | 7 | 1 | 88% | 0 |

## Change

Overlapping whole-bike and rear-bike detections could coexist in one frame, spawning independent tracker IDs for the same rider. The pipeline now aliases a newly created overlapping track to an existing violation identity when both boxes correspond to the same nearest person. Both boxes remain available for rider and plate association. Established tracks are not merged just because they overlap. Pending events remain active while any aliased raw track remains active. Simultaneously observed separate motorcycles are protected from later position-only save suppression.

## Per-clip changes

- 1.0s IMG_7086.MOV: lost []; gained ['7086-E2'].
- 1.0s IMG_7094.MOV: lost []; gained ['7094-E1'].
- 0.5s IMG_7094.MOV: lost []; gained ['7094-E1'].
- 0.25s IMG_7094.MOV: lost []; gained ['7094-E1'].

## Remaining duplicates

- 1s: IMG_7082, second motorcycle, frames 438 and 486. The earlier IMG_7086 duplicate is removed, but this extra record is newly exposed by protecting distinct passages from broad positional suppression.
- 0.5s: IMG_7090, frames 345 and 417, after the tracker loses and reacquires the rider.
- 0.25s: IMG_7080, courier, frames 590 and 627. These are one event under separate identities.
- Duplicate counts are now 1 at each interval. This is a reduction at higher sampling, not complete deduplication. Resolving the remaining cases needs reliable identity matching across tracking gaps; widening a positional exclusion zone risks suppressing different riders.

## Verification and limits

- Regression tests cover the actual duplicate boxes from IMG_7086, separate nearby motorcycles, established overlapping tracks, and pending-event lifetime after the original raw ID disappears. Existing plate-collection tests also pass.
- Visual event labels and output matching were performed by the AI assistant; there is no independent human validation. One uncertain head-cloth case remains excluded.
- Results are specific to these clips. Crowded occlusions, similar riders, and track losses can still create identity errors. Higher sampling is not guaranteed to be duplicate-free.
- Plate/OCR correctness is not established by helmet-event accuracy. Application model paths and sampling defaults were not changed.

- Runtime is retained in raw results, but these validation runs used a different order (0.5s, 0.25s, 1s) from the original benchmark. No controlled speedup claim is made.

Artifacts: [raw results](results.json), [record identity labels](record-labels.json), [scored identities](accuracy.json), [evidence browser](evidence.html), [original v3 benchmark](../v3/report.md).
