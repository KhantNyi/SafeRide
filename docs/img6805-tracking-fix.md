# IMG_6805 motorcycle identity fix

## Problem

In job `9f121c0013a846c6847c89cbc4256280`, two detector boxes at frame 300 created raw tracks 3 and 4 for the same motorcycle. Track 4 was aliased to identity 3, then remained lost near the entry position. At frame 444 it matched the last motorcycle entering the scene while the original motorcycle was still visible elsewhere. Both then contributed votes and evidence to identity 3. Three violating passages produced only two records.

## Change

- Retire unmatched aliased duplicate tracks so their stale positions cannot attract later arrivals.
- Separate visible aliases when their boxes are spatially disjoint, before accumulating helmet votes, plate crops or evidence. Record these identities as distinct for duplicate suppression.
- Preserve active duplicate aliases and normal same-motorcycle deduplication.

## Validation

All 34 backend unit tests passed, including regressions for a later arrival at an old duplicate's position and for visible aliases separating without sharing helmet votes.

Restarted the backend and restored its current UI settings: v4 models, full-frame helmet inference, 0.5-second base sampling, OCR enabled. Adaptive sampling remained enabled, resulting in 75 analyzed frames in both compared runs.

New app job: `469013c45c0f4b0783e800cc7ff3f7d4`, completed with three records:

| Passage | Track | Evidence frame |
|---|---:|---:|
| Bareheaded rider on cream scooter | 2 | 222 |
| Bareheaded rider with rear cargo box | 3 | 306 |
| Bareheaded passenger behind helmeted driver | 7 | 468 |

Visually checked all three saved evidence images. At frames 444, 468 and 480 the earlier rider and last motorcycle retain separate IDs 3 and 7. The helmeted solo rider retains ID 5 and generates no violation record. See [evidence comparison](../analysis/img6805-tracking-fix-evidence.jpg).

Previous jobs and model weights were preserved. This is a tracking correction, not retraining. It does not change helmet detection when heads become small or occluded; a later overlay can still lose a head association. The full 24-video benchmark has not been rerun with this change.
