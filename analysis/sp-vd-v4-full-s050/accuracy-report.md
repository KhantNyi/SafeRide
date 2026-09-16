# SafeRide v4: Sp vd visual evaluation

## Result

V4 correctly recorded **23 of 40 confirmed motorcycle passages (57.5% recall)**. It missed 17. Of 29 saved records, 23 correctly identify unique passages, 4 are duplicates, 1 targets the wrong motorcycle, and 1 is uncertain.

**Precision: 82.1%; recall: 57.5%; F1: 67.6%.** Precision penalizes duplicate records and excludes the one uncertain record. These measure the complete saved-alert pipeline, including detection, association, tracking and alert rules; they are not detector mAP.

## Test configuration and runtime

- 24 clips from `C:\Users\ADMIN\Downloads\Sp vd`; all completed.
- 456.58 seconds of source video; 923 sampled frames; 103.15 seconds processing (about 4.43 times video duration per processing second). Visual-review time is separate.
- Helmet-v4 and plate-v4, existing YOLO11s object detector, GPU 0; full-frame inference with helmet crop inference disabled.
- Interpreted requested 0.5 sample rate as one frame every 0.5 seconds (2 fps), not 0.5 fps. Fixed sampling, adaptive sampling off; metadata confirms 30-frame spacing.
- Image size 960; confidence thresholds: object 0.35, helmet 0.35, plate 0.30. OCR enabled.

## Visual reference method

I visually reviewed source-only contact sheets at 0.5-second spacing for all clips, with targeted full-resolution source frames to resolve heads. A positive event is one motorcycle passage with at least one visibly unhelmeted rider or passenger. Multiple unhelmeted people on one motorcycle count once. I then matched every saved alert by motorcycle identity and its highlighted target, not timestamp alone.

These are AI visual reference labels requested by the user, not independently human-validated ground truth. Hood-covered, distant or clipped heads that cannot be resolved are excluded. Short events entirely between samples could be missed by this reference review. No per-person, bounding-box, true-negative accuracy or OCR exact-match score is claimed. Source labels, uncertainty notes and frame references are in [reference-labels.json](reference-labels.json). Every alert decision is in [record-adjudications.json](record-adjudications.json).

## Per-clip passage results

| Clip | Confirmed passages | Correctly recorded | Missed | Duplicate records | Wrong-target/false records | Uncertain records |
|---|---:|---:|---:|---:|---:|---:|
| IMG_6665.MOV | 2 | 0 | 2 | 0 | 1 | 0 |
| IMG_6698.MOV | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG_6699.MOV | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG_6701.MOV | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG_6705.MOV | 2 | 2 | 0 | 1 | 0 | 0 |
| IMG_6712.MOV | 1 | 0 | 1 | 0 | 0 | 0 |
| IMG_6714.MOV | 1 | 0 | 1 | 0 | 0 | 0 |
| IMG_6760.MOV | 2 | 2 | 0 | 0 | 0 | 0 |
| IMG_6761.MOV | 1 | 1 | 0 | 0 | 0 | 0 |
| IMG_6766.MOV | 2 | 1 | 1 | 0 | 0 | 0 |
| IMG_6767.MOV | 3 | 2 | 1 | 0 | 0 | 0 |
| IMG_6768.MOV | 2 | 1 | 1 | 0 | 0 | 0 |
| IMG_6773.MOV | 2 | 1 | 1 | 0 | 0 | 0 |
| IMG_6774.MOV | 2 | 0 | 2 | 0 | 0 | 0 |
| IMG_6776.MOV | 1 | 0 | 1 | 0 | 0 | 0 |
| IMG_6779.MOV | 2 | 1 | 1 | 1 | 0 | 0 |
| IMG_6786.MOV | 1 | 1 | 0 | 0 | 0 | 0 |
| IMG_6787.MOV | 4 | 3 | 1 | 0 | 0 | 0 |
| IMG_6788.MOV | 2 | 2 | 0 | 0 | 0 | 0 |
| IMG_6791.MOV | 1 | 1 | 0 | 0 | 0 | 1 |
| IMG_6792.MOV | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG_6793.MOV | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG_6798.MOV | 6 | 4 | 2 | 2 | 0 | 0 |
| IMG_6805.MOV | 3 | 1 | 2 | 0 | 0 | 0 |

## Missed passages

| Event | Clip | Approx. start (s) | Visual description |
|---|---|---:|---|
| 6665-E1 | IMG_6665.MOV | 8.0 | Tan-shirt backpack passenger, helmeted dark-clothed driver, yellow plate 3399 |
| 6665-E2 | IMG_6665.MOV | 11 | Gray-shirt solo rider with backpack on white scooter; visible bare hair |
| 6712-E1 | IMG_6712.MOV | 2.5 | Blue-shirt backpack rider, green sports motorcycle, visible bare hair |
| 6714-E1 | IMG_6714.MOV | 18.5 | Brown/olive top backpack rider on red scooter, visible bare hair |
| 6766-E2 | IMG_6766.MOV | 22 | Dark green shirt solo rider with loose black hair on dark scooter |
| 6767-E3 | IMG_6767.MOV | 18.5 | Bareheaded white-shirt passenger with light backpack behind green-shirt helmeted driver |
| 6768-E2 | IMG_6768.MOV | 13.5 | Solo bareheaded ponytail rider in number 12 sports jersey |
| 6773-E2 | IMG_6773.MOV | 10 | Purple-shirt bareheaded passenger behind black-shirt helmeted driver |
| 6774-E1 | IMG_6774.MOV | 9.5 | Bareheaded black-clothed long-haired passenger with pale pink backpack behind white-helmet driver |
| 6774-E2 | IMG_6774.MOV | 13 | Beige-shirt bareheaded passenger behind helmeted driver |
| 6776-E1 | IMG_6776.MOV | 7.5 | Bareheaded solo rider in white graphic T-shirt and black shorts |
| 6779-E1 | IMG_6779.MOV | 2 | Bareheaded tan-top backpack passenger behind black-helmet driver |
| 6787-E2 | IMG_6787.MOV | 8.5 | Bareheaded beige-top passenger on mint scooter, white-top driver |
| 6798-E1 | IMG_6798.MOV | 10 | Bareheaded white-shirt passenger behind white-helmet driver, first scooter |
| 6798-E6 | IMG_6798.MOV | 23 | Bareheaded white-shirt solo rider with beige backpack on sports bike |
| 6805-E1 | IMG_6805.MOV | 3.5 | Bareheaded long-haired white-top rider on cream scooter |
| 6805-E3 | IMG_6805.MOV | 7.5 | Bareheaded white-shirt black-backpack passenger behind white-helmet driver |

## Main failure examples

- IMG_6665, frame 540: a bare head on the motorcycle ahead is assigned to a helmeted foreground rider and plate 3872. The wrong-target record is penalized, and the actual unhelmeted passage remains missed.
- Duplicate records: IMG_6705 tracks 4/3; IMG_6779 tracks 6/8; IMG_6798 tracks 5/6 and 13/14. Different track IDs recorded the same physical passage twice.
- IMG_6712, IMG_6714, IMG_6774 and IMG_6776 contain confirmed events but produced no saved records. The final alert pipeline therefore needs better recall; these results alone do not isolate whether detection, association or alert gating caused each miss.

## Plate and OCR output

20 of 29 records saved a plate crop; 10 produced nonempty OCR text. Pipeline OCR statuses: {'uncertain': 10, 'unreadable': 19}.

These are output-availability counts, not recognition accuracy. Full plate strings were not independently transcribed and scored. The wrong-motorcycle example also shows that detecting a plate does not ensure it belongs to the offending rider.

## Preservation and artifacts

All inventoried baseline/v1/v2/v3/v4 model weights match their pre-benchmark SHA-256 hashes. Previous training reports and app configuration remain preserved. Round-four training is recorded in [model-training-round4.md](../../docs/model-training-round4.md). This benchmark used its own database and media directory.

- [Raw benchmark report](report.md)
- [Evidence gallery](evidence.html)
- [Machine-readable metrics](accuracy-metrics.json)
- [Full settings and results](results.json)

Conclusion: on these clips, full-frame v4 at a 0.5-second interval runs faster than video playback, but the saved-alert pipeline misses 42.5% of confirmed passages and creates duplicate records. Association and tracking errors should be addressed alongside detector recall before treating the records as reliable final results.
