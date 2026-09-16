# Sp vd: v4 full-frame video benchmark

Settings: helmet-v4 + plate-v4; existing YOLO11s object detector; full-frame detection; fixed 0.5-second sampling (adaptive sampling off); OCR enabled. Image sizes and confidence thresholds are recorded in results.json.

These are raw model outputs. For a completed visual review, see accuracy-report.md when present. Raw record counts alone do not establish accuracy or OCR correctness.

## Summary

- clips: 24
- statuses: {'completed': 24}
- video_seconds: 456.58
- wall_seconds: 103.15213059999223
- sampled_frames: 923
- violation_records: 29
- clips_with_records: 15
- records_with_plate_crop: 20
- records_with_ocr_text: 10
- plate_statuses: {'uncertain': 10, 'unreadable': 19}

## Per-video results

| Video | Duration (s) | Analyzed frames | Records | Plate crops | OCR text | Processing (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| IMG_6665.MOV | 12.8 | 26 | 1 | 1 | 1 | 11.0 | completed |
| IMG_6698.MOV | 3.0 | 6 | 0 | 0 | 0 | 0.6 | completed |
| IMG_6699.MOV | 5.9 | 12 | 0 | 0 | 0 | 1.1 | completed |
| IMG_6701.MOV | 12.0 | 24 | 0 | 0 | 0 | 2.2 | completed |
| IMG_6705.MOV | 7.7 | 16 | 3 | 1 | 1 | 2.2 | completed |
| IMG_6712.MOV | 7.3 | 15 | 0 | 0 | 0 | 1.4 | completed |
| IMG_6714.MOV | 19.9 | 40 | 0 | 0 | 0 | 3.8 | completed |
| IMG_6760.MOV | 20.2 | 41 | 2 | 2 | 0 | 3.9 | completed |
| IMG_6761.MOV | 10.7 | 22 | 1 | 1 | 1 | 2.2 | completed |
| IMG_6766.MOV | 24.0 | 48 | 1 | 1 | 1 | 4.4 | completed |
| IMG_6767.MOV | 20.6 | 42 | 2 | 2 | 1 | 4.1 | completed |
| IMG_6768.MOV | 16.1 | 33 | 1 | 1 | 0 | 3.1 | completed |
| IMG_6773.MOV | 15.4 | 31 | 1 | 1 | 0 | 2.9 | completed |
| IMG_6774.MOV | 27.4 | 55 | 0 | 0 | 0 | 4.8 | completed |
| IMG_6776.MOV | 17.4 | 35 | 0 | 0 | 0 | 3.6 | completed |
| IMG_6779.MOV | 33.1 | 67 | 2 | 1 | 0 | 7.4 | completed |
| IMG_6786.MOV | 11.5 | 24 | 1 | 1 | 0 | 2.7 | completed |
| IMG_6787.MOV | 38.9 | 78 | 3 | 3 | 2 | 8.7 | completed |
| IMG_6788.MOV | 36.1 | 73 | 2 | 1 | 0 | 8.0 | completed |
| IMG_6791.MOV | 32.3 | 65 | 2 | 0 | 0 | 7.4 | completed |
| IMG_6792.MOV | 33.9 | 68 | 0 | 0 | 0 | 7.2 | completed |
| IMG_6793.MOV | 10.8 | 22 | 0 | 0 | 0 | 2.1 | completed |
| IMG_6798.MOV | 29.4 | 59 | 6 | 4 | 3 | 6.2 | completed |
| IMG_6805.MOV | 10.3 | 21 | 1 | 0 | 0 | 2.0 | completed |

## Preservation

Model checksums unchanged after benchmark: True

Previous training reports and application configuration were preserved. Evidence is in evidence.html; detailed settings, jobs and records are in results.json. Media and benchmark.db belong only to this test.
