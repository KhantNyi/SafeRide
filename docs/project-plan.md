# SafeRide Project Plan

## Objective

Build a full web application that detects motorcycle riders without helmets and captures license plate evidence from Thai traffic footage.

## MVP Scope

- Uploaded video processing
- Backend job queue state stored in SQLite
- Evidence image capture
- Violation dashboard
- Live annotated preview frames with YOLO detections

## CV Upgrade Path

1. Detect motorcycle and person with YOLO.
2. Detect helmet/no-helmet on rider regions.
3. Track riders across frames to avoid duplicate violations.
4. Detect license plate in the motorcycle region.
5. Run Thai-aware OCR on plate crops.
6. Add confidence thresholds and manual review states.

## Evaluation Metrics

- Helmet detection precision, recall, and F1 score
- Plate crop detection accuracy
- OCR character accuracy and full-plate accuracy
- Processing FPS on RTX 4070 Super
- Duplicate violation rate per rider
