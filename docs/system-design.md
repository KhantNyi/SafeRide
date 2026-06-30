# SafeRide System Design

## Overview

SafeRide is a local full-stack computer vision web application for analyzing uploaded motorcycle traffic videos. It detects helmet violations, captures evidence images, attempts license plate OCR, and lets reviewers inspect results through a browser UI.

The current system is designed for a senior-project MVP and local demo workflow. It prioritizes a working end-to-end pipeline, explainable review screens, and local file/database persistence over distributed scale or production security.

## Goals

- Accept uploaded traffic videos from a browser.
- Process videos with YOLO-based object, helmet, and plate detection.
- Detect no-helmet motorcycle riders and save reviewable evidence.
- Preserve original uploaded video playback with synchronized detection overlays.
- Show processing progress, elapsed time, FPS, and ETA during analysis.
- Allow runtime tuning of detection thresholds for subsequent analyses.
- Persist jobs, violations, evidence frames, plate crops, and detection metadata locally.

## Non-Goals

- Real-time CCTV, webcam, or RTSP ingestion.
- Multi-user authentication or role-based access.
- Cloud deployment, horizontal scaling, or distributed workers.
- Formal model evaluation dashboards.
- Production-grade audit logging or retention policies.

## High-Level Architecture

```text
Browser / Next.js frontend
    |
    | REST API, media URLs, detection metadata
    v
FastAPI backend
    |
    | background video processing
    v
OpenCV + Ultralytics YOLO + EasyOCR
    |
    | writes records and generated media
    v
SQLite database + local filesystem
```

## Runtime Components

### Frontend

Location: `frontend/`

Technology:

- Next.js
- React
- TypeScript
- CSS in `frontend/app/globals.css`

Main screens:

- `/upload`: Analysis Console for upload, playback, overlays, telemetry, runtime settings, results, and evidence.
- `/dashboard`: Job history and saved evidence overview.
- `/violations`: Violation review table with plate crops, evidence inspector, CSV export, and review decisions.

Important files:

- `frontend/components/UploadClient.tsx`
  - Upload workflow
  - Live video playback
  - Canvas detection overlays
  - Runtime settings panel
  - Processing telemetry
  - Results and evidence tabs
- `frontend/components/DashboardClient.tsx`
  - Job history
  - Evidence feed
  - Saved playback reopen action
- `frontend/components/ViolationsClient.tsx`
  - Violation review table
  - Plate/evidence preview
  - Review status updates
- `frontend/lib/api.ts`
  - API client and shared frontend types
- `frontend/components/AppShell.tsx`
  - Shared navigation shell

### Backend

Location: `backend/`

Technology:

- FastAPI
- SQLite
- OpenCV
- Ultralytics YOLO
- EasyOCR

Important files:

- `backend/app/main.py`
  - FastAPI application setup
  - CORS
  - Static `/media` mount
  - Database initialization
- `backend/app/api/routes.py`
  - REST API routes
- `backend/app/services/pipeline.py`
  - Video processing pipeline
  - YOLO inference
  - Rider/helmet/motorcycle/plate association
  - ByteTrack-style rider identity integration
  - Violation capture
  - Detection metadata writing
  - Timing telemetry
- `backend/app/services/byte_tracker.py`
  - Local ByteTrack-style tracker for sampled rider association boxes
- `backend/app/services/repository.py`
  - SQLite read/write helpers
- `backend/app/services/storage.py`
  - Upload persistence and media deletion
- `backend/app/services/streaming.py`
  - In-memory MJPEG frame hub for legacy/preview streaming
- `backend/app/core/config.py`
  - Paths, model settings, confidence thresholds, OCR settings
- `backend/app/core/database.py`
  - SQLite schema and migrations

## API Design

Base URL:

```text
http://127.0.0.1:8000/api
```

### Health

```http
GET /api/health
```

Returns backend availability.

### Runtime Settings

```http
GET /api/settings
PATCH /api/settings
```

Settings:

- `object_confidence`
- `helmet_confidence`
- `plate_confidence`
- `sample_every_seconds`
- `max_violations_per_video`
- `enable_ocr`

Runtime settings are in memory. They apply to subsequent jobs and reset to `.env`/defaults when the backend restarts.

### Video Upload

```http
POST /api/videos/upload
```

Input:

- multipart video file

Behavior:

- Saves uploaded video under `data/uploads/`.
- Creates a `jobs` row.
- Starts `process_uploaded_video()` as a FastAPI background task.
- Returns the created job.

### Jobs

```http
GET /api/jobs
GET /api/jobs/{job_id}
DELETE /api/jobs
DELETE /api/jobs/{job_id}
```

Jobs store:

- lifecycle status
- progress
- frame counts
- timing telemetry
- result state
- source video URL
- latest preview image URL

### Detection Metadata

```http
GET /api/jobs/{job_id}/detections
```

Returns sampled-frame detection metadata from `data/metadata/{job_id}_detections.json`.

The frontend uses this metadata to draw synchronized canvas overlays on top of the original uploaded video.

### Stream

```http
GET /api/jobs/{job_id}/stream
```

Returns MJPEG frames from the in-memory frame hub. This remains available, but the primary UI now uses native browser video playback with canvas overlays.

### Violations

```http
GET /api/violations
DELETE /api/violations/{violation_id}
PATCH /api/violations/{violation_id}/review
```

Review statuses:

- `pending`
- `confirmed`
- `false_positive`

## Data Model

### `jobs`

Stores uploaded video analysis state.

Key columns:

- `id`
- `filename`
- `source_path`
- `status`
- `message`
- `progress`
- `current_frame`
- `total_frames`
- `sampled_frames`
- `violation_count`
- `elapsed_seconds`
- `processing_fps`
- `eta_seconds`
- `preview_image`
- `result`
- `created_at`
- `updated_at`

Status values used by the app:

- `queued`
- `processing`
- `completed`
- `failed`

Result values used by the app:

- `processing`
- `violations_detected`
- `no_violations`
- `failed`

### `violations`

Stores detected helmet violation evidence.

Key columns:

- `id`
- `job_id`
- `detected_at`
- `helmet_status`
- `helmet_confidence`
- `plate_text`
- `plate_confidence`
- `evidence_image`
- `plate_image`
- `frame_number`
- `review_status`

## Filesystem Storage

Generated media and metadata live under `data/`.

```text
data/
  uploads/      Original uploaded videos
  previews/     Latest annotated preview image per job
  evidence/     Saved violation evidence frames
  plates/       Saved license plate crops
  metadata/     Sampled-frame detection JSON
```

Models live under `models/`.

```text
models/
  yolo11s.pt
  helmet-yolov8n.pt
  license-plate-yolo11n.pt
```

OCR and model caches live under `.cache/`.

## Video Processing Flow

```text
User uploads video
    -> Backend saves video to data/uploads
    -> Backend creates queued job
    -> Background task starts
    -> OpenCV opens video
    -> Models are loaded lazily
    -> Frames are sampled based on sample_every_seconds
    -> YOLO detects person, motorcycle, car/bus/truck context, helmet/no-helmet, and plate
    -> Rider association links person -> motorcycle -> helmet -> plate
    -> Hard gates reject weak no-helmet rider and implausible plate links
    -> Detection metadata is appended to JSON
    -> Preview frames are annotated and published/saved
    -> No-helmet rider tracks are aggregated briefly for better plate crops
    -> Violations are written to SQLite and data/evidence
    -> Plate crops are written to data/plates
    -> Job telemetry is updated throughout processing
    -> Job completes with violations_detected or no_violations
```

## Computer Vision Pipeline

The pipeline uses three model roles:

- General object detector:
  - detects `person`
  - detects `motorcycle`
  - detects `car`, `bus`, and `truck` as negative vehicle context for plate matching
- Helmet detector:
  - detects `With Helmet`
  - detects `Without Helmet`
- Plate detector:
  - detects `License_Plate`

Inference is controlled by runtime and config settings:

- `object_confidence`
- `helmet_confidence`
- `plate_confidence`
- `object_imgsz`
- `helmet_imgsz`
- `plate_imgsz`
- `sample_every_seconds`
- `min_helmet_person_score`
- `min_person_motorcycle_score`
- `min_helmet_motorcycle_score`
- `min_no_helmet_association_score`
- `min_plate_motorcycle_score`

Rider association is geometry-based:

- match person to motorcycle
- match helmet/no-helmet box to upper body
- reject no-helmet evidence unless a plausible motorcycle is linked
- match plate to lower motorcycle region only after location, size, aspect, and lower-region gates pass
- reject plate candidates that score better against nearby car/bus/truck boxes
- score associations and save no-helmet rider evidence only above the minimum association score

The plate-to-helmet fallback is not used for saved no-helmet rider associations because it can attach unrelated car or neighboring-motorcycle plates in dense scenes.

Rider associations are passed through a local ByteTrack-style tracker. The tracker matches high-confidence association boxes first, then uses lower-confidence boxes as a second-stage continuation pass. Stable `track_id` values are written into sampled detection metadata, evidence annotations, saved violation records, the violation detail modal, and CSV exports.

Duplicate suppression is per-job and keyed by the tracked rider identity plus cooldown and plate aggregation windows.

## OCR Design

EasyOCR is used for plate text extraction.

Plate OCR flow:

```text
plate detected
    -> crop plate image
    -> preprocess crop
    -> run EasyOCR if enable_ocr is true
    -> save plate_text when readable
    -> display fallback wording if unreadable or missing
```

Plate display rules:

- OCR text available: show text
- plate crop exists but OCR failed: `Unreadable plate`
- no crop exists: `Plate not captured`

## Frontend Playback And Overlay Design

The Live tab uses native browser video playback:

```text
<video src={source_video}>
<canvas className="detection-overlay">
```

The frontend polls detection metadata while analysis is active. For each video timestamp, it chooses the nearest sampled detection frame and draws:

- person boxes
- motorcycle boxes
- helmet boxes
- no-helmet boxes
- plate boxes
- association guide lines
- no-helmet rider track labels

This design keeps playback smooth because the browser plays the original video directly while detection annotations update at sampled-frame cadence.

## Telemetry Design

Job telemetry is persisted in SQLite and refreshed by polling job endpoints.

Metrics:

- progress percentage
- current frame
- total frames
- sampled frames
- violation count
- elapsed seconds
- processing FPS
- ETA seconds
- result state

`processing_fps` is calculated as processed frames divided by elapsed processing time. `eta_seconds` is estimated from remaining frames and current processing FPS.

## Runtime Settings Design

The Analysis page includes a runtime Settings panel. It updates backend process memory through `PATCH /api/settings`.

Design choices:

- Settings are bounded by Pydantic validation.
- Settings are disabled while a job is active.
- Settings apply to future jobs, not the currently running job.
- Settings are not persisted yet.

This is useful for demo tuning and local evaluation without editing `.env` or source files.

## Error Handling And Cleanup

Failure paths:

- Invalid upload content type returns HTTP 400.
- Missing jobs or violations return HTTP 404.
- Processing exceptions mark jobs as `failed`.

Cleanup paths:

- Deleting a job removes:
  - job database row
  - related violation rows
  - upload video
  - preview images
  - evidence images
  - plate crops
  - detection metadata
- Deleting a violation removes:
  - violation database row
  - evidence image
  - plate crop
  - decrements job violation count

File deletion is restricted to known media roots for safety.

## Current Tradeoffs

- FastAPI background tasks are simple and demo-friendly, but not durable if the process exits mid-job.
- SQLite is easy for local use, but not ideal for concurrent production workloads.
- Local filesystem media is straightforward, but lacks retention policy, access control, or object storage semantics.
- Detection is sampled, which improves performance but misses events between sampled frames.
- Overlay metadata is stored as JSON files, which is simple but could become large for long videos.
- Runtime settings are convenient but currently not persisted.
- Rider identity now uses ByteTrack-style tracking, but its thresholds still need tuning on real traffic clips.

## Security And Privacy Notes

Current MVP limitations:

- No authentication.
- No user roles.
- No encryption at rest.
- Uploaded videos and evidence remain on local disk until deleted.
- The `/media` mount serves generated media directly from `data/`.

For production, add authentication, authorization, retention policy, access-controlled media serving, and audit logging.

## Model Improvement Plan

The helmet model currently depends on a public baseline that may not match Thai motorcycle footage. The recommended accuracy path is:

1. Export sampled frames and rider/helmet crops from local traffic videos.
2. Label helmet/no-helmet boxes in YOLO format.
3. Use classes:
   - `0`: `With Helmet`
   - `1`: `Without Helmet`
4. Start with 300-800 labeled frames.
5. Split 70% train, 20% validation, 10% test.
6. Fine-tune from `models/helmet-yolov8n.pt`.
7. Compare validation precision/recall against the current baseline.
8. Replace `models/helmet-yolov8n.pt` only after validation improves.

## Future Work

- Persist runtime settings or add named tuning presets.
- Add timeline markers and jump-to-violation playback controls.
- Add debug export for sampled frames and rider crops.
- Tune ByteTrack thresholds against dense and occluded real traffic clips.
- Improve plate OCR for Thai motorcycle plates.
- Add webcam and RTSP inputs.
- Add PDF/HTML report generation.
- Add metrics and evaluation dashboards.
- Add authentication and production media access control.
