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
- `/live`: Live Monitor for webcam/RTSP ingestion with a real-time annotated stream, session telemetry, and latest evidence.
- `/violations`: Violation review table with plate crops, evidence inspector, CSV export, and review decisions.
- `/jobs/{jobId}`: Completed-job replay page with video playback, synchronized overlays, and jump-to-violation controls.

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

The frame hub tracks active stream viewers. When nobody is watching, the pipeline skips MJPEG JPEG encoding and does not pace processing to real time, so jobs run at full hardware speed.

### Live Ingestion

```http
POST /api/live/start
POST /api/live/{job_id}/stop
```

`start` takes `{"source": "0"}` (webcam device index) or `{"source": "rtsp://..."}` and creates a live job that runs the same detection/tracking/violation pipeline in real time. Every frame is recorded to an MP4 under `data/uploads` so completed sessions replay like uploaded jobs, and annotated frames are published to the MJPEG hub for the Live Monitor page. RTSP credentials are stripped from the stored job name.

Live sessions end on operator stop, source loss, the per-session violation cap, or the `live_max_seconds` safety limit (default 900 s). Sampling is wall-clock based; progress/ETA stay at zero while running because the total length is unknown. Recording uses the H.264 (`avc1`) encoder when available and falls back to `mp4v` (which some browsers cannot play inline) otherwise.

### Review Metrics

```http
GET /api/metrics/review
```

Aggregates human review decisions into precision metrics:

- `overall`: total / pending / confirmed / false_positive counts and precision.
- `jobs`: the same buckets per job.
- `confidence_bands`: the same buckets grouped by helmet confidence (`under_50`, `50_to_65`, `65_to_80`, `80_plus`).

Precision is `confirmed / (confirmed + false_positive)`, i.e. how often a saved violation survives human review. It is `null` until at least one record has been reviewed.

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
- `track_id`
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
    -> Non-sampled frames are advanced with grab() (no decode cost)
    -> Frames are sampled based on sample_every_seconds
    -> Sampling densifies temporarily after any no-helmet detection (adaptive sampling)
    -> YOLO detects person, motorcycle, and car/bus/truck context on the full frame
    -> Helmet model runs on batched rider crops around each motorcycle
    -> Plate detector runs on the full frame
    -> Plates are assigned one-to-one to motorcycles (greedy, margin-gated)
    -> Helmet boxes are assigned one-to-one to people (drivers and passengers)
    -> Rider association links person -> motorcycle -> helmet -> plate
    -> Hard gates reject weak no-helmet rider and implausible plate links
    -> Motorcycle boxes are tracked by the ByteTrack-style tracker
    -> Helmet status votes accumulate per motorcycle track
    -> Violations become eligible only after enough no-helmet votes
    -> Detection metadata JSON is written at most once per second
    -> Preview frames are annotated; MJPEG is published only when someone is watching
    -> No-helmet rider tracks are aggregated briefly for better plate crops
    -> Plate crops must co-travel with the track across samples to be attached
    -> OCR readings are voted across the track's samples
    -> Violations are written to SQLite and data/evidence
    -> Plate crops are written to data/plates
    -> Job telemetry is updated throughout processing
    -> Job completes with violations_detected or no_violations
```

Processing runs at full hardware speed by default. Real-time pacing only applies while an MJPEG stream viewer is connected or when `REALTIME_PREVIEW=true` is set.

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

Inference device is selected once per process via `model_device` (default `auto`): CUDA on NVIDIA machines (with FP16), MPS on Apple Silicon, CPU otherwise. OCR follows CUDA availability unless `ocr_gpu` is forced.

Inference is controlled by runtime and config settings:

- `model_device`
- `object_confidence`
- `helmet_confidence`
- `plate_confidence`
- `object_imgsz`
- `helmet_imgsz` (full-frame fallback mode)
- `helmet_crop_inference` / `helmet_crop_imgsz`
- `plate_imgsz`
- `sample_every_seconds`
- `adaptive_sampling` / `adaptive_sample_divisor` / `adaptive_hold_seconds`
- `min_helmet_person_score`
- `min_person_motorcycle_score`
- `min_helmet_motorcycle_score`
- `min_no_helmet_association_score`
- `min_plate_motorcycle_score`
- `min_no_helmet_votes`
- `plate_min_aspect` / `plate_max_aspect` / `plate_horizontal_slop`
- `plate_assignment_margin`
- `plate_min_track_sightings`

### Helmet Crop Inference

The helmet model runs on rider-focused crops instead of the full frame. Each motorcycle box is expanded upward (rider heads sit above the box top), unioned with overlapping person boxes, merged with intersecting neighbor regions, and the resulting crops are batched into one helmet-model call. Detections are mapped back to frame coordinates and deduplicated across overlapping crops (higher confidence wins, label-agnostic, so one head can never be reported as both helmet and no-helmet).

This gives the helmet model several times more effective resolution on distant riders and skips helmet inference entirely on frames with no motorcycles. Set `HELMET_CROP_INFERENCE=false` to restore the old full-frame pass at `helmet_imgsz`.

### Plate Assignment

Plates are assigned to motorcycles one-to-one per sampled frame before rider association runs:

- Hard plausibility gates: plate center inside the expanded motorcycle box, in the lower region, plausible area ratio, near-square aspect ratio (`plate_min_aspect`-`plate_max_aspect`, tuned for Thai motorcycle plates vs the much wider car plates), and horizontal slop capped at `plate_horizontal_slop` of the motorcycle width.
- Each plate picks its single best motorcycle. If the runner-up motorcycle scores within `plate_assignment_margin`, the plate is ambiguous and dropped for that frame.
- Plates that fit a nearby car/bus/truck better than the motorcycle are rejected.
- Greedy one-to-one assignment guarantees two riders can never claim the same plate in one frame.

At violation time there is an additional temporal gate: the plate must have been sighted with the rider's track in at least `plate_min_track_sightings` samples (capped by the pending sample count), so a one-off plate from a passing vehicle is dropped rather than attached.

### Passengers And Multi-Rider Motorcycles

Helmet boxes are assigned to people one-to-one (greedy, by geometric score). On two-up motorcycles the driver's and passenger's heads sit close together; without exclusive assignment both person boxes could claim the driver's helmet and the passenger's own no-helmet box would be orphaned or mislabeled. With one-to-one assignment the driver and passenger each contribute their own helmet-status vote to the shared motorcycle track, so a helmeted driver with an unhelmeted passenger still produces a violation (the vote rule tolerates mixed tracks).

### Adaptive Sampling

Sampling normally follows `sample_every_seconds` (default 1 s). Whenever a sampled frame contains any no-helmet detection, sampling temporarily densifies to `adaptive_sample_divisor` times per interval (default 5x) for `adaptive_hold_seconds` (default 2.5 s), extended while no-helmet detections continue. This serves two purposes: short-lived riders — visible for only 1–2 seconds — accumulate enough helmet votes and plate sightings to be saved, and fast-moving bikes displace little enough between samples that the tracker can hold their identity. The dense-sampling cost is paid only around candidate violations. Disable with `ADAPTIVE_SAMPLING=false`.

### Rider Identity Tracking And Helmet Voting

The ByteTrack-style tracker runs on raw motorcycle detections, not on gated rider associations. Motorcycle boxes are the most stable detection in traffic scenes, so identities survive frames where the helmet model or association gates flicker. The tracker matches high-confidence detections first, then gives unmatched tracks a second chance with lower-confidence detections.

Matching combines two cues:

- **Kalman motion**: each track carries a constant-velocity Kalman filter over (cx, cy, w, h) whose transition step uses the actual frame gap between sampled updates, so predictions stay meaningful under irregular (adaptive) sampling. Noise scales with box height, ByteTrack-style.
- **Appearance**: each motorcycle detection carries an HSV color-histogram feature of its crop; tracks keep an EMA of their feature, and the match score blends motion with appearance similarity (`tracker_appearance_weight`, default 0.30). Appearance can rescue a weak motion match on fast riders between samples, but a small motion floor prevents identity jumps across the frame between similar-looking bikes.

Track ids propagate from motorcycle tracks onto rider associations, and each sampled association records a helmet-status vote for its track. A no-helmet violation becomes eligible only when the track has:

- at least `min_no_helmet_votes` no-helmet observations, and
- no-helmet observations that are not drowned out by with-helmet observations (`no_helmet * 2 >= with_helmet`).

This suppresses single-frame helmet-model flickers on helmeted riders while still letting genuinely mixed tracks (helmeted driver, unhelmeted passenger) through to human review. Adaptive sampling densifies the sample rate as soon as a no-helmet detection appears, so even fast-crossing riders normally accumulate enough votes; lower `MIN_NO_HELMET_VOTES` to `1` if single-sample saving is ever needed.

Stable `track_id` values are written into sampled detection metadata, evidence annotations, saved violation records, the violation detail modal, and CSV exports.

Duplicate suppression is per-job and keyed by the tracked rider identity plus cooldown and plate aggregation windows.

## OCR Design

Plate reading lives in a dedicated module, `backend/app/services/plate_ocr.py`, which specializes EasyOCR for Thai plates:

- The recognizer is restricted to an **allowlist of Thai characters and Arabic digits**, so Latin junk reads ("allo", "1o") are impossible by construction.
- OCR lines are classified by shape (registration prefix / digit group / province) and recombined top-to-bottom; a plate-format quality score ranks readings across three preprocess variants.
- Across a rider's track, readings are **voted per character position**, weighted by confidence: "1กข 1234" read four times beats "1กข 1284" read once — a single misread character is outvoted instead of winning on one lucky confidence score.

Plate OCR flow:

```text
plate detected
    -> crop plate image (best crop aggregated over the track)
    -> preprocess crop (raw, upscaled, adaptive-threshold variants)
    -> run EasyOCR with the Thai plate allowlist if enable_ocr is true
    -> classify and recombine multi-line readings
    -> vote text character-by-character across the track's samples
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

Since real-time pacing was removed (2026-07-03), `processing_fps` reflects true hardware throughput. Before that change it was capped at the source video's own frame rate.

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
- Detection is sampled, which improves performance but misses events between sampled frames. The helmet vote requirement (`min_no_helmet_votes`, default 2) additionally means a rider must appear in at least two sampled frames to be saved — riders crossing the frame in under `2 * sample_every_seconds` are intentionally skipped in favor of precision.
- Overlay metadata is stored as JSON files; writes are batched to once per second, but the file is still rewritten in full each time.
- Runtime settings are convenient but currently not persisted.
- Rider identity is anchored to motorcycle tracks, which is stable but means a rider who switches between detected motorcycles (dense occlusion) can still change identity.
- Helmet crop inference only looks near detected motorcycles; helmet boxes away from any motorcycle are no longer detected or drawn (they could never become violations anyway).

## Security And Privacy Notes

Current MVP limitations:

- No authentication.
- No user roles.
- No encryption at rest.
- Uploaded videos and evidence remain on local disk until deleted.
- The `/media` mount serves generated media directly from `data/`.

For production, add authentication, authorization, retention policy, access-controlled media serving, and audit logging.

## Evaluation

Two measurement tools exist so threshold and pipeline changes can be validated instead of guessed:

### Offline eval harness

`scripts/evaluate.py` runs labeled clips through the real pipeline and reports event-level precision, recall, duplicate rate, plate capture rate, and OCR exact-match rate.

```powershell
python scripts/evaluate.py scripts/eval-labels.example.json
python scripts/evaluate.py my-labels.json --json results.json --keep-jobs
```

The labels file lists clips with the frame ranges of real no-helmet riders (and optionally their plate text). Clips with an empty event list measure false positives on clean footage. Eval jobs run through the normal database and are deleted afterwards unless `--keep-jobs` is passed. Detection settings come from config/env, so threshold sweeps are done with environment variables.

### Review-decision metrics

`GET /api/metrics/review` turns the human decisions already collected on the Violations page into live precision metrics, overall, per job, and per helmet-confidence band. Confirmed and false-positive counts also identify which evidence images are worth exporting as fine-tuning data later.

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
- Add timeline markers on replay playback and refine violation navigation controls.
- Add debug export for sampled frames and rider crops (confirmed/false-positive evidence export for fine-tuning).
- Tune tracker, voting, and plate-gate thresholds against labeled clips using `scripts/evaluate.py`.
- Fine-tune the helmet model on local footage (see Model Improvement Plan).
- Train a dedicated Thai plate recognition model (current OCR is allowlisted + voted EasyOCR).
- GPU inference and OCR (`ocr_gpu`, model device selection).
- Multi-camera live sessions and durable live-job recovery after backend restarts.
- Add PDF/HTML report generation.
- Surface `/api/metrics/review` in the frontend as an accuracy dashboard.
- Add authentication and production media access control.

## Change Log

### 2026-07-06 - Tracker, OCR, And Live Ingestion

1. Tracker upgraded with per-track Kalman motion (dt-aware for sampled frames) and HSV-histogram appearance matching (`tracker_appearance_weight`); on the benchmark clip recall rose from 4/6 to 5/6 events with all-distinct saved riders and far fewer spawned identities.
2. Plate OCR moved to a dedicated `plate_ocr.py` module: Thai + digit allowlist (no Latin misreads possible), multi-line recombination, and character-level voting across each track's samples.
3. Live ingestion: `POST /api/live/start` / `POST /api/live/{id}/stop`, wall-clock sampled real-time pipeline, MP4 session recording for replay, MJPEG annotated live stream, and a new `/live` Live Monitor page in the frontend.

### 2026-07-03 - Accuracy And Speed Overhaul

Backend pipeline changes (see `progresslog.md` for full detail):

1. Real-time pacing removed by default (`realtime_preview` now `false`); processing runs at hardware speed and only paces when an MJPEG viewer is connected.
2. Added the offline eval harness (`scripts/evaluate.py`) and the `GET /api/metrics/review` endpoint.
3. Plate association: near-square aspect gate, tighter horizontal slop, one-to-one greedy plate-to-motorcycle assignment with an ambiguity margin, and a temporal co-travel requirement before a plate is attached to a violation.
4. Rider identity: the ByteTrack-style tracker now runs on raw motorcycle detections instead of gated associations, and helmet status is decided by per-track voting (`min_no_helmet_votes`).
5. Helmet detection now runs on batched rider-focused crops (`helmet_crop_inference`) instead of the full frame.
6. OCR: multi-line Thai plate combination, plate-format quality scoring across preprocess variants, and cross-sample text voting per track.
7. Loop efficiency: `grab()` skipping for undecoded frames, detection-metadata writes batched to once per second, MJPEG encoding skipped when nobody is streaming.

Measured on the same 62 s test clip (CPU): processing time 62.0 s -> 26.3 s, saved records 16 -> 3 with stable track identity and zero duplicates.

### 2026-07-03 - Follow-up Fixes From Field Testing

1. Passenger helmet detection: helmet boxes are now assigned one-to-one to people (`assign_helmets_to_people()`), so passengers get their own helmet observation instead of being shadowed by the driver's helmet; leftover no-helmet boxes can no longer re-claim an already-assigned person.
2. Adaptive sampling (`adaptive_sampling`, default on): sampling densifies 5x for 2.5 s after any no-helmet detection, so riders visible for only 1-2 seconds accumulate enough helmet votes to be saved and fast-moving bikes stay trackable between samples.
3. Spatial duplicate suppression: a violation whose motorcycle box overlaps (IoU >= 0.35) or lies within one bike-size of a violation saved within the cooldown window is skipped regardless of track id, so tracker identity churn on one motorcycle can no longer produce multiple records; suppressed saves still extend the dedup chain for moving riders.
4. Fixed the violation evidence modal opening off-viewport: the page-enter animation kept a persisted `transform`, which made the page the containing block for the `position: fixed` modal; the animation is now opacity-only (`frontend/app/globals.css`).
