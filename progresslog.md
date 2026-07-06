# SafeRide Progress Log

## 2026-07-06 (later)

### GPU Acceleration (CUDA On Windows, MPS-Ready For macOS)

- Discovered the Windows machine has an RTX 4070 SUPER but the venv carried CPU-only PyTorch; every prior benchmark ran on CPU.
- Installed `torch 2.12.0+cu130` / `torchvision 0.27.0+cu130` in the venv (the backend must be stopped during the reinstall or pip hits locked DLLs and leaves `~orch`-style leftovers in site-packages).
- Added `model_device = "auto"` (`backend/app/core/config.py`): resolved once per process in `resolve_model_device()` - CUDA when available (with FP16 via `half=True`), MPS on Apple Silicon, CPU otherwise. All four YOLO `predict()` calls now pass `device`/`half`.
- OCR follows CUDA availability by default: `ocr_gpu` is now `bool | None`; unset means "use GPU when CUDA exists" (EasyOCR's GPU path is CUDA-only, so Macs stay on CPU).
- README gained a GPU section (CUDA install command, `MODEL_DEVICE` override) and a macOS/Apple Silicon setup section (bash commands, MPS note, webcam permission note).
- Benchmark clip on GPU: 27.4 s wall clock at 91% precision / 83% recall while analyzing more frames than the CPU runs (denser adaptive windows now essentially free). Per-inference is several times faster; remaining wall-clock cost is video decode and OCR, which benefits live sessions most - far more samples per real-time second.

## 2026-07-06

### Kalman/Appearance Tracking, Dedicated Thai Plate OCR, Live Ingestion

Implemented three roadmap items in one pass.

- Tracker (`backend/app/services/byte_tracker.py` rewritten):
  - Per-track constant-velocity Kalman filter over (cx, cy, w, h) with dt-aware transitions, so predictions stay meaningful across irregular sampled/adaptive frame gaps. Noise scales with box height, ByteTrack-style.
  - Appearance matching: motorcycle detections carry an HSV color-histogram feature (`appearance_feature()` in pipeline.py); tracks keep an EMA feature and the match score blends motion with appearance (`tracker_appearance_weight = 0.30`). A small motion floor prevents identity jumps across the frame between similar-looking bikes.
  - Benchmark clip: recall 4/6 -> 5/6 events, 89% precision, all saved records distinct riders (verified by distinct plate texts), spawned track identities down from 111 to 64.
- Dedicated Thai plate OCR (`backend/app/services/plate_ocr.py`, extracted from pipeline.py):
  - EasyOCR restricted to a Thai + Arabic-digit allowlist - Latin junk reads ("allo", "1o") are impossible by construction.
  - Multi-line classification/recombination retained; new character-level voting across a track's samples (`vote_plate_texts()`): per character position, weighted by confidence, so one misread character is outvoted by the samples that read it correctly.
  - OCR confidences on the benchmark clip rose from the 0.25-0.28 range to 0.42-0.56.
  - `read_plate_text` is re-exported from pipeline.py so `scripts/backfill_plate_ocr.py` keeps working.
- Live ingestion (`backend/app/services/live.py` + `/live` frontend page):
  - `POST /api/live/start` accepts a webcam device index or an RTSP URL (credentials stripped from the stored job name); `POST /api/live/{id}/stop` ends the session.
  - Wall-clock sampling with the same RiderTrackManager/violation flow; adaptive densification kept, time-based.
  - Every frame is recorded to `data/uploads/{job_id}.mp4` (avc1 with mp4v fallback) so live sessions replay like uploads; annotated frames publish to the MJPEG hub only while someone is watching.
  - Sessions end on operator stop, source loss (60 consecutive read failures), the violation cap, or `live_max_seconds` (default 900).
  - New Live Monitor page (`frontend/app/live`, `LiveClient.tsx`, nav item): webcam/RTSP source picker, Go Live/Stop, live annotated stream, session telemetry, latest evidence rail, replay link after the session.

### Verification

- Backend py_compile + import check across all changed modules; frontend `tsc --noEmit` and production build pass (new `/live` route present).
- Eval harness on the 62 s benchmark clip: 89% precision, 83% recall (5/6 events) vs estimated labels; all records carry distinct plates - no churn duplicates.
- Live loop tested end-to-end with a file source: session auto-completed on source end (426 frames, recording saved, replayable) and a second session stopped cleanly via the stop endpoint ("Stopped by operator").
- `/live` page screenshot-verified against the running backend.

### Notes

- Live recording uses H.264 when an encoder is available; otherwise mp4v (still saved, but some browsers cannot play it inline).
- Webcam capture opens with CAP_DSHOW on Windows for fast init.

## 2026-07-03 (later)

### Field-Test Fixes: Passengers, Short Riders, Duplicates, Evidence Modal

User testing on 5-6 clips surfaced three issues; all fixed.

- Passenger helmet detection was weak:
  - Root cause: each person independently picked its best helmet box, so on two-up motorcycles both the driver's and passenger's person boxes could claim the driver's helmet, orphaning or mislabeling the passenger's no-helmet box.
  - Fix: `assign_helmets_to_people()` performs greedy one-to-one helmet-to-person assignment; the leftover no-helmet loop can no longer re-claim an already-assigned person. Verified on evidence frames: both riders of a two-up motorcycle now get separate no-helmet boxes.
- Short 1-2 s no-helmet appearances produced no violation:
  - Root cause: the vote gate (2 no-helmet samples) combined with 1 s sampling meant short-lived riders left the frame before accumulating votes.
  - Fix: adaptive sampling (`adaptive_sampling = true`, `adaptive_sample_divisor = 5`, `adaptive_hold_seconds = 2.5`). After any no-helmet detection, sampling densifies 5x for 2.5 s (rolling), so short riders accumulate votes quickly and fast-moving bikes displace little enough between samples for the tracker to hold their identity. Cost is localized to moments with candidate violations.
- Tracker identity churn could double-save the same rider:
  - Detection metadata showed one fast-moving motorcycle displacing ~a full box-size per sampled frame, spawning a fresh track id each sample (tracks 28 -> 7 -> 45 across 20 frames); several of those churned tracks each passed the vote gate and saved.
  - Fix 1: denser adaptive sampling (above) keeps per-sample displacement inside the tracker's match radius.
  - Fix 2: spatial duplicate suppression in `RiderTrackManager` - a violation is skipped when its motorcycle box overlaps (IoU >= 0.35) or sits within 1.0 bike-size of a violation saved inside the cooldown window, regardless of track id. Suppressed saves still record their location, so a moving rider keeps extending the dedup chain.
- Violation evidence modal appeared empty when expanded:
  - Root cause: the page-enter CSS animation kept a persisted `transform` (fill-mode both), which per spec makes the page element the containing block for `position: fixed` descendants - the modal rendered at the top of the page instead of the viewport, off-screen for scrolled users. Media files themselves were verified serving HTTP 200.
  - Fix: page-enter animation is opacity-only in `frontend/app/globals.css`.

## 2026-07-03

### CV Pipeline Accuracy And Speed Overhaul

Implemented the top six items from the pipeline analysis (speed, evaluation, plate association, tracking, helmet accuracy, OCR).

- Speed:
  - Removed real-time pacing by default (`realtime_preview = false` in `backend/app/core/config.py`). Processing now runs at full hardware speed; pacing only applies while an MJPEG stream viewer is connected or `REALTIME_PREVIEW=true`.
  - `FrameHub` (`backend/app/services/streaming.py`) now counts stream viewers; the pipeline skips MJPEG JPEG encoding when nobody is watching.
  - Non-sampled frames are advanced with OpenCV `grab()` instead of `read()`, skipping decode cost.
  - Detection metadata JSON writes are batched to once per second (`metadata_write_seconds`) instead of a full rewrite every sampled frame, plus a final write at completion.
  - Removed redundant `frame.copy()` calls before model inference.
- Evaluation:
  - Added `scripts/evaluate.py`: an offline harness that runs labeled clips through the real pipeline and reports event-level precision, recall, duplicate rate, plate capture rate, and OCR exact-match rate. Example labels in `scripts/eval-labels.example.json`.
  - Added `GET /api/metrics/review` (`review_metrics()` in `backend/app/services/repository.py`): precision from human review decisions, overall, per job, and per helmet-confidence band.
- Plate association (wrong-plate fixes):
  - Aspect-ratio gate tightened to near-square Thai motorcycle plates (`plate_min_aspect = 0.55`, `plate_max_aspect = 2.0`); wide car plates are rejected before scoring.
  - Horizontal slop reduced from 0.28 to `plate_horizontal_slop = 0.22`.
  - New one-to-one greedy plate-to-motorcycle assignment (`assign_plates_to_motorcycles()`): two riders can no longer claim the same plate, and a plate whose two candidate motorcycles score within `plate_assignment_margin = 0.04` is dropped as ambiguous.
  - New co-travel gate: a plate is attached to a violation only if sighted with the rider's track in at least `plate_min_track_sightings = 2` samples (capped by pending sample count), so one-off plates from passing vehicles are dropped.
  - Removed the dead plate-to-helmet fallback code paths.
- Tracking consistency:
  - `AssociationTracker` replaced by `RiderTrackManager`: the ByteTrack-style tracker now runs on raw motorcycle detections instead of gated rider associations, so identities survive helmet-model and association-gate flickers.
  - Track ids propagate from motorcycle tracks onto rider associations.
  - New helmet-status voting per track: a violation requires at least `min_no_helmet_votes = 2` no-helmet observations and `no_helmet * 2 >= with_helmet`, suppressing single-frame flickers while letting mixed driver/passenger tracks reach human review.
- Helmet detection accuracy:
  - Helmet inference now runs on batched rider-focused crops (`helmet_crop_inference = true`, `helmet_crop_imgsz = 640`): motorcycle boxes expanded upward for the rider's head, unioned with overlapping person boxes, merged when intersecting, then one batched model call. Detections are offset back to frame coordinates and deduplicated across overlapping crops.
  - Frames with no motorcycles skip helmet inference entirely.
  - `HELMET_CROP_INFERENCE=false` restores the old full-frame pass.
- OCR quality:
  - Multi-line Thai plate handling: OCR lines are classified by shape (registration prefix / digit group / province) and recombined top-to-bottom instead of trusting the single best line.
  - Plate-format quality scoring picks the best reading across preprocess variants; pattern-conforming readings beat raw high-confidence noise, and 1-character junk reads are rejected.
  - Cross-sample OCR voting per track (`vote_plate_text()`): the text read consistently across samples beats a single high-confidence misread.

### Verification

- `py_compile` across all changed backend modules.
- Ran the eval harness end-to-end on the 62 s IMG_5631 clip (CPU):
  - processing time 62.0 s -> 26.3 s (pacing removed; now compute-bound)
  - `processing_fps` 29.9 -> 70.5
  - saved records 16 -> 3, zero duplicates; two records at frames 1590/1710 share one track id (identity persisted 4 s, previously every record had a fresh id)
  - OCR produced a plate-shaped digit-line read (`9903`) instead of junk (`1o`, `94`)
- Smoke-tested `GET /api/metrics/review` against the live database (overall precision 0.6875 from 16 existing reviews).

### Notes

- Riders visible in fewer than two sampled frames are now intentionally skipped (precision over recall). Restore old behavior with `MIN_NO_HELMET_VOTES=1`, or catch faster riders by lowering `sample_every_seconds`.
- All new gates default from `backend/app/core/config.py` and are env-overridable; tune them against labeled clips with `scripts/evaluate.py` instead of eyeballing.
- Frontend required no changes; the API is backward compatible with one additive endpoint.

## 2026-06-30

### Hardened Rider-Motorcycle-Plate Association

- Tightened the current geometry-plus-ByteTrack association pipeline instead of replacing it.
- Expanded object-model context from person/motorcycle only to include COCO car, bus, and truck boxes as negative vehicle context.
- Added association tuning defaults in `backend/app/core/config.py`:
  - `min_helmet_person_score`
  - `min_person_motorcycle_score`
  - `min_helmet_motorcycle_score`
  - `min_no_helmet_association_score`
  - `min_plate_motorcycle_score`
- Updated no-helmet association saving so pedestrian-only no-helmet detections are rejected unless a plausible motorcycle is linked.
- Added a save-time guard so future weak no-helmet associations still cannot persist without a motorcycle and minimum association score.
- Tightened plate assignment:
  - plate candidates must pass motorcycle-relative location, size, aspect, and lower-region gates before scoring
  - plates that score better against nearby car/bus/truck boxes are rejected
  - the plate-to-helmet fallback is no longer used for saved no-helmet rider associations
- Kept existing ByteTrack-style rider association tracking and multi-frame plate aggregation in place.

### Verification

- Ran backend syntax check:
  - `./.venv/bin/python -m py_compile backend/app/services/pipeline.py backend/app/core/config.py`

### Notes

- This is a stricter MVP association pass. It should reduce pedestrian false positives and wrong car/different-motorcycle plate attachment, but thresholds still need tuning against real Thai traffic clips.
- Future higher-accuracy work can still add rider-local plate crop detection or a trained rider/motorcycle/plate association model.

## 2026-06-21 01:10:00 +07:00

### Professional UI/UX Polish And Playback Framing Fix

- Reworked the frontend visual system in `frontend/app/globals.css` for a cleaner operations-console feel.
- Consolidated styling around consistent surfaces, controls, badges, filters, tables, modals, empty states, and responsive behavior.
- Updated main page hierarchy and copy:
  - `/upload`: Analysis Console
  - `/dashboard`: Operations Dashboard
  - `/violations`: Review Queue
- Added small accessibility improvements for live status/error messages.
- Investigated a playback issue where a detected/uploaded video looked zoomed or showed no meaningful subjects.
- Confirmed the latest uploaded clip was portrait (`504x900`) while the Live viewer was using a fixed stage.
- Updated `frontend/components/UploadClient.tsx` so the Live tab sizes its video/canvas layer from the actual media or detection metadata aspect ratio.
- Updated overlay drawing to map boxes directly onto the aspect-aware video/canvas layer.
- Restarted the frontend dev server after `npm run build` running alongside `npm run dev` caused the local page to appear without CSS.
- Verified the Next CSS asset for `/upload` returned 200 and contained generated CSS.

### Verification

- Ran frontend build:
  - `npm run build`
- Ran frontend type check after build completed:
  - `npm run typecheck`
- Verified runtime services:
  - `http://localhost:3000/upload` returned 200.
  - `http://127.0.0.1:8000/api/health` returned 200 during frontend polling.

### Notes

- Avoid running `npm run build` while `npm run dev` is active on Windows. It can rewrite `.next` assets/types underneath the dev server and temporarily break CSS or typecheck.
- The current app still analyzes uploaded videos, not RTSP/CCTV streams. Real-time CCTV support is feasible, but should be implemented as a separate live ingest/inference pipeline.

## 2026-05-21 00:59:02 +07:00

### Implemented ByteTrack-Style Rider Tracking

- Added `backend/app/services/byte_tracker.py`, a local ByteTrack-style tracker for sampled rider association boxes.
- Integrated the tracker into `backend/app/services/pipeline.py` so rider associations receive stable `track_id` values before:
  - evidence annotations are drawn
  - sampled detection metadata is written
  - duplicate suppression and plate aggregation decide what to save
- Replaced the previous geometry-only association track list with tracker-backed per-rider violation state.
- Added tracker configuration defaults in `backend/app/core/config.py`.
- Added `track_id` persistence for saved violations through SQLite migration, repository writes/reads, and API schemas.
- Added track IDs to frontend detection metadata types, overlay labels, violation detail records, and CSV export.
- Updated `docs/system-design.md` and `handoff.md` so ByteTrack is no longer listed as future work.

### Verification

- Ran backend compile check:
  - `.\.venv\Scripts\python.exe -m compileall backend\app`
- Ran frontend type check:
  - `npm --prefix frontend run typecheck`
- Ran a tracker smoke check confirming a low-confidence second-stage detection continues the same track.
- Ran SQLite migration smoke check confirming `violations.track_id` exists.

## 2026-05-20 00:13:36 +07:00

### Removed Stale Frontend Styles

- Performed a quick cleanup pass for unused code after the current Analysis/Dashboard redesigns.
- Removed orphaned CSS from `frontend/app/globals.css` for older UI pieces that are no longer referenced:
  - old upload/drop-zone panel
  - old current job panel
  - old recent/compact job list
  - old dashboard/page grid helpers
  - old violation card/list layout
  - old Analysis console/grid/sidebar styles
  - old job row/list and progress strip styles
- Left the backend MJPEG stream endpoint and frame hub in place because they remain part of the documented API surface and cleanup lifecycle.

### Verification

- Ran frontend type check:
  - `npm run typecheck`
- Ran backend compile check:
  - `.\.venv\Scripts\python.exe -m compileall backend\app`
- Verified runtime pages:
  - `GET /upload` returned 200.
  - `GET /dashboard` returned 200.
  - `GET /violations` returned 200.

## 2026-05-19 23:56:55 +07:00

### Added System Design Documentation

- Created `docs/system-design.md`.
- Documented the current SafeRide architecture:
  - frontend and backend responsibilities
  - API surface
  - SQLite data model
  - filesystem storage layout
  - video processing flow
  - computer vision pipeline
  - OCR design
  - playback overlay design
  - telemetry and runtime settings design
  - cleanup behavior
  - tradeoffs, security notes, and future work
- Included the model improvement plan for local helmet/no-helmet fine-tuning.

## 2026-05-19 23:32:14 +07:00

### Updated Handoff And Fine-Tuning Plan

- Updated `handoff.md` to reflect the latest completed work:
  - completed analyses remain playable
  - Dashboard playback reopen action
  - processing elapsed/FPS/ETA telemetry
  - runtime detection settings endpoints and UI
- Added current limitations around runtime settings:
  - settings apply only to subsequent jobs
  - settings reset on backend restart until persistence is added
- Replaced outdated next-step notes that still listed ETA and confidence controls as future work.
- Added a concrete local helmet model fine-tuning path:
  - export frames/crops from Thai motorcycle footage
  - label YOLO-format `With Helmet` and `Without Helmet` boxes
  - start with roughly 300-800 labeled frames
  - split 70% train, 20% validation, 10% test
  - fine-tune from `models/helmet-yolov8n.pt`
  - compare validation metrics before replacing the baseline model

## 2026-05-19 23:17:35 +07:00

### Added Runtime Detection Settings Controls

- Added backend settings endpoints:
  - `GET /api/settings`
  - `PATCH /api/settings`
- Added bounded runtime controls for:
  - object confidence
  - helmet confidence
  - plate confidence
  - sample interval
  - max violations per video
  - OCR on/off
- Added a Settings panel to the Analysis Source column with sliders, numeric inputs, an OCR checkbox, and an apply action.
- Updated the frontend API client with detection settings types and requests.
- Settings changes apply to subsequent analyses; the Apply action is disabled while a job is active.

### Verification

- Ran backend compile check:
  - `.\.venv\Scripts\python.exe -m compileall backend\app`
- Ran frontend type check:
  - `npm run typecheck`
- Restarted the backend and verified:
  - `GET /api/settings` returns the current settings.
  - `PATCH /api/settings` accepts and returns settings.
  - `GET /upload` returned 200.

## 2026-05-19 23:09:44 +07:00

### Added Processing Timing Telemetry

- Added persisted job timing fields:
  - `elapsed_seconds`
  - `processing_fps`
  - `eta_seconds`
- Updated the backend video processing loop to calculate elapsed runtime, frame throughput, and estimated remaining time during analysis.
- Added SQLite migrations and API schema/type updates so new and existing jobs can include timing telemetry.
- Updated the Analysis Console telemetry grid with Elapsed, ETA, and FPS.
- Added a compact FPS/ETA readout under the video playback progress strip.

### Verification

- Ran backend compile check:
  - `.\.venv\Scripts\python.exe -m compileall backend\app`
- Ran frontend type check:
  - `npm run typecheck`
- Restarted the backend and verified:
  - `GET /api/jobs` includes the new timing fields.
  - `GET /upload` returned 200.
  - `GET /dashboard` returned 200.

## 2026-05-19 17:05:08 +07:00

### Kept Completed Analyses Playable

- Fixed the Analysis Console lifecycle so completed jobs stay on the Live playback tab instead of immediately switching to Results.
- The final result/evidence records still load in the background after processing finishes.
- Added `/upload?job=<job_id>` support so a saved job can be reopened directly in the Analysis Console with the original uploaded video and overlay metadata.
- Added a play action to each Dashboard job row for reopening saved playback.
- New uploads clear any old `job` query parameter so a fresh run does not accidentally reload a previous job after refresh.

### Verification

- Ran frontend type check:
  - `npm run typecheck`
- Verified runtime pages respond:
  - `GET /upload` returned 200.
  - `GET /dashboard` returned 200.
  - `GET /api/health` returned `ok`.

## 2026-05-18 17:22:43 +07:00

### Improved Rider-Helmet-Plate Association

- Replaced the old global plate matching behavior that picked a plate below or near the strongest no-helmet detection.
- Added per-rider association scoring in `backend/app/services/pipeline.py`.
- New association flow links:
  - person to motorcycle
  - helmet/no-helmet detection to the upper body of the matched person
  - license plate to the lower region of the matched motorcycle
- Updated violation saving so each saved violation uses the associated no-helmet rider and associated plate instead of a single frame-level helmet/plate pair.
- Added lightweight per-job association tracking to reduce duplicate violations for the same rider association.
- Added debug association guide lines in annotated preview/evidence frames for no-helmet riders, showing which helmet, motorcycle, and plate were linked.
- Removed the old unused `choose_plate()` heuristic after the new association path replaced it.

### Verification

- Ran backend compile check:
  - `.\.venv\Scripts\python.exe -m compileall backend\app scripts\backfill_plate_ocr.py`
- Ran a synthetic association smoke check confirming one no-helmet rider links to a motorcycle and plate.

### Files And Code Notes

- `backend/app/services/pipeline.py`
  - `AssociationTracker`: keeps short-lived rider association tracks during one video job and applies per-rider cooldown before saving another violation.
  - `process_uploaded_video()`: now asks `AssociationTracker` which no-helmet associations should be saved, then saves each one separately.
  - `analyze_frame()`: now returns `associations` in addition to raw person, motorcycle, helmet, no-helmet, and plate detections.
  - `associate_riders()`: builds scored rider groups by linking person, motorcycle, helmet/no-helmet, and plate boxes.
  - `best_*` and `score_*` helpers: calculate geometry-based matching scores for helmet-to-person, person-to-motorcycle, and plate-to-motorcycle.
  - `save_violation()`: now saves the associated helmet and plate for the specific no-helmet rider instead of using one global frame-level plate.
  - `draw_association()`: draws guide lines on annotated evidence/preview frames so the linked helmet, motorcycle, and plate are visible during review.
- `handoff.md`
  - Updated the current CV status and limitations so the handoff reflects that association now exists, while full ByteTrack/SORT and multi-frame OCR aggregation are still future work.
- `progresslog.md`
  - Added this dated progress entry to separately track the association improvement and verification steps.

### Remaining Follow-Up

- Tune scoring thresholds against real Thai traffic clips.
- Add full ByteTrack or SORT tracking for stronger cross-frame rider identity.
- Add multi-frame plate crop and OCR aggregation per rider association.

## 2026-05-18 17:28:27 +07:00

### Added Multi-Frame Plate Aggregation

- Changed violation saving so no-helmet rider tracks wait briefly before saving, giving the pipeline several sampled frames to find a better plate crop.
- Added configurable aggregation settings in `backend/app/core/config.py`:
  - `plate_aggregation_seconds`
  - `plate_aggregation_min_samples`
- Extended `AssociationTracker` in `backend/app/services/pipeline.py` to keep pending no-helmet violations per rider association.
- Each pending rider association now stores the best plate candidate seen during the aggregation window.
- Plate candidates are scored by:
  - detector confidence
  - crop size
  - crop sharpness
  - OCR confidence
  - whether OCR produced readable text
- `save_violation()` now writes the best aggregated plate crop and reuses its OCR result instead of OCR-reading only the final save frame.
- Pending violations are flushed at the end of the video so short clips still save detected no-helmet riders.

### Verification

- Ran backend compile check:
  - `.\.venv\Scripts\python.exe -m compileall backend\app scripts\backfill_plate_ocr.py`
- Ran a plate candidate scoring smoke check.
- Ran an `AssociationTracker` smoke check confirming a pending no-helmet rider is saved after the minimum aggregation samples.

### Files And Code Notes

- `backend/app/core/config.py`
  - Added aggregation knobs so the delay/sample count can be tuned without rewriting the pipeline.
- `backend/app/services/pipeline.py`
  - `AssociationTracker`: now manages pending violations, per-track plate candidates, aggregation readiness, and end-of-video flushes.
  - `build_plate_candidate()`: crops the associated plate and OCR-reads it once during candidate collection.
  - `plate_candidate_score()`: ranks candidate crops using detector, image quality, and OCR signals.
  - `crop_sharpness()`: measures blur using Laplacian variance.
  - `save_violation()`: saves the best aggregated plate crop/OCR result for the associated rider.

### Remaining Follow-Up

- Tune `plate_aggregation_seconds` and `plate_aggregation_min_samples` against real sample videos.
- Store association score/track id in the database for easier review and metrics.
- Add full ByteTrack or SORT once geometry-only tracking is no longer enough.

## 2026-05-18 17:36:37 +07:00

### Added Real Video Preview With Detection Overlay

- Replaced the main Live tab behavior from MJPEG image preview to actual browser video playback with a canvas overlay.
- Backend jobs now expose the original uploaded video as `source_video` in job responses.
- Added `GET /api/jobs/{job_id}/detections` to return sampled-frame detection metadata.
- During video processing, `backend/app/services/pipeline.py` writes detection metadata for each sampled frame to `data/metadata/`.
- Detection metadata includes:
  - frame number and timestamp
  - source frame dimensions
  - person, motorcycle, helmet, no-helmet, and plate boxes
  - no-helmet association links for helmet to motorcycle to plate
- The frontend Live tab now uses:
  - `<video>` for real playback controls
  - `<canvas>` for synchronized detection boxes and association guide lines
  - polling while analysis is active so new sampled-frame metadata appears during processing
- The overlay chooses the closest sampled detection frame to the current video time.

### Verification

- Ran backend compile check:
  - `.\.venv\Scripts\python.exe -m compileall backend\app scripts\backfill_plate_ocr.py`
- Ran frontend TypeScript check:
  - `npm run typecheck`

### Files And Code Notes

- `backend/app/core/config.py`
  - Added `metadata_dir` for sampled-frame detection JSON.
- `backend/app/services/pipeline.py`
  - Added `write_detection_metadata()`, `detection_metadata_path()`, and `serialize_detection_frame()`.
  - Writes sampled detection metadata as the video is processed.
- `backend/app/api/routes.py`
  - Added `/api/jobs/{job_id}/detections`.
- `backend/app/services/repository.py`
  - Adds `source_video` to job responses from the stored upload path.
- `frontend/lib/api.ts`
  - Added detection metadata types and `fetchDetections()`.
- `frontend/components/UploadClient.tsx`
  - Replaced the Live tab image stream with video playback and canvas overlay drawing.
- `frontend/app/globals.css`
  - Added styles for the video preview stage and detection overlay.

### Remaining Follow-Up

- Add timeline markers for saved violations.
- Add pause/jump controls for each detected violation.
- Consider persisting detection metadata in SQLite if JSON files become too large.
