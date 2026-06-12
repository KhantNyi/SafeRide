# SafeRide Development Handoff

## Current Status

SafeRide is now a local full-stack computer vision web app for uploaded-video helmet violation analysis. The app supports video upload, YOLO-based frame analysis, ByteTrack-style rider identity tracking, real video playback with detection overlays, job progress tracking, processing timing telemetry, runtime detection settings, result summaries, and evidence persistence.

The project is still an MVP. The web app and backend workflow are functional, but model accuracy, OCR quality, tracker tuning, and production hardening remain future work.

## Progress Log - 2026-05-17

### Accomplished Today

- Performed a site-wide frontend visual cleanup focused on typography, visual weight, and dashboard polish.
- Normalized global font weights in `frontend/app/globals.css` to avoid the previous overuse of bold/semi-bold labels.
- Softened cards, panels, buttons, chips, tables, and shadows so the UI feels cleaner and more professional.
- Removed stale dark standalone Violation page CSS that was no longer used after the page was unified into the shared app shell.
- Verified frontend styling changes with:
  - `npm run typecheck`
  - `npm run build`
  - `GET /dashboard`
  - `GET /violations`
- Started both runtime services for testing:
  - Frontend at `http://127.0.0.1:3000`
  - Backend at `http://127.0.0.1:8000`
- Clarified the current Live preview behavior:
  - It is an MJPEG stream of annotated JPEG frames from the uploaded video.
  - It is not native browser video playback and not a true camera/live feed yet.
  - Detection is sampled, while preview frames reuse the latest analysis between sampled frames.
- Enabled EasyOCR-based plate text reading by default.
- Added OCR preprocessing for plate crops:
  - upscale
  - grayscale
  - bilateral filtering
  - adaptive threshold variant
- Downloaded and cached EasyOCR model weights under `.cache/easyocr/`.
- Added `scripts/backfill_plate_ocr.py` to populate OCR text for existing saved plate crops.
- Backfilled existing violation records:
  - attempted 9 saved plate crops
  - updated 4 records with OCR text
- Updated plate display wording across Analysis, Dashboard, and Violation Review:
  - actual OCR text when available
  - `Unreadable plate` when a crop exists but OCR fails
  - `Plate not captured` when no crop exists
- Added click-to-preview for plate crops in Violation Review.
- Added a focused plate crop modal, separate from the full evidence snapshot modal.
- Updated `handoff.md` during the session so OCR and plate labeling changes are captured.

### Challenges Faced And Tracked

- The app had site-wide visual inconsistency from many competing font weights, heavy chips, heavy table labels, and old override CSS.
  - Fixed by normalizing typography and visual primitives in `frontend/app/globals.css`.
- The Violation page had old dark-mode standalone shell CSS still present even though the current page uses the shared SafeRide shell.
  - Fixed by removing stale `.violation-system`, rail, topbar, hero, and related media-query styles.
- The workspace is not currently a Git repository, so changes could not be reviewed with normal `git diff`/status workflows.
  - Worked around by keeping edits tightly scoped and running checks after each meaningful change.
- The initial backend startup was fine, but restarting a long-running attached Uvicorn session was noisy because the frontend was polling endpoints continuously.
  - Fixed by force-stopping the old backend process and starting a fresh Uvicorn session.
- EasyOCR was installed but its model weights were not cached locally.
  - First OCR initialization attempted a network download and failed inside the sandbox with a socket permission error.
  - Fixed by approving the model download into `.cache/easyocr/`.
- EasyOCR's verbose download progress bar crashed once on Windows console encoding because it printed block characters not supported by the active code page.
  - Fixed by rerunning initialization with `verbose=False`.
- Existing saved plate crops varied heavily in quality.
  - Backfill succeeded on 4 of 9 crops; the rest were too small, blurry, angled, or unreadable.
- OCR works better on car plates than motorcycle plates in current testing.
  - Current rough observed accuracy is around 70% overall.
  - Motorcycle plates remain harder because they are smaller in-frame, blur more, tilt more often, and are frequently occluded.
- The Live tab now uses browser video playback with canvas detection overlays for uploaded videos.
  - It is still not true live camera/RTSP input yet.
- Completed analyses now remain playable in the Live tab instead of immediately switching away to Results.
- Dashboard job rows include a play action that reopens saved jobs through `/upload?job=<job_id>`.
- Processing telemetry now includes elapsed time, processing FPS, and ETA.
- The Analysis Source panel now includes runtime detection settings for confidence thresholds, sample interval, max violations, and OCR.

## Implemented So Far

### Frontend

- Built a Next.js + TypeScript frontend under `frontend/`.
- Added a main Analysis Console at `/upload`.
- Added a Detection Review page at `/dashboard`.
- Added a Violation Review page at `/violations`.
- Added a shared app shell/navigation component with active route states.
- Added backend health indicator.
- Added source/upload panel.
- Added drag-and-drop upload handling, client-side video validation, and selected-file size feedback.
- Added live analysis viewer.
- Reworked the Live tab into actual video playback with synchronized canvas overlays.
- Added tabs:
  - `Live`
  - `Results`
  - `Evidence`
- Added telemetry metrics:
  - progress
  - current frame
  - total frames
  - sampled frames
  - violation count
  - elapsed time
  - processing FPS
  - ETA
  - result state
- Added a runtime Settings panel in the Analysis Source column:
  - object confidence
  - helmet confidence
  - plate confidence
  - sample interval
  - max violations per video
  - OCR on/off
- Added professional UI styling in `frontend/app/globals.css`.
- Added responsive behavior for smaller screens.
- Added Dashboard search and job status filters.
- Added Violation search, status filters, CSV export, and evidence image inspector.
- Unified the Violation page back into the shared SafeRide shell and updated the global font stack.
- Added delete/clear actions for previous jobs and violation records.
- Removed the Recent Jobs panel from Analysis to reduce clutter.
- Replaced ambiguous violation statuses with review-oriented labels: pending, confirmed violation, and false positive.
- Reduced heavy UI font weights that made the app feel overly bold.
- Added visible plate crop previews in Violation Review and the evidence inspector.
- Added persisted review decisions for violations: pending, confirmed, and false positive.
- Added reviewer actions in Violation Review:
  - Confirm violation
  - Mark false positive
- Enabled EasyOCR-based plate text reading for saved plate crops.
- Updated plate labels so unreadable or uncaptured plates are shown clearly instead of as pending.
- Added `scripts/backfill_plate_ocr.py` to populate OCR text for existing saved violation crops.
- Added `/upload?job=<job_id>` support so saved jobs can be reopened for playback from Dashboard.

### Backend

- Built a FastAPI backend under `backend/`.
- Added upload endpoint:
  - `POST /api/videos/upload`
- Added health endpoint:
  - `GET /api/health`
- Added runtime settings endpoints:
  - `GET /api/settings`
  - `PATCH /api/settings`
- Added job endpoints:
  - `GET /api/jobs`
  - `GET /api/jobs/{job_id}`
  - `GET /api/jobs/{job_id}/stream`
  - `GET /api/jobs/{job_id}/detections`
- Added violations endpoint:
  - `GET /api/violations`
- Added cleanup/review endpoints:
  - `DELETE /api/jobs`
  - `DELETE /api/jobs/{job_id}`
  - `DELETE /api/violations/{violation_id}`
  - `PATCH /api/violations/{violation_id}/review`
- Added SQLite persistence in `database/saferide.db`.
- Added schema migration for expanded job fields:
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
- Added schema migration for violation review status:
  - `review_status`
- Added generated media directories:
  - `data/uploads/`
  - `data/previews/`
  - `data/evidence/`
  - `data/plates/`
  - `data/metadata/`

### Computer Vision

- Downloaded baseline models into `models/`:
  - `yolo11s.pt`
  - `helmet-yolov8n.pt`
  - `license-plate-yolo11n.pt`
- Implemented YOLO inference in `backend/app/services/pipeline.py`.
- Current detection classes:
  - COCO `person`
  - COCO `motorcycle`
  - helmet model `With Helmet`
  - helmet model `Without Helmet`
  - plate model `License_Plate`
- Added OpenCV annotation for:
  - person
  - motorcycle
  - helmet
  - no helmet
  - license plate
- Added violation evidence capture when `Without Helmet` is detected.
- Added license plate crop saving when plate detection is available.
- Added EasyOCR plate text reading for detected plate crops. OCR model weights are cached under `.cache/easyocr/`.
- Added MJPEG live stream support with an in-memory frame buffer in `backend/app/services/streaming.py`.
- Pinned YOLO inference image sizes to 960 for object, helmet, and plate models. The helmet model missed obvious no-helmet riders when Ultralytics used the weight file's implicit default size.
- Replaced the old global helmet-to-plate heuristic with per-rider association scoring:
  - person to motorcycle
  - helmet/no-helmet to person upper body
  - plate to motorcycle lower region
- Added ByteTrack-style rider association tracking during each video job so repeated no-helmet detections are suppressed by stable rider track IDs instead of by a single global cooldown.
- Added association guide lines to annotated frames for no-helmet riders so reviewer/debugging evidence shows which helmet, motorcycle, and plate were linked.
- Added `track_id` to sampled detection metadata, evidence annotations, saved violation records, the violation detail modal, and CSV exports.
- Added multi-frame plate aggregation per no-helmet rider association:
  - pending violations wait briefly before saving
  - multiple plate crops can be scored during that window
  - saved records use the best crop/OCR result found for that rider
- Added sampled-frame detection metadata for video overlays:
  - source video URLs are returned as `source_video`
  - metadata is written under `data/metadata/`
  - the frontend overlays boxes and association lines on top of `<video>`
- Added runtime tuning for detection thresholds used by subsequent analyses:
  - `object_confidence`
  - `helmet_confidence`
  - `plate_confidence`
  - `sample_every_seconds`
  - `max_violations_per_video`
  - `enable_ocr`

## Current Architecture

```text
Next.js frontend
    |
    | REST + video media + detection metadata
    v
FastAPI backend
    |
    | Background video processing
    v
YOLO pipeline
    |
    | writes records/media
    v
SQLite + local filesystem
```

Main runtime flow:

```text
Upload video
    -> Create job
    -> Background processor samples detection frames and writes overlay metadata
    -> YOLO detects motorcycle/person/helmet/plate
    -> Uploaded video is played by the browser while canvas overlays metadata
    -> Latest preview is saved to data/previews
    -> Violations are saved to database and data/evidence
    -> Plate crops are saved to data/plates
    -> UI shows Live / Results / Evidence
    -> Processing telemetry shows elapsed time, FPS, and ETA
    -> Runtime settings can tune subsequent analyses
    -> Violation Review supports reviewer decisions and plate crop inspection
```

## Important Files

- `frontend/components/AppShell.tsx`
  - Shared navigation/sidebar shell.
- `frontend/components/UploadClient.tsx`
  - Main analysis console UI, video playback, canvas detection overlay, telemetry, and runtime settings panel.
- `frontend/components/DashboardClient.tsx`
  - Review/history/evidence UI.
- `frontend/components/ViolationsClient.tsx`
  - Violation review table, plate crops, evidence inspector, CSV export, and review decisions.
- `frontend/lib/api.ts`
  - Frontend API client and shared types, including detection settings requests.
- `frontend/app/globals.css`
  - Main UI styling.
- `backend/app/api/routes.py`
  - FastAPI routes, including settings, job, detection metadata, and review endpoints.
- `backend/app/services/pipeline.py`
  - YOLO video processing pipeline, rider association, tracker integration, and sampled-frame metadata writer.
- `backend/app/services/byte_tracker.py`
  - Local ByteTrack-style tracker for sampled rider association boxes.
- `backend/app/services/streaming.py`
  - MJPEG frame hub.
- `backend/app/services/repository.py`
  - SQLite access helpers.
- `backend/app/core/database.py`
  - SQLite schema and migrations.
- `backend/app/core/config.py`
  - Paths and detection settings.
- `scripts/backfill_plate_ocr.py`
  - Utility script for running OCR against existing saved plate crops and updating stored violation records.
- `models/README.md`
  - Model notes.

## How To Run

Backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://localhost:3000/upload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Development Notes

- Prefer running the backend without `--reload` during inference. Reload mode can behave poorly when generated media files are changing.
- If using reload, restrict it to the backend directory:

```powershell
python -m uvicorn app.main:app --reload --reload-dir backend --app-dir backend --host 127.0.0.1 --port 8000
```

- Do not run `npm run build` while `npm run dev` is active. On Windows this has corrupted the Next `.next` dev cache before.
- If Next localhost starts crashing with missing chunk/module errors, stop the dev server and remove `frontend/.next`.
- Runtime settings are in-memory only. They reset to `.env`/defaults when the backend restarts unless persistence is added later.
- `.pt` model files, generated media, database files, venv, cache, and `node_modules` are ignored by git.

## Current Limitations

- Detection is sampled, not every frame. Controlled by `sample_every_seconds` in `backend/app/core/config.py`.
- Live preview plays the original uploaded video and overlays the nearest sampled detection metadata, so boxes update at sampled-frame cadence rather than every video frame.
- Larger inference sizes improve small rider recall but cost more processing time. Controlled by `object_imgsz`, `helmet_imgsz`, and `plate_imgsz`.
- Runtime settings apply to subsequent analyses, not jobs that are already running.
- Runtime settings are not persisted across backend restarts yet.
- OCR is enabled by default, but plate text can still be missing when the crop is too small, blurry, angled, or partially occluded.
- Helmet detection accuracy depends heavily on the public baseline model.
- Rider-to-motorcycle-to-plate association now exists, but it is still geometry based and should be tuned against labeled real traffic clips.
- Plate selection now prefers the best scored crop seen during a short aggregation window for the matched motorcycle, but small/blurred/occluded motorcycle plates can still be missed.
- Duplicate suppression now uses ByteTrack-style rider identities within each job, but tracker thresholds still need tuning against real traffic clips.
- No webcam or RTSP support yet.
- No authentication or role-based access.
- CSV export exists for violations. No PDF/report generator yet.
- No formal evaluation dataset or metrics dashboard yet.

## Areas To Improve

### Accuracy

- Test on real Thai motorcycle traffic videos.
- Tune thresholds with the Analysis Settings panel:
  - `object_confidence`
  - `helmet_confidence`
  - `plate_confidence`
- Build a local helmet/no-helmet dataset from Thai motorcycle footage.
- Fine-tune `models/helmet-yolov8n.pt` on local examples using YOLO-format labels:
  - class `0`: `With Helmet`
  - class `1`: `Without Helmet`
- Suggested dataset target:
  - 300-800 labeled frames to start
  - 70% train, 20% validation, 10% test
  - include side/back views, small/far riders, passengers, blur, night, occlusion, and dense traffic
- Tune rider/motorcycle association and ByteTrack thresholds on real clips.
- Continue reducing duplicate violation records for dense, occluded traffic.
- Evaluate the current helmet model on Thai traffic footage.
- Fine-tune helmet and plate models on Thailand-specific data.

### OCR

- Improve EasyOCR accuracy for motorcycle plate crops.
- Consider PaddleOCR or a plate-specialized OCR model as an alternative baseline.
- Tune the multi-frame crop/OCR aggregation window against real clips.
- Score plate crop candidates by size, sharpness, and angle before OCR.
- Add Thai plate post-processing.
- Save OCR confidence and raw text.
- Compare local OCR against an API baseline if needed.

### Performance

- Install CUDA-enabled PyTorch for RTX 4070 Super acceleration.
- Add frame resizing options.
- Consider a proper task queue if jobs become long-running.
- Avoid loading all models in every process if multiple backend workers are used.

### UX

- Add clear empty states for old jobs with no preview.
- Persist settings or add named tuning presets.
- Add PDF export/report generation for school demo/reporting.
- Consider a dedicated settings page for thresholds, review policy, and OCR.

### Inputs

- Add webcam mode.
- Add RTSP/CCTV stream mode.
- Add image upload mode.
- Add batch video processing.

### Reporting

- Add metrics page:
  - total jobs
  - total violations
  - average FPS
  - no-helmet confidence distribution
  - plate detection rate
- Add project evaluation notebook or script.
- Add confusion matrix support once labeled data exists.

## Suggested Next Session Plan

1. Add a script or debug mode to export sampled frames and rider/helmet crops from uploaded videos into a local dataset folder.
2. Label a first Thai motorcycle helmet dataset in YOLO format with `With Helmet` and `Without Helmet`.
3. Fine-tune the helmet model from `models/helmet-yolov8n.pt` and compare validation metrics against the current baseline.
4. Tune runtime thresholds against 3-5 sample Thailand traffic clips using the Settings panel.
5. Add timeline markers and jump-to-violation controls for completed video playback.
6. Add tracker-based duplicate suppression.
7. Improve motorcycle plate OCR with Thai plate post-processing and better crop scoring.
8. Add PDF/report export for school presentation/reporting.

## Known Good Checks

These checks passed during this session:

```powershell
npm run typecheck
npm run build
.\.venv\Scripts\python.exe -m compileall backend\app
.\.venv\Scripts\python.exe -m compileall scripts\backfill_plate_ocr.py
.\.venv\Scripts\python.exe -m pip check
```

Also verified:

- `http://localhost:3000/upload` returns 200.
- `http://localhost:3000/dashboard` returns 200.
- `http://localhost:3000/violations` returns 200.
- `http://127.0.0.1:8000/api/health` returns `ok`.
- `GET /api/jobs/{job_id}/stream` returns `multipart/x-mixed-replace`.
- `GET /api/settings` returns current runtime detection settings.
- `PATCH /api/settings` accepts bounded runtime detection settings.
- `PATCH /api/violations/{violation_id}/review` persists `review_status`.

## Current Runtime Ports

- Frontend: `http://localhost:3000`
- Backend: `http://127.0.0.1:8000`

## Model Sources

- Ultralytics YOLO11: `models/yolo11s.pt`
- Helmet model: Hugging Face `iam-tsr/yolov8n-helmet-detection`
- License plate model: Hugging Face `morsetechlab/yolov11-license-plate-detection`

Use these in citations for the school report.
