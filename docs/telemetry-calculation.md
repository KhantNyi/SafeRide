# Telemetry Calculation Report

## Overview

The right-hand Telemetry sidebar on the SafeRide Analysis Console shows live job statistics for the currently selected video analysis. These values are produced by the FastAPI backend while `process_uploaded_video()` scans the uploaded video, persisted on the job record in SQLite, returned through the jobs API, and rendered by the Next.js frontend.

Frontend display code:

- `frontend/components/UploadClient.tsx`
- `MetricGrid()`
- `formatEta()`
- `formatDuration()`

Backend calculation code:

- `backend/app/services/pipeline.py`
- `progress_for_frame()`
- `timing_metrics()`
- `update_job()`

## Data Flow

```text
Uploaded video
  -> backend reads frames with OpenCV
  -> selected frames are sampled for YOLO analysis
  -> backend calculates progress, elapsed time, FPS, ETA, samples, and violations
  -> backend writes telemetry fields to the jobs table
  -> frontend polls /api/jobs/{job_id}
  -> Telemetry sidebar renders the latest job values
```

## Sidebar Metrics

| Metric | Source Field | Calculation / Meaning |
|---|---|---|
| Progress | `job.progress` | Percentage of the video frame range already processed. During processing, it is calculated as `current_frame / total_frames * 100` and capped at `99%`. On completion, it is set to `100%`. |
| Frame | `job.current_frame` | The latest frame number read by the backend processing loop. |
| Elapsed | `job.elapsed_seconds` | Wall-clock time since backend video processing started. The frontend formats it as seconds or minutes/seconds. |
| ETA | `job.eta_seconds` | Estimated remaining time. It is calculated as remaining frames divided by current processing FPS. |
| FPS | `job.processing_fps` | End-to-end processing speed, calculated as processed video frames divided by elapsed seconds. |
| Samples | `job.sampled_frames` | Number of frames actually sent through the YOLO analysis path. This is usually much smaller than the total frame count. |
| Violations | `job.violation_count` | Number of saved no-helmet violation records for the current job. |
| Result | derived from `job.status` and `job.violation_count` | Displays the current short state: queued, processing, failed, Violation, or Clear. |

## Backend Formulas

### Progress

The backend calculates progress from the current frame and total frame count:

```text
progress = current_frame / total_frames * 100
```

While the job is still running, progress is capped at `99%` so the UI does not show a complete job before final cleanup and database updates finish. When processing completes, the backend explicitly stores:

```text
progress = 100
```

### Elapsed Time

Elapsed time is based on the monotonic process clock:

```text
elapsed_seconds = now - processing_started_at
```

This is wall-clock processing time for the backend job.

### Processing FPS

Processing FPS is calculated as:

```text
processing_fps = processed_frames / elapsed_seconds
```

This is not raw YOLO inference FPS. It includes the full backend loop: frame reading, sampled inference, tracking, annotation, OCR when enabled, file writing, and database updates.

Since 2026-07-03 this is true hardware throughput. Previously the loop paced itself to the source video's real-time speed, so `processing_fps` was capped at the video's own frame rate (a 30 FPS video always reported ~30 FPS). Pacing now only applies while someone is watching the legacy MJPEG stream, or when `REALTIME_PREVIEW=true` is set. Non-sampled frames are advanced with OpenCV `grab()` (no decode), so FPS values well above the video frame rate are expected and correct.

### ETA

ETA is calculated as:

```text
remaining_frames = total_frames - processed_frames
eta_seconds = remaining_frames / processing_fps
```

If there is not enough information yet, the frontend displays `Estimating`. After completion, the frontend displays `Done`.

### Sample Count

The backend does not analyze every video frame with YOLO. It samples frames using:

```text
analysis_interval = fps * sample_every_seconds
```

For example, if a video is `30 FPS` and `sample_every_seconds` is `1`, the backend analyzes about one frame every 30 frames.

That means:

```text
Samples = number of YOLO-analyzed frames
Frame = latest video frame read
```

These two values are expected to differ.

## Example

For a 30 FPS video with 3,000 total frames and a sample interval of 1 second:

```text
total video duration = 100 seconds
analysis interval = 30 frames
approximate sampled frames = 100
```

The Telemetry sidebar might show:

```text
Frame: 1500
Progress: 50.0%
Samples: 50
```

This means the backend has read halfway through the video, but only 50 sampled frames have been analyzed by YOLO.

## Important Interpretation Notes

- `FPS` means end-to-end analysis speed, not detector-only model speed.
- `FPS` can exceed the video's own frame rate now that real-time pacing is off by default.
- `Samples` means YOLO-analyzed frames, not total frames processed.
- `Violations` counts saved evidence records, not every no-helmet box seen in every sampled frame.
- Duplicate suppression, motorcycle-track identity, and helmet-status voting reduce repeated violation records for the same rider.
- OCR and evidence saving can lower processing FPS because they add extra work after detection.
- ETA is an estimate and can shift when OCR, evidence writes, or dense detection scenes add more processing time.

## Current Limitations

- Telemetry is job-level, not per-model. It does not separately report object detector FPS, helmet detector FPS, plate detector FPS, or OCR latency.
- ETA uses average end-to-end FPS, so it can be less accurate when the video changes from sparse traffic to dense traffic.
- Sample count depends on `sample_every_seconds`, so changing runtime settings changes how many frames YOLO analyzes.
- The UI displays the latest polled job values, so it can lag backend updates by the polling interval.

## Recommended Future Improvements

- Add detector-specific timings:
  - object detection time
  - helmet detection time
  - plate detection time
  - OCR time
- Add sampled-frame FPS separately from full-loop processing FPS.
- Add average, minimum, and maximum processing latency per sampled frame.
- Add a telemetry export for evaluation reports.
- Add a metrics dashboard for total jobs, total violations, plate capture rate, OCR success rate, and duplicate suppression rate.
