# Confidence Settings Report

## Overview

The SafeRide Analysis Console includes runtime detection settings that control how strict the computer vision pipeline should be when detecting motorcycles, helmets, no-helmet riders, and license plates.

These settings are exposed in the frontend Settings panel and sent to the backend through:

```http
GET /api/settings
PATCH /api/settings
```

The settings are stored in backend process memory. They apply to new video analysis jobs started after the settings are saved.

## Settings Panel Fields

| Setting | Backend Field | Purpose |
|---|---|---|
| Object confidence | `object_confidence` | Minimum YOLO confidence for general object detections such as person, motorcycle, car, bus, and truck. |
| Helmet confidence | `helmet_confidence` | Minimum YOLO confidence for helmet model detections: `With Helmet` and `Without Helmet`. |
| Plate confidence | `plate_confidence` | Minimum YOLO confidence for license plate detections. |
| Sample interval | `sample_every_seconds` | How often the backend samples video frames for YOLO analysis. |
| Max violations | `max_violations_per_video` | Maximum number of saved violation records for one video job. |
| OCR | `enable_ocr` | Enables or disables EasyOCR reading on saved license plate crops. |

## Confidence Slider Bounds

The backend validates confidence settings before applying them:

```text
minimum = 0.05
maximum = 0.95
```

The frontend displays these values as percentages. For example:

```text
0.35 = 35%
0.70 = 70%
```

## How Confidence Works

YOLO returns detections with confidence scores. A confidence score is the model's estimate that a detected box contains the predicted class.

For example:

```text
motorcycle 0.82
person 0.76
without helmet 0.61
license plate 0.44
```

Each confidence setting acts as a minimum threshold. If the model's detection confidence is below the threshold, the detection is discarded before later association and tracking logic uses it.

Example with `helmet_confidence = 0.50`:

```text
without helmet 0.61 -> kept
without helmet 0.42 -> discarded
```

## Object Confidence

`object_confidence` is used by the general YOLO object detector.

It controls detections for:

- person
- motorcycle
- car
- bus
- truck

The pipeline uses these detections to identify riders, motorcycles, and nearby non-motorcycle vehicles.

Lower object confidence:

- Finds more people and motorcycles.
- Can help with small, far, blurry, or partially occluded riders.
- Can introduce more false detections.
- Can make rider-to-motorcycle association noisier.

Higher object confidence:

- Keeps only stronger person and motorcycle detections.
- Reduces false detections.
- Can miss small or distant motorcycles.
- Can cause true no-helmet detections to be rejected if no motorcycle is confidently detected.

## Helmet Confidence

`helmet_confidence` is used by the helmet YOLO model.

It controls detections for:

- `With Helmet`
- `Without Helmet`

This is the most important confidence setting for helmet violation sensitivity.

Lower helmet confidence:

- Increases sensitivity to possible no-helmet riders.
- Can catch more difficult cases.
- May increase false positives.
- May save more evidence for manual review.

Higher helmet confidence:

- Requires stronger helmet/no-helmet evidence.
- Reduces false positives.
- May miss uncertain no-helmet riders.
- Can make the demo look cleaner but less sensitive.

## Plate Confidence

`plate_confidence` is used by the license plate YOLO model.

It controls which plate boxes are available for association and OCR.

Lower plate confidence:

- Increases the chance of capturing a plate crop.
- Can include blurry or incorrect plate candidates.
- May reduce OCR quality if poor crops are selected.

Higher plate confidence:

- Keeps only stronger plate detections.
- Can improve crop quality.
- May miss small, angled, occluded, or motion-blurred motorcycle plates.

Plate confidence does not decide whether a helmet violation exists. A no-helmet violation can still be saved without a readable plate if the rider and motorcycle association is valid.

## Confidence vs Association Score

The confidence sliders are not the only decision rules.

After YOLO detections are produced, SafeRide performs geometry-based association:

```text
person -> motorcycle -> helmet/no-helmet -> plate
```

The pipeline scores whether detections plausibly belong together. For example:

- Is the helmet/no-helmet box near the upper part of the person?
- Is the person positioned on or near the motorcycle?
- Is the plate located in a plausible lower motorcycle region?
- Is the plate near-square like a Thai motorcycle plate (`plate_min_aspect`-`plate_max_aspect`), not wide like a car plate?
- Does the plate fit a nearby car/bus/truck better than the motorcycle?
- Is this plate clearly this motorcycle's plate? Plates are assigned one-to-one per frame; a plate whose two candidate motorcycles score within `plate_assignment_margin` is dropped as ambiguous.

This means a high-confidence no-helmet box is not automatically saved as a violation. It must also pass the rider/motorcycle association gates.

## Tracking Votes (since 2026-07-03)

Beyond geometry, a violation now also requires temporal consistency:

- Rider identity is tracked on motorcycle boxes across sampled frames.
- Each sampled frame records a helmet-status vote for the track.
- A violation is saved only when the track has at least `min_no_helmet_votes` (default 2) no-helmet votes and they are not drowned out by with-helmet votes.
- A plate is attached only if it was sighted with the track in at least `plate_min_track_sightings` (default 2) samples.
- Plate OCR text is voted across all of the track's samples, so a plate read consistently the same way beats a single high-confidence misread.

Practical effect: a rider must be observed without a helmet in at least two sampled frames to produce a violation. Adaptive sampling makes this easy to satisfy — as soon as any no-helmet detection appears, sampling densifies (`adaptive_sample_divisor` times per interval for `adaptive_hold_seconds`), so even a rider visible for 1-2 seconds accumulates several votes. Lower `MIN_NO_HELMET_VOTES` to `1` (env var) only if very brief single-frame appearances must be saved.

Helmet boxes are assigned one-to-one to people, so on two-up motorcycles the driver and passenger each contribute their own helmet-status vote — an unhelmeted passenger behind a helmeted driver is detected and reaches review.

Duplicate suppression works on two levels: per-track cooldown (`violation_cooldown_seconds`), and a spatial gate that skips any violation whose motorcycle box overlaps or lies within one bike-size of a violation saved inside the cooldown window, so tracker identity churn cannot double-save the same rider even when the bike is moving fast across the frame.

## Runtime Behavior

When the user clicks `Apply Settings`, the frontend sends the current settings to:

```http
PATCH /api/settings
```

The backend updates the in-memory settings object. These settings are then used when the next upload job starts.

Important behavior:

- Settings apply to future jobs.
- Settings are disabled while a job is active.
- Settings reset to config or environment defaults when the backend restarts.
- Settings are not currently persisted in SQLite.

## Current Defaults

Current default values in backend config:

```text
object_confidence = 0.35
helmet_confidence = 0.35
plate_confidence = 0.30
sample_every_seconds = 1
max_violations_per_video = 25
enable_ocr = true
```

Pipeline gates not exposed in the Settings panel (config/env only):

```text
min_no_helmet_votes = 2        # no-helmet samples required per track
plate_min_aspect = 0.55        # plate box width/height lower bound
plate_max_aspect = 2.0         # rejects wide car plates
plate_horizontal_slop = 0.22   # max plate offset vs motorcycle width
plate_assignment_margin = 0.04 # ambiguity margin between candidate motorcycles
plate_min_track_sightings = 2  # plate must co-travel with the track
helmet_crop_inference = true   # helmet model runs on rider crops
helmet_crop_imgsz = 640        # inference size per rider crop
adaptive_sampling = true       # densify sampling after a no-helmet detection
adaptive_sample_divisor = 5    # dense mode samples 5x per interval
adaptive_hold_seconds = 2.5    # how long dense mode holds after a detection
realtime_preview = false       # no pacing; process at full speed
```

These defaults are tuned for a demo-style balance between recall and precision. They are not final accuracy settings. Validate any change with `scripts/evaluate.py` against labeled clips instead of eyeballing.

## Recommended Tuning Strategy

For initial real-world testing:

```text
object confidence: 35%
helmet confidence: 35%
plate confidence: 30%
sample interval: 1 second
```

If the system misses no-helmet riders:

- Lower helmet confidence first.
- Lower object confidence if riders or motorcycles are not detected.
- Lower sample interval to analyze more frames.

If the system produces too many false no-helmet violations:

- Raise helmet confidence.
- Raise object confidence if false person/motorcycle detections are involved.
- Keep manual review enabled and mark false positives.

If plate crops are often missing:

- Lower plate confidence.
- Keep OCR enabled.
- Check whether the plate box is visible in the replay overlay.

If OCR text is poor:

- Do not rely only on lowering plate confidence.
- Poor OCR often comes from blur, low resolution, angle, or bad crop quality.
- Better plate model tuning and Thai plate post-processing may help more than threshold changes.

## Example Scenarios

### More Sensitive Detection

Use this when the system misses riders:

```text
object confidence = 0.25
helmet confidence = 0.25
plate confidence = 0.20
sample interval = 0.5
```

Expected effect:

- More detections.
- More saved evidence.
- More false positives to review.
- Slower processing because more frames are analyzed.

### Cleaner Demo Detection

Use this when the system is too noisy:

```text
object confidence = 0.45
helmet confidence = 0.50
plate confidence = 0.40
sample interval = 1
```

Expected effect:

- Fewer detections.
- Fewer false positives.
- Higher chance of missing uncertain riders.

### Faster Processing

Use this when processing speed matters more than recall:

```text
sample interval = 2
```

Expected effect:

- Fewer YOLO-analyzed frames.
- Faster processing.
- Higher chance of missing short or fast-moving violation moments.

## Important Interpretation Notes

- Confidence is model certainty, not legal certainty.
- A lower threshold does not improve model accuracy; it only allows weaker detections through.
- A higher threshold does not guarantee correctness; it only filters out lower-confidence detections.
- Detection confidence and OCR confidence are separate concepts.
- Plate detection confidence says the crop may contain a plate.
- OCR confidence says EasyOCR is confident in the text it read from the crop.
- The final violation decision depends on detection confidence, association geometry, tracking, cooldown, and plate aggregation.

## Recommended Future Improvements

- Persist runtime settings in the database.
- Add named presets such as `Sensitive`, `Balanced`, and `Strict`.
- Store the settings used for each job so results are reproducible.
- Show per-job settings on the replay and violation review pages.
- Add threshold evaluation tools against a labeled dataset.
- Add precision, recall, and F1 score reports for each threshold combination.
