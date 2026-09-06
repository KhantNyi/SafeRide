# SafeRide user FAQ

Practical answers for uploading footage, viewing analysis, reviewing violations, and using live cameras.

Prepared 6 September 2026 against the current workspace implementation. Administrator settings can change defaults. Older results may retain labels or screenshots generated before recent improvements.

## Getting started and uploading

### 1. What does SafeRide detect?

SafeRide looks for motorcycle occupants who appear not to be wearing a helmet. It links detections across sampled video frames and saves candidate violations for human review. When possible, it also captures a license plate crop and reads its text.

A saved alert is a review candidate, not an automatic final decision.

### 2. Does selecting a video mean it has been uploaded and analyzed?

No. The thumbnail, filename, and “Ready to upload” message show that you selected a file. Start the upload/analysis action and check the job status to see whether processing has started or finished.

### 3. What video files can I use?

The upload screen accepts video files and advertises MP4, MOV, and camera exports up to 500 MB. Successful analysis and browser playback also depend on the codec inside the file. A file extension alone does not guarantee compatibility.

If a clip cannot be played or decoded, try a browser-compatible H.264 MP4 export. Keep the original footage for reference.

### 4. Why is the thumbnail blank even though I selected a file?

The browser may still be loading a frame, the opening scene may be dark, or the browser may not support the video's codec. Check the filename and upload status too. A missing thumbnail alone does not prove that backend analysis will fail.

### 5. Why does my portrait video appear sideways?

Some cameras store rotated pixels together with orientation metadata. SafeRide has an automatic orientation setting. If the result looks wrong, check that setting before running another analysis. Changing it does not rewrite an existing result.

### 6. What do queued, processing, completed, and failed mean?

| Job status | Meaning |
|---|---|
| Queued | The job was created and is waiting to begin processing. |
| Processing | The backend is handling the footage. |
| Completed | The job ended successfully according to its processing rules. Check its message and result. |
| Failed | Processing encountered a problem. Read the job message for the reason. |

“Completed” does not mean every real violation was found. Processing limits can also affect how much footage is analyzed, so read the completion message rather than relying on the status alone.

### 7. Why does the first analysis take longer?

The backend may need to load detector weights, initialize GPU processing, or initialize plate-text recognition. Later jobs can reuse loaded resources. Clip resolution, scene complexity, sampling, and OCR also affect processing time.

### 8. What should I check if analysis fails or the backend is offline?

Check that the backend is running, refresh the connection indicator, and read the job's error message. For a single failing clip, verify that the source video plays correctly. If the backend was restarted during processing, inspect the job before submitting the clip again; do not assume it resumed automatically.

## Sampling and performance

### 9. Does a sampling value of 0.5 mean half a frame per second?

No. The value is the time between base samples, in seconds:

| Interval | Nominal base analysis rate |
|---|---:|
| 1 | 1 frame per second |
| 0.5 | 2 frames per second |
| 0.25 | 4 frames per second |

A smaller interval means more frequent analysis. The code default is 1 second, but check the current Settings panel for the value applied by your backend.

### 10. Does changing the value in the UI actually change analysis?

Yes, after you click **Apply Settings** and the app confirms that the settings were applied. Set it before starting the next analysis. Editing the field alone does not apply it, and changing settings does not reanalyze existing jobs.

### 11. Are settings saved permanently?

UI-applied settings are held for the current backend session. A backend restart loads configured defaults again. Settings are shared by that backend, so coordinate changes when several people use the same installation.

### 12. Why are there more sampled frames than the base rate suggests?

Adaptive sampling temporarily analyzes additional frames after a recent no-helmet detection. This gives short appearances more opportunities to accumulate evidence. Exact sampling also depends on how the interval maps onto the video's frame rate.

### 13. Will more frequent sampling always improve accuracy?

It can catch brief appearances and provide more evidence, but it cannot guarantee a detection when the rider is obscured, too small, blurred, or consistently misclassified. It also increases processing work. Try a shorter interval when reviewing a missed event, then compare the result rather than assuming it must be better.

### 14. Will more frequent sampling create duplicate violations?

It can expose more opportunities for tracking errors. SafeRide suppresses repeated records for the same motorcycle, but losing and reacquiring a rider can still produce duplicates. More samples do not guarantee either more duplicates or duplicate-free results.

### 15. Is the displayed processing FPS the same as the detection rate or playback FPS?

No. These describe different things:

- **Playback FPS:** how quickly the original video's frames are displayed.
- **Sampling rate:** how often the system analyzes frames.
- **Processing FPS:** a throughput measurement based on processed frames and elapsed time.

A high processing FPS does not mean every frame was passed through the detectors. Remaining-time estimates can change as the scene and workload change.

## Analyzed playback and bounding boxes

### 16. Why is the video smooth while the boxes jump?

The video plays at its original frame rate, while detection positions come from sampled analysis frames. Playback holds one sample's box coordinates and switches to another sample as the video advances. It does not currently interpolate the boxes smoothly between positions.

### 17. Why can a box appear slightly ahead of or behind the rider?

The overlay uses the closest available analyzed timestamp, which can be before or after the displayed video frame. It can reuse a sample up to 1.25 seconds away. A moving rider may therefore have changed position. This is a playback-overlay limitation; it does not by itself mean the detector analyzed the wrong frame.

### 18. How can I inspect a detection more precisely?

Pause playback, use the controls that move between analyzed frames, and open the saved evidence. At an analyzed frame, the displayed coordinates correspond more closely to the image that produced them. Hiding the overlay also helps inspect the original footage without boxes covering details.

### 19. Why does replay start shortly before the evidence moment?

Selecting a violation starts replay shortly before its saved evidence timestamp, providing context as the rider approaches. The first displayed frame is therefore not necessarily the exact evidence frame.

### 20. Does every no-helmet box become a saved violation?

No. The pipeline also checks the association with a motorcycle and accumulates helmet-status votes across the track. A brief or weak no-helmet detection can appear in an overlay without passing the checks needed to save a record.

### 21. Why did older footage show “track 3” when only one motorcycle passed?

Track IDs are internal identifiers, not motorcycle counts. Temporary detections, overlapping boxes, or losing and reacquiring the same motorcycle can consume several IDs. Current user-facing labels hide these numbers, while internal tracking still uses them.

Older saved screenshots can retain the text because it is embedded in the image. Reprocessing generates evidence using the current labels.

## Evidence frames and license plates

### 22. What does the frame number in Violation Review mean?

It is the original video frame selected for the saved rider screenshot. It is not necessarily the first detection or the moment when the system first had enough evidence to confirm a candidate violation.

For example, a rider could first be detected at frame 90, pass the confirmation checks at frame 96, and have the best qualifying screenshot at frame 120. The record would show frame 120.

### 23. How does SafeRide choose that screenshot?

Among qualifying no-helmet frames, it scores visibility away from image boundaries, head size, head-crop sharpness, and detection confidence. It keeps the better-scoring view rather than continually replacing it with the latest frame.

This is a quality heuristic. It does not guarantee that every saved screenshot will be ideal.

### 24. Why is the rider still near the edge in some evidence images?

The record may have been generated before the screenshot-selection improvement, or the available qualifying frames may all have been poor. Clear earlier footage might not have passed the detection and confirmation checks, making it unavailable for selection.

An edge-of-frame screenshot alone does not tell you when the first detection happened.

### 25. Does the plate crop come from the displayed evidence frame?

Not necessarily. SafeRide chooses plate evidence separately across the motorcycle's track. The best rider view might be at frame 120 and the clearest plate at frame 150. The displayed frame number belongs to the rider screenshot; it is not a guarantee of the plate crop's timestamp.

### 26. Why can a violation have no plate crop?

The rider may be visible while the plate is facing away, too small, obscured, or blurred. The system may also reject a plate that is not consistently associated with the motorcycle. A helmet violation can still be saved without a usable plate.

### 27. Why is there a plate image but no recognized text?

Finding a plate and reading its characters are separate tasks. OCR may be disabled, or the image may be too poor for a useful reading. Inspect the crop directly before assuming the text is unavailable to a human reviewer.

### 28. Can I trust the plate text automatically?

Check it against the crop. Recognition can confuse digits and letters, return only part of a plate, or produce text from an incorrectly associated plate. A correct helmet alert does not guarantee that the plate belongs to the same motorcycle.

### 29. What does the confidence percentage mean?

In the violation details, it is the model's no-helmet confidence for the selected evidence detection. It is not an overall accuracy score or a guarantee that the violation, tracking identity, and plate are all correct. Manual reports are shown as human reports instead of model confidence.

## Reviewing and managing results

### 30. Does “No violations detected” mean nobody violated the helmet rule?

No. It means the job has no saved violation records under the applied processing rules. Missed detections, insufficient evidence, or footage limitations are still possible. Replay a clear-result clip if you believe a violation was missed.

### 31. Why are some clips listed separately from violation clips?

The review panel separates jobs with saved violations from completed jobs with no saved violations so you can scan the results quickly. This grouping is based on records, not an independent human certification that a clip is violation-free.

### 32. What do Pending, Confirmed, and False positive mean?

| Review status | Meaning |
|---|---|
| Pending | The saved model alert has not received a final review decision. |
| Confirmed | A reviewer has accepted the alert as a violation. |
| False positive | A reviewer has marked the alert as incorrect. |

These are review decisions, separate from job processing status and evidence frame number. “Confirmed” does not automatically verify the OCR text.

### 33. What does “Confirm Pending” do?

It marks the pending records in that job group as confirmed after confirmation from the user. Review the evidence first: this is a bulk review action, not another detection pass.

### 34. What should I do when the system misses a violation?

Open the clip in replay, pause at the relevant moment, and use the missed-violation reporting control. Add a description and plate text if you can read it reliably. The report appears in the review workflow as a manual report.

Any displayed “likely cause” is a diagnostic suggestion based on available analysis, not proof of why the event was missed.

### 35. Does confirming an alert or reporting a miss train the model immediately?

No. Those actions save review information. Model improvement requires a separate workflow to select examples, label them, train updated weights, evaluate them, and deploy the result.

### 36. Do violation numbers tell me how many riders passed?

No. “Violation 1” and “Violation 2” distinguish saved records within a clip. They do not count all passing motorcycles or people. The current event logic generally treats a motorcycle passage with at least one unhelmeted occupant as one event, so a driver and passenger do not necessarily produce two records.

### 37. Why are there two records for what looks like the same rider?

The tracker can lose and reacquire a motorcycle or create separate identities from overlapping detections. Duplicate suppression reduces this, but does not eliminate it. Compare the motorcycle, occupants, and replay before treating the records as separate events. There is currently no dedicated duplicate-review category.

### 38. Does marking a record false positive delete its evidence?

No. It records a review decision and retains the record for inspection. Deleting a record is a separate action that removes it and its associated evidence files.

### 39. What does “Clear Records” remove?

The review panel's Clear Records action deletes all previous jobs, violation records, and associated stored media after confirmation. It is broader than clearing a filter or deleting the currently displayed row. The app does not provide an undo action.

Deleting a single job also removes its stored footage and related generated media. Export or back up anything you need before deletion. The original file outside SafeRide's storage is separate from the uploaded copy.

### 40. What is included in a CSV export?

Exports contain record details such as the per-clip number, job, plate text, status, confidence, evidence frame, and evidence-image link. Internal track IDs are omitted in the current export. CSV files do not embed images or videos; links still depend on the evidence remaining available at the app's address.

### 41. Why did a record disappear after I changed filters?

Review-status filters and search can hide records without deleting them. Return to All and clear the search before assuming a record is missing. Also check whether the clip is in the section for jobs with no saved violations.

## Live monitoring

### 42. Which camera does webcam device 0 use?

It selects a camera attached to the computer running the backend, usually that computer's default camera. It does not select the camera on a phone or another device merely because that device opened the website.

### 43. Can I use an IP camera?

The Live Monitor accepts an RTSP stream address. The backend computer must be able to reach the camera and use any required credentials. Camera availability and stream compatibility need to be tested with the actual device.

### 44. Is the live preview's frame rate the detection rate?

No. The preview targets up to 12 FPS by default, while detection follows the configured sampling interval with adaptive sampling. Actual preview and analysis speed depend on the camera and processing capacity.

### 45. Why doesn't a live violation appear immediately?

The system accumulates evidence and collects plate candidates before saving an event. Processing time and UI refresh add delay too. Live alerts should not be expected at the instant a rider first enters the image.

### 46. Why did my live session stop?

Check the completion message. Sessions can end when you press Stop, the source is lost, the configured violation cap is reached, or the session time limit expires. The default time limit is 900 seconds, or 15 minutes. Persistent source loss does not trigger automatic reconnection.

### 47. Is every live session recorded for browser replay?

The app attempts to record frames it reads. Browser-compatible recording depends on encoder availability; a fallback recording may need to be downloaded to play. If no supported encoder is available, the job reports that recording is unavailable. Do not assume a live session is a guaranteed complete archive of every camera frame.

### 48. Can I use this for unattended, continuous street monitoring?

The current feature is better suited to supervised trials. Session limits, disconnect handling, processing lag, and recording continuity need to be validated for the intended camera. The recorded development test used a file as the live source; it does not establish reliable 24/7 RTSP operation or multi-camera capacity.

## Understanding limitations and improvements

### 49. Will the system work equally well from the opposite side of the street?

It may work, but equivalent performance is not established. The new view can change visible faces, helmet shapes, passengers, plates, lighting, and occlusion. Test representative footage from that camera before relying on the earlier results.

### 50. What accuracy should I expect?

There is no single accuracy percentage that covers every camera and setting. On the September development benchmark, v3 with refined tracking found 21 of 25 labeled motorcycle violation events at a 1-second interval and 22 of 25 at 0.25 seconds. One duplicate remained in each run.

Those labels were AI-reviewed, not independently human-validated, and the benchmark does not establish OCR accuracy or live-camera performance. See the [improvement report](model-and-tracking-improvement-report.md) for settings and limitations.

### 51. Do updates improve old results automatically?

Not generally. Playback-interface changes can affect how an existing job is displayed, but changes to detection, tracking, or saved screenshot selection require processing the footage again. Old evidence images retain text and pixels already written into them.

### 52. Does SafeRide currently count all motorcycles or riders passing through?

No. Violation record totals are not traffic totals. Counting all motorcycles, or counting drivers and passengers separately, has been discussed as a possible extension but is not an implemented user-facing feature.

---

This FAQ describes the application workflow, not a guarantee for every recording or camera. For detailed measured results, use the [model and tracking improvement report](model-and-tracking-improvement-report.md). Technical references: [settings](../backend/app/core/config.py), [pipeline](../backend/app/services/pipeline.py), [live processing](../backend/app/services/live.py), and [API routes](../backend/app/api/routes.py).
