# Optional replay enhancement

On a completed video's replay page, choose **Enhance playback**. The original
replay remains available during processing. When finished, **Smooth playback:
On/Off** compares the estimated motion with the normal sampled overlay.

Smoothing covers both helmeted and unhelmeted riders. Use the fullscreen
button in the video's upper-right corner to expand the video together with
its detection overlay. The selected smoothing mode continues in fullscreen;
overlay sizing updates when entering or leaving fullscreen.

This is an on-demand pass over the saved video, not a change to detection
sampling. It also works on existing recordings. Sparse Lucas–Kanade optical
flow runs at approximately 10 updates/second on images capped at 640 pixels on
the longest side. Feature consistency, forward/backward error and transform
checks reject unreliable motion. Each segment is checked against the next
sample's observed boxes. Estimated boxes without a supported endpoint are
hidden; original detection anchors remain available. Small heads and plates
can disappear between samples when they lack enough image features.

The output is `<job_id>_playback.json`, separate from `_detections.json`.
Visual identities are local to each sampled interval. They never modify
detection track IDs, helmet votes, violations, or evidence. Review navigation,
sample counts and manual reporting still use the original observations.
The enhancement service has no dependency on violation storage or detection
model inference. Only one enhancement runs at a time in the single-server
process; starting one is refused while a detection job has reported activity
within five minutes. This avoids old interrupted jobs blocking replay forever.
This guard is not a resource scheduler: newly started detections can overlap
an already running enhancement.

Endpoints: GET `/api/jobs/{job_id}/playback` returns status and, on completion,
frames. POST to that path starts the optional background pass. Failures leave
normal replay usable. Deleting a job uses the existing job-media cleanup to
remove its playback artifact as well.

Measured locally on the existing recordings: IMG_7084 added 17.53 seconds,
IMG_7090 added 10.20 seconds, and IMG_7075 added 11.38 seconds. These are
single-run enhancement costs, not detection runtimes or general benchmarks.
All three retained their single violation; database and original detection
file SHA-256 hashes were identical before and after each pass. This does not
repair duplicate violations already present in an older result.

Validation includes synthetic translation, texture loss, missing endpoint,
changed detection IDs, original-data immutability and frontend interpolation
using only visual identities. The original overlay is still useful for
checking estimated positions against actual model observations.
