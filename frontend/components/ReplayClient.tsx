"use client";

import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ArrowLeft, Clock3, Eye, EyeOff, FileVideo, Pause, Play, RefreshCcw, SkipBack, SkipForward } from "lucide-react";

import { DetectionBox, DetectionFrame, fetchDetections, fetchJob, fetchViolations, Job, mediaUrl, Violation } from "@/lib/api";

export function ReplayClient({ jobId }: { jobId: string }) {
  const searchParams = useSearchParams();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const appliedTargetRef = useRef<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [detections, setDetections] = useState<DetectionFrame[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [overlayFrame, setOverlayFrame] = useState<DetectionFrame | null>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number } | null>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const targetFrameParam = searchParams.get("frame");
  const targetFrame = targetFrameParam === null ? null : Number(targetFrameParam);
  const displaySize = videoSize ?? detections[0];
  const videoFrameStyle = displaySize
    ? ({
        "--video-aspect-ratio": `${displaySize.width} / ${displaySize.height}`,
        "--video-ratio-number": displaySize.width / displaySize.height
      } as CSSProperties)
    : undefined;

  const jobViolations = useMemo(() => violations.filter((violation) => violation.job_id === jobId), [violations, jobId]);
  const currentViolation = useMemo(() => closestViolation(jobViolations, overlayFrame?.frame_number ?? null), [jobViolations, overlayFrame]);

  const loadReplay = useCallback(async () => {
    setError(null);
    try {
      const [jobData, detectionData, violationData] = await Promise.all([fetchJob(jobId), fetchDetections(jobId), fetchViolations()]);
      setJob(jobData);
      setDetections(detectionData);
      setViolations(violationData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load replay data");
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const bounds = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(bounds.width * pixelRatio));
    canvas.height = Math.max(1, Math.round(bounds.height * pixelRatio));
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);

    if (!overlayEnabled) {
      setOverlayFrame(null);
      return;
    }

    const frame = closestDetectionFrame(detections, video.currentTime);
    setOverlayFrame(frame);
    if (!frame || video.videoWidth <= 0 || video.videoHeight <= 0) {
      return;
    }

    const videoRect = containedRect(
      bounds.width,
      bounds.height,
      video.videoWidth || frame.width,
      video.videoHeight || frame.height
    );
    const drawBox = (box: DetectionBox, color: string, label: string) => {
      const [x1, y1, x2, y2] = box.xyxy;
      const left = videoRect.x + (x1 / frame.width) * videoRect.width;
      const top = videoRect.y + (y1 / frame.height) * videoRect.height;
      const width = ((x2 - x1) / frame.width) * videoRect.width;
      const height = ((y2 - y1) / frame.height) * videoRect.height;

      context.strokeStyle = color;
      context.lineWidth = 2;
      context.strokeRect(left, top, width, height);
      context.font = "12px Aptos, Segoe UI, sans-serif";
      context.fillStyle = color;
      context.fillText(label, left + 4, Math.max(14, top - 6));
    };

    frame.people.forEach((box) => drawBox(box, "#dc9850", `person ${Math.round(box.confidence * 100)}%`));
    frame.motorcycles.forEach((box) => drawBox(box, "#ff7d2d", `motorcycle ${Math.round(box.confidence * 100)}%`));
    frame.helmets.forEach((box) => drawBox(box, "#12835b", `helmet ${Math.round(box.confidence * 100)}%`));
    frame.no_helmets.forEach((box) => drawBox(box, "#ba3d2d", `no helmet ${Math.round(box.confidence * 100)}%`));
    frame.plates.forEach((box) => drawBox(box, "#006d77", `plate ${Math.round(box.confidence * 100)}%`));

    frame.associations.forEach((association) => {
      drawAssociationLine(context, association.helmet_box, association.motorcycle_box, frame, videoRect, "#ffffff");
      drawAssociationLine(context, association.motorcycle_box ?? association.helmet_box, association.plate_box, frame, videoRect, "#1fd1d1");
      drawTrackLabel(context, association.track_id, association.helmet_box, frame, videoRect);
    });
  }, [detections, overlayEnabled]);

  useEffect(() => {
    loadReplay();
    const timer = window.setInterval(() => {
      if (job?.status === "queued" || job?.status === "processing") {
        loadReplay();
      }
    }, 3000);
    return () => window.clearInterval(timer);
  }, [loadReplay, job?.status]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay, detections, overlayEnabled]);

  useEffect(() => {
    window.addEventListener("resize", drawOverlay);
    return () => window.removeEventListener("resize", drawOverlay);
  }, [drawOverlay]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || targetFrame === null || !Number.isFinite(targetFrame) || detections.length === 0) {
      return;
    }

    const targetKey = `${jobId}:${targetFrame}:${detections.length}`;
    if (appliedTargetRef.current === targetKey) {
      return;
    }

    const timestamp = timestampForFrame(targetFrame, detections);
    if (timestamp === null) {
      return;
    }

    video.currentTime = Math.max(timestamp - 0.75, 0);
    appliedTargetRef.current = targetKey;
    window.requestAnimationFrame(drawOverlay);
  }, [detections, drawOverlay, jobId, targetFrame]);

  const videoUrl = job?.source_video ? mediaUrl(job.source_video) : null;

  function handleLoadedMetadata() {
    const video = videoRef.current;
    if (video?.videoWidth && video.videoHeight) {
      setVideoSize({ width: video.videoWidth, height: video.videoHeight });
    }
    window.requestAnimationFrame(drawOverlay);
  }

  function seekBy(seconds: number) {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    video.currentTime = Math.max(video.currentTime + seconds, 0);
    window.requestAnimationFrame(drawOverlay);
  }

  function jumpToViolation(violation: Violation) {
    if (violation.frame_number === null) {
      return;
    }
    const timestamp = timestampForFrame(violation.frame_number, detections);
    const video = videoRef.current;
    if (timestamp === null || !video) {
      return;
    }
    video.currentTime = Math.max(timestamp - 0.75, 0);
    window.requestAnimationFrame(drawOverlay);
  }

  function jumpDetection(direction: -1 | 1) {
    const video = videoRef.current;
    if (!video || detections.length === 0) {
      return;
    }
    const sorted = [...detections].sort((a, b) => a.timestamp - b.timestamp);
    const next = direction > 0
      ? sorted.find((frame) => frame.timestamp > video.currentTime + 0.1)
      : [...sorted].reverse().find((frame) => frame.timestamp < video.currentTime - 0.1);
    if (next) {
      video.currentTime = next.timestamp;
      window.requestAnimationFrame(drawOverlay);
    }
  }

  return (
    <div className="replay-page">
      <header className="console-header">
        <div>
          <span className="eyebrow">Replay</span>
          <h1>{job?.filename ?? "Analysis Replay"}</h1>
          <p>{job?.message ?? (loading ? "Loading saved analysis..." : "Saved analysis detail")}</p>
        </div>
        <div className="header-actions">
          <Link className="button secondary" href="/dashboard">
            <ArrowLeft size={16} />
            Dashboard
          </Link>
          <button className="button secondary" type="button" onClick={loadReplay}>
            <RefreshCcw size={16} />
            Refresh
          </button>
          <Link className="button" href="/upload">
            New Analysis
          </Link>
        </div>
      </header>

      {error ? <div className="notice danger" role="alert">{error}</div> : null}

      <section className="replay-layout">
        <main className="replay-main">
          <div className="preview-stage professional video-preview-stage">
            {videoUrl ? (
              <div className="video-canvas-layer" style={videoFrameStyle}>
                <video
                  key={videoUrl}
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={handleLoadedMetadata}
                  onPlay={() => {
                    setPlaying(true);
                    drawOverlay();
                  }}
                  onPause={() => {
                    setPlaying(false);
                    drawOverlay();
                  }}
                  onSeeked={drawOverlay}
                  onTimeUpdate={drawOverlay}
                />
                <canvas ref={canvasRef} className="detection-overlay" aria-hidden="true" />
              </div>
            ) : (
              <div className="empty-preview">
                <FileVideo size={42} />
                <span>{loading ? "Loading source video" : "Source video unavailable"}</span>
              </div>
            )}
          </div>

          <div className="replay-controls">
            <button className="icon-button" type="button" onClick={() => seekBy(-2)} aria-label="Back 2 seconds">
              <SkipBack size={17} />
            </button>
            <button
              className="icon-button"
              type="button"
              onClick={() => {
                const video = videoRef.current;
                if (!video) {
                  return;
                }
                if (video.paused) {
                  video.play();
                } else {
                  video.pause();
                }
              }}
              aria-label={playing ? "Pause replay" : "Play replay"}
            >
              {playing ? <Pause size={17} /> : <Play size={17} />}
            </button>
            <button className="icon-button" type="button" onClick={() => seekBy(2)} aria-label="Forward 2 seconds">
              <SkipForward size={17} />
            </button>
            <button className="button secondary" type="button" onClick={() => jumpDetection(-1)}>
              Previous Detection
            </button>
            <button className="button secondary" type="button" onClick={() => jumpDetection(1)}>
              Next Detection
            </button>
            <button className="button secondary" type="button" onClick={() => setOverlayEnabled((enabled) => !enabled)}>
              {overlayEnabled ? <Eye size={16} /> : <EyeOff size={16} />}
              Boxes
            </button>
            <span className="replay-readout">
              {overlayFrame ? `Overlay frame ${overlayFrame.frame_number}` : "No frame overlay"} | {detections.length} analyzed frames
            </span>
          </div>
        </main>

        <aside className="replay-side">
          <section className="replay-summary">
            <h2>Analysis</h2>
            <dl>
              <div>
                <dt>Status</dt>
                <dd>{job?.status ?? "-"}</dd>
              </div>
              <div>
                <dt>Result</dt>
                <dd>{job ? resultLabel(job) : "-"}</dd>
              </div>
              <div>
                <dt>Violations</dt>
                <dd>{jobViolations.length}</dd>
              </div>
              <div>
                <dt>Current Match</dt>
                <dd>{currentViolation ? plateLabel(currentViolation) : "-"}</dd>
              </div>
            </dl>
          </section>

          <section className="replay-violations">
            <div className="section-title">
              <h2>Evidence Moments</h2>
              <span className="pill">{jobViolations.length} records</span>
            </div>
            {jobViolations.length ? (
              <div className="replay-evidence-list">
                {jobViolations.map((violation) => (
                  <button className="replay-evidence-item" type="button" key={violation.id} onClick={() => jumpToViolation(violation)}>
                    <img src={mediaUrl(violation.evidence_image)} alt="Violation evidence frame" />
                    <span>
                      <strong>{plateLabel(violation)}</strong>
                      <small>Frame {violation.frame_number ?? "-"} | Track {violation.track_id ?? "-"}</small>
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty-state compact-empty">
                <Clock3 size={28} />
                <span>No saved violation moments.</span>
              </div>
            )}
          </section>
        </aside>
      </section>
    </div>
  );
}

function closestDetectionFrame(frames: DetectionFrame[], currentTime: number) {
  if (!frames.length) {
    return null;
  }

  let closest = frames[0];
  let smallestGap = Math.abs(currentTime - closest.timestamp);
  for (const frame of frames) {
    const gap = Math.abs(currentTime - frame.timestamp);
    if (gap < smallestGap) {
      closest = frame;
      smallestGap = gap;
    }
  }

  return smallestGap <= 1.25 ? closest : null;
}

function timestampForFrame(frameNumber: number, frames: DetectionFrame[]) {
  if (!Number.isFinite(frameNumber) || !frames.length) {
    return null;
  }

  const exact = frames.find((frame) => frame.frame_number === frameNumber);
  if (exact) {
    return exact.timestamp;
  }

  if (frames.length >= 2) {
    const sorted = [...frames].sort((a, b) => a.frame_number - b.frame_number);
    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    const frameDelta = last.frame_number - first.frame_number;
    const timeDelta = last.timestamp - first.timestamp;
    if (frameDelta > 0 && timeDelta > 0) {
      return first.timestamp + (frameNumber - first.frame_number) * (timeDelta / frameDelta);
    }
  }

  return null;
}

function containedRect(containerWidth: number, containerHeight: number, mediaWidth: number, mediaHeight: number) {
  const containerRatio = containerWidth / containerHeight;
  const mediaRatio = mediaWidth / mediaHeight;
  if (mediaRatio > containerRatio) {
    const width = containerWidth;
    const height = width / mediaRatio;
    return { x: 0, y: (containerHeight - height) / 2, width, height };
  }

  const height = containerHeight;
  const width = height * mediaRatio;
  return { x: (containerWidth - width) / 2, y: 0, width, height };
}

function drawAssociationLine(
  context: CanvasRenderingContext2D,
  fromBox: DetectionBox | null,
  toBox: DetectionBox | null,
  frame: DetectionFrame,
  videoRect: { x: number; y: number; width: number; height: number },
  color: string
) {
  if (!fromBox || !toBox) {
    return;
  }

  const [fromX, fromY] = scaledBoxCenter(fromBox, frame, videoRect);
  const [toX, toY] = scaledBoxCenter(toBox, frame, videoRect);
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(fromX, fromY);
  context.lineTo(toX, toY);
  context.stroke();
  context.fillStyle = color;
  context.beginPath();
  context.arc(toX, toY, 4, 0, Math.PI * 2);
  context.fill();
}

function drawTrackLabel(
  context: CanvasRenderingContext2D,
  trackId: number | null,
  box: DetectionBox | null,
  frame: DetectionFrame,
  videoRect: { x: number; y: number; width: number; height: number }
) {
  if (trackId === null || !box) {
    return;
  }

  const [centerX, centerY] = scaledBoxCenter(box, frame, videoRect);
  context.font = "12px Aptos, Segoe UI, sans-serif";
  context.fillStyle = "#ffffff";
  context.fillText(`track ${trackId}`, centerX + 8, Math.max(14, centerY - 8));
}

function scaledBoxCenter(
  box: DetectionBox,
  frame: DetectionFrame,
  videoRect: { x: number; y: number; width: number; height: number }
) {
  const [x1, y1, x2, y2] = box.xyxy;
  return [
    videoRect.x + (((x1 + x2) / 2) / frame.width) * videoRect.width,
    videoRect.y + (((y1 + y2) / 2) / frame.height) * videoRect.height
  ];
}

function closestViolation(violations: Violation[], frameNumber: number | null) {
  if (frameNumber === null || !violations.length) {
    return null;
  }

  let closest: Violation | null = null;
  let smallestGap = Number.POSITIVE_INFINITY;
  for (const violation of violations) {
    if (violation.frame_number === null) {
      continue;
    }
    const gap = Math.abs(violation.frame_number - frameNumber);
    if (gap < smallestGap) {
      smallestGap = gap;
      closest = violation;
    }
  }

  return smallestGap <= 8 ? closest : null;
}

function resultLabel(job: Job) {
  if (job.status === "queued" || job.status === "processing") {
    return "Analyzing";
  }
  if (job.status === "failed") {
    return "Failed";
  }
  return job.violation_count > 0 ? "Violation" : "Clear";
}

function plateLabel(violation: Violation) {
  const text = violation.plate_text?.trim();
  if (text) {
    return text;
  }
  return violation.plate_image ? "Unreadable plate" : "Plate not captured";
}
