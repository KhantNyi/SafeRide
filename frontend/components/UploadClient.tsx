"use client";

import { DragEvent, FormEvent, type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Clock3,
  FileVideo,
  Gauge,
  Loader2,
  RefreshCcw,
  Save,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Upload
} from "lucide-react";

import {
  API_BASE,
  DetectionBox,
  DetectionFrame,
  DetectionSettings,
  fetchDetections,
  fetchHealth,
  fetchJob,
  fetchSettings,
  fetchViolations,
  Job,
  mediaUrl,
  updateSettings,
  Violation
} from "@/lib/api";

type ConsoleTab = "live" | "results" | "evidence";
type NumericSettingKey = Exclude<keyof DetectionSettings, "enable_ocr">;
const MAX_UPLOAD_MB = 500;
const DEFAULT_SETTINGS: DetectionSettings = {
  object_confidence: 0.35,
  helmet_confidence: 0.35,
  plate_confidence: 0.3,
  sample_every_seconds: 1,
  max_violations_per_video: 25,
  enable_ocr: true
};

export function UploadClient() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [status, setStatus] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<ConsoleTab>("live");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [settingsDraft, setSettingsDraft] = useState<DetectionSettings>(DEFAULT_SETTINGS);
  const [settingsStatus, setSettingsStatus] = useState("");
  const [savingSettings, setSavingSettings] = useState(false);

  async function refreshHealth() {
    setBackendOnline(await fetchHealth());
  }

  async function loadViolations(jobId: string) {
    const records = await fetchViolations();
    setViolations(records.filter((record) => record.job_id === jobId));
  }

  useEffect(() => {
    refreshHealth();
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadSettings() {
      try {
        const settings = await fetchSettings();
        if (!cancelled) {
          setSettingsDraft(settings);
        }
      } catch (error) {
        if (!cancelled) {
          setSettingsStatus(error instanceof Error ? error.message : "Could not load settings");
        }
      }
    }

    loadSettings();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSaveSettings() {
    setSavingSettings(true);
    setSettingsStatus("Saving...");
    try {
      const settings = await updateSettings(settingsDraft);
      setSettingsDraft(settings);
      setSettingsStatus("Settings saved.");
    } catch (error) {
      setSettingsStatus(error instanceof Error ? error.message : "Could not update settings");
    } finally {
      setSavingSettings(false);
    }
  }

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const updated = await fetchJob(job.id);
        setJob(updated);
        setStatus(updated.status === "completed" ? statusForJob(updated) : updated.message ?? statusForJob(updated));
        if (updated.status === "completed" || updated.status === "failed") {
          await loadViolations(updated.id);
        }
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "Could not refresh job status");
      }
    }, 1500);

    return () => window.clearInterval(timer);
  }, [job]);

  useEffect(() => {
    const jobId = new URLSearchParams(window.location.search).get("job");
    if (!jobId) {
      return;
    }
    const selectedJobId = jobId;

    let cancelled = false;
    async function loadSelectedJob() {
      setStatus("Loading saved analysis...");
      try {
        const selectedJob = await fetchJob(selectedJobId);
        if (cancelled) {
          return;
        }
        setJob(selectedJob);
        setFile(null);
        setActiveTab("live");
        setStatus(selectedJob.status === "completed" ? statusForJob(selectedJob) : selectedJob.message ?? statusForJob(selectedJob));
        await loadViolations(selectedJob.id);
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Could not load saved analysis");
        }
      }
    }

    loadSelectedJob();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setStatus("Choose a video file first.");
      return;
    }

    setUploading(true);
    setStatus("Uploading video...");
    setJob(null);
    setViolations([]);
    setActiveTab("live");
    window.history.replaceState(null, "", "/upload");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/api/videos/upload`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail ?? "Upload failed");
      }

      const createdJob = (await response.json()) as Job;
      setJob(createdJob);
      setStatus(createdJob.message ?? "Processing started.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function handleFile(nextFile: File | null) {
    if (!nextFile) {
      setFile(null);
      return;
    }

    if (!nextFile.type.startsWith("video/")) {
      setStatus("Please choose a video file.");
      return;
    }

    const maxBytes = MAX_UPLOAD_MB * 1024 * 1024;
    if (nextFile.size > maxBytes) {
      setStatus(`Video is too large. Keep uploads under ${MAX_UPLOAD_MB} MB.`);
      return;
    }

    setFile(nextFile);
    setStatus(`${nextFile.name} selected.`);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0] ?? null);
  }

  const resultTone = resultToneForJob(job);

  return (
    <div className="console-page">
      <header className="console-header">
        <div>
          <span className="eyebrow">SafeRide</span>
          <h1>Analysis Console</h1>
        </div>
        <div className="console-status">
          <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
          <span>{backendOnline ? "Backend online" : backendOnline === false ? "Backend offline" : "Checking backend"}</span>
          <button className="icon-button" type="button" onClick={refreshHealth} aria-label="Refresh backend status">
            <RefreshCcw size={16} />
          </button>
        </div>
      </header>

      <section className="ops-layout">
        <aside className="source-panel">
          <div className="panel-heading">
            <h2>Source</h2>
            <span className="pill">Video</span>
          </div>

          <form onSubmit={handleSubmit}>
            <label
              className={`compact-drop ${isDragging ? "dragging" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <FileVideo size={26} />
              <span>{file ? file.name : "Drop or select video file"}</span>
              {file ? <small>{formatBytes(file.size)} | Ready to upload</small> : <small>MP4, MOV, or camera exports up to {MAX_UPLOAD_MB} MB</small>}
              <input
                accept="video/*"
                type="file"
                onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <div className="source-actions">
              <button className="button full" type="submit" disabled={uploading || isActive(job) || !file || backendOnline === false}>
                {uploading || isActive(job) ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
                {uploading ? "Uploading" : isActive(job) ? "Analyzing" : "Run Analysis"}
              </button>
              {file ? (
                <button className="button secondary" type="button" onClick={() => handleFile(null)} disabled={uploading || isActive(job)}>
                  Clear
                </button>
              ) : null}
            </div>
          </form>

          <div className="source-message" aria-live="polite">{status || "Ready"}</div>

          <SettingsPanel
            disabled={isActive(job) || savingSettings}
            draft={settingsDraft}
            onChange={setSettingsDraft}
            onSave={handleSaveSettings}
            saving={savingSettings}
            status={settingsStatus}
          />

          <p className="source-hint">Dashboard and Violations stay synced with completed analyses.</p>
        </aside>

        <main className="viewer-panel">
          <div className="viewer-toolbar">
            <div className="tab-list compact" role="tablist" aria-label="Analysis views">
              <button className={activeTab === "live" ? "active" : ""} type="button" onClick={() => setActiveTab("live")}>
                Live
              </button>
              <button
                className={activeTab === "results" ? "active" : ""}
                type="button"
                onClick={() => setActiveTab("results")}
              >
                Results
              </button>
              <button
                className={activeTab === "evidence" ? "active" : ""}
                type="button"
                onClick={() => setActiveTab("evidence")}
              >
                Evidence
              </button>
            </div>
            {job ? <span className={`result-badge ${resultTone}`}>{resultLabel(job)}</span> : null}
          </div>

          {activeTab === "live" ? <LiveTab job={job} /> : null}
          {activeTab === "results" ? <ResultsTab job={job} violations={violations} /> : null}
          {activeTab === "evidence" ? <EvidenceTab job={job} violations={violations} /> : null}
        </main>

        <aside className="telemetry-panel">
          <PanelTitle title="Telemetry" />
          <MetricGrid job={job} />
          <Legend />
          <CurrentJob job={job} violations={violations} />
        </aside>
      </section>
    </div>
  );
}

function SettingsPanel({
  disabled,
  draft,
  onChange,
  onSave,
  saving,
  status
}: {
  disabled: boolean;
  draft: DetectionSettings;
  onChange: (settings: DetectionSettings) => void;
  onSave: () => void;
  saving: boolean;
  status: string;
}) {
  const updateNumber = (key: NumericSettingKey, value: number) => {
    onChange({ ...draft, [key]: value });
  };

  return (
    <section className="settings-panel" aria-label="Detection settings">
      <div className="panel-heading">
        <h2>Settings</h2>
        <SlidersHorizontal size={16} />
      </div>

      <div className="setting-control">
        <label htmlFor="helmet-confidence">Helmet confidence</label>
        <div className="range-row">
          <input
            id="helmet-confidence"
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={draft.helmet_confidence}
            onChange={(event) => updateNumber("helmet_confidence", Number(event.target.value))}
            disabled={disabled}
          />
          <span>{formatPercent(draft.helmet_confidence)}</span>
        </div>
      </div>

      <div className="setting-control">
        <label htmlFor="plate-confidence">Plate confidence</label>
        <div className="range-row">
          <input
            id="plate-confidence"
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={draft.plate_confidence}
            onChange={(event) => updateNumber("plate_confidence", Number(event.target.value))}
            disabled={disabled}
          />
          <span>{formatPercent(draft.plate_confidence)}</span>
        </div>
      </div>

      <div className="setting-control">
        <label htmlFor="object-confidence">Object confidence</label>
        <div className="range-row">
          <input
            id="object-confidence"
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={draft.object_confidence}
            onChange={(event) => updateNumber("object_confidence", Number(event.target.value))}
            disabled={disabled}
          />
          <span>{formatPercent(draft.object_confidence)}</span>
        </div>
      </div>

      <div className="setting-grid">
        <label>
          <span>Sample interval</span>
          <input
            type="number"
            min="0.25"
            max="10"
            step="0.25"
            value={draft.sample_every_seconds}
            onChange={(event) => updateNumber("sample_every_seconds", Number(event.target.value))}
            disabled={disabled}
          />
        </label>
        <label>
          <span>Max violations</span>
          <input
            type="number"
            min="1"
            max="200"
            step="1"
            value={draft.max_violations_per_video}
            onChange={(event) => updateNumber("max_violations_per_video", Number(event.target.value))}
            disabled={disabled}
          />
        </label>
      </div>

      <label className="setting-toggle">
        <input
          type="checkbox"
          checked={draft.enable_ocr}
          onChange={(event) => onChange({ ...draft, enable_ocr: event.target.checked })}
          disabled={disabled}
        />
        <span>OCR</span>
      </label>

      <button className="button secondary full" type="button" onClick={onSave} disabled={disabled}>
        {saving ? <Loader2 className="spin" size={16} /> : <Save size={16} />}
        Apply Settings
      </button>
      {status ? <span className="settings-status">{status}</span> : null}
    </section>
  );
}

function LiveTab({ job }: { job: Job | null }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [detections, setDetections] = useState<DetectionFrame[]>([]);
  const [overlayFrame, setOverlayFrame] = useState<DetectionFrame | null>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number } | null>(null);
  const videoUrl = job?.source_video ? mediaUrl(job.source_video) : null;
  const displaySize = detections[0] ?? videoSize;
  const videoFrameStyle = displaySize
    ? ({
        "--video-aspect-ratio": `${displaySize.width} / ${displaySize.height}`,
        "--video-ratio-number": displaySize.width / displaySize.height
      } as CSSProperties)
    : undefined;

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

    const frame = closestDetectionFrame(detections, video.currentTime);
    setOverlayFrame(frame);
    if (!frame || video.videoWidth <= 0 || video.videoHeight <= 0) {
      return;
    }

    const videoRect = { x: 0, y: 0, width: bounds.width, height: bounds.height };
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
  }, [detections]);

  useEffect(() => {
    setDetections([]);
    setOverlayFrame(null);
    setVideoSize(null);
  }, [job?.id]);

  useEffect(() => {
    if (!job) {
      return;
    }

    const jobId = job.id;
    const jobIsActive = isActive(job);
    let cancelled = false;
    async function load() {
      try {
        const frames = await fetchDetections(jobId);
        if (!cancelled) {
          setDetections(frames);
        }
      } catch {
        if (!cancelled) {
          setDetections([]);
        }
      }
    }

    load();
    if (!jobIsActive) {
      return () => {
        cancelled = true;
      };
    }

    const timer = window.setInterval(load, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.id, job?.status]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay, detections, job?.current_frame]);

  useEffect(() => {
    window.addEventListener("resize", drawOverlay);
    return () => window.removeEventListener("resize", drawOverlay);
  }, [drawOverlay]);

  function handleLoadedMetadata() {
    const video = videoRef.current;
    if (video?.videoWidth && video.videoHeight) {
      setVideoSize({ width: video.videoWidth, height: video.videoHeight });
    }
    window.requestAnimationFrame(drawOverlay);
  }

  return (
    <div className="live-view">
      <div className="preview-stage professional video-preview-stage">
        {videoUrl ? (
          <div className="video-canvas-layer" style={videoFrameStyle}>
            <video
              key={videoUrl}
              ref={videoRef}
              src={videoUrl}
              controls
              muted
              playsInline
              preload="metadata"
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={drawOverlay}
              onPause={drawOverlay}
              onSeeked={drawOverlay}
              onTimeUpdate={drawOverlay}
            />
            <canvas ref={canvasRef} className="detection-overlay" aria-hidden="true" />
          </div>
        ) : (
          <div className="empty-preview">
            <FileVideo size={42} />
            <span>No active preview</span>
          </div>
        )}
      </div>
      {job ? (
        <div className="frame-strip">
          <span>{videoUrl ? "Video playback" : "No video"}</span>
          <div className="progress-meter">
            <span style={{ width: `${clampProgress(job.progress)}%` }} />
          </div>
          <span>
            {overlayFrame ? `Overlay frame ${overlayFrame.frame_number}` : `Analyzed frame ${job.current_frame || "-"}`} | {job.progress.toFixed(1)}% | {job.processing_fps ? `${job.processing_fps.toFixed(1)} FPS` : "FPS -"} | ETA {formatEta(job)}
          </span>
        </div>
      ) : null}
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

function ResultsTab({ job, violations }: { job: Job | null; violations: Violation[] }) {
  if (!job) {
    return <EmptyPanel icon={<Clock3 size={38} />} title="No result selected" text="Select or upload a video." />;
  }

  const hasViolations = job.violation_count > 0 || violations.length > 0;
  const Icon = hasViolations ? ShieldAlert : ShieldCheck;

  return (
    <div className={`result-panel refined ${hasViolations ? "danger" : "clear"}`}>
      <div className="result-icon">
        <Icon size={42} />
      </div>
      <div>
        <span className="eyebrow">{job.filename}</span>
        <h2>{hasViolations ? "Violations detected" : job.status === "completed" ? "No violations detected" : "Analysis running"}</h2>
        <p>{job.message ?? statusForJob(job)}</p>
        <div className="violation-meta">
          <span className={`pill ${job.status}`}>{job.status}</span>
          <span className="pill">{job.sampled_frames} sampled frames</span>
          <span className="pill warning">{job.violation_count} violations</span>
        </div>
      </div>
    </div>
  );
}

function EvidenceTab({ job, violations }: { job: Job | null; violations: Violation[] }) {
  if (!job) {
    return <EmptyPanel icon={<Clock3 size={38} />} title="No evidence selected" text="Select or upload a video." />;
  }

  if (!violations.length) {
    return <EmptyPanel icon={<ShieldCheck size={38} />} title="No saved evidence" text="No no-helmet violations were detected." />;
  }

  return (
    <div className="evidence-grid">
      {violations.map((violation) => (
        <article className="evidence-card" key={violation.id}>
          <img src={mediaUrl(violation.evidence_image)} alt="Saved violation evidence frame" />
          <div>
            <strong>{plateLabel(violation)}</strong>
            <div className="violation-meta">
              <span className="pill failed">{violation.helmet_status.replaceAll("_", " ")}</span>
              <span className="pill warning">Frame {violation.frame_number ?? "-"}</span>
              <span className="pill">{Math.round(violation.helmet_confidence * 100)}%</span>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}

function MetricGrid({ job }: { job: Job | null }) {
  const metrics = useMemo(
    () => [
      { label: "Progress", value: job ? `${job.progress.toFixed(1)}%` : "-" },
      { label: "Frame", value: job?.current_frame || "-" },
      { label: "Elapsed", value: job ? formatDuration(job.elapsed_seconds) : "-" },
      { label: "ETA", value: job ? formatEta(job) : "-" },
      { label: "FPS", value: job?.processing_fps ? job.processing_fps.toFixed(1) : "-" },
      { label: "Samples", value: job?.sampled_frames ?? "-" },
      { label: "Violations", value: job?.violation_count ?? "-" },
      { label: "Result", value: job ? shortResult(job) : "-" }
    ],
    [job]
  );

  return (
    <div className="metric-tiles">
      {metrics.map((metric) => (
        <div className="metric-tile" key={metric.label}>
          <span>{metric.label}</span>
          <strong>{metric.value}</strong>
        </div>
      ))}
    </div>
  );
}

function Legend() {
  return (
    <div className="legend-panel">
      <PanelTitle title="Box Legend" />
      <span><i className="legend-swatch person" />Person</span>
      <span><i className="legend-swatch motorcycle" />Motorcycle</span>
      <span><i className="legend-swatch helmet" />Helmet</span>
      <span><i className="legend-swatch nohelmet" />No helmet</span>
      <span><i className="legend-swatch plate" />Plate</span>
    </div>
  );
}

function CurrentJob({ job, violations }: { job: Job | null; violations: Violation[] }) {
  return (
    <div className="current-summary">
      <PanelTitle title="Current Job" />
      {job ? (
        <>
          <strong>{job.filename}</strong>
          <p>{job.message ?? statusForJob(job)}</p>
          <div className="summary-actions">
            <span className={`pill ${job.status}`}>{job.status}</span>
            <span className="pill">{violations.length} evidence</span>
          </div>
          <Link href="/dashboard" className="button secondary full">
            Open History
          </Link>
          {violations.length ? (
            <Link href="/violations" className="button full">
              Review Violations
            </Link>
          ) : null}
        </>
      ) : (
        <p className="muted">No job selected.</p>
      )}
    </div>
  );
}

function PanelTitle({ title }: { title: string }) {
  return (
    <div className="panel-title">
      <Gauge size={16} />
      <h2>{title}</h2>
    </div>
  );
}

function EmptyPanel({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) {
  return (
    <div className="empty-panel">
      {icon}
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

function isActive(job: Job | null) {
  return job?.status === "queued" || job?.status === "processing";
}

function statusForJob(job: Job) {
  if (job.status === "queued") {
    return "Waiting for detector";
  }
  if (job.status === "processing") {
    return "Scanning sampled frames";
  }
  if (job.status === "completed") {
    return "Analysis completed. Playback is ready.";
  }
  return "Analysis failed";
}

function resultLabel(job: Job) {
  if (job.status === "processing" || job.status === "queued") {
    return "Analyzing";
  }
  if (job.status === "failed") {
    return "Failed";
  }
  return job.violation_count > 0 ? "Violations detected" : "No violations";
}

function shortResult(job: Job) {
  if (job.status !== "completed") {
    return job.status;
  }
  return job.violation_count > 0 ? "Violation" : "Clear";
}

function resultToneForJob(job: Job | null) {
  if (!job || job.status === "queued" || job.status === "processing") {
    return "neutral";
  }
  if (job.status === "failed" || job.violation_count > 0) {
    return "danger";
  }
  return "clear";
}

function clampProgress(progress: number) {
  return Math.max(0, Math.min(100, progress));
}

function plateLabel(violation: Violation) {
  const text = violation.plate_text?.trim();
  if (text) {
    return text;
  }
  return violation.plate_image ? "Unreadable plate" : "Plate not captured";
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatEta(job: Job) {
  if (job.status === "completed") {
    return "Done";
  }
  if (job.status === "failed") {
    return "-";
  }
  if (!job.eta_seconds || job.progress <= 0) {
    return "Estimating";
  }
  return formatDuration(job.eta_seconds);
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0s";
  }

  const totalSeconds = Math.round(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  if (minutes <= 0) {
    return `${remainingSeconds}s`;
  }
  return `${minutes}m ${remainingSeconds.toString().padStart(2, "0")}s`;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

