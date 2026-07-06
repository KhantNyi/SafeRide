"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Camera, Loader2, PlayCircle, Radio, RefreshCcw, Square, Video } from "lucide-react";

import {
  fetchHealth,
  fetchJob,
  fetchViolations,
  Job,
  liveStreamUrl,
  mediaUrl,
  startLiveAnalysis,
  stopLiveAnalysis,
  Violation
} from "@/lib/api";

type SourceMode = "webcam" | "rtsp";

export function LiveClient() {
  const [mode, setMode] = useState<SourceMode>("webcam");
  const [webcamIndex, setWebcamIndex] = useState("0");
  const [rtspUrl, setRtspUrl] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [status, setStatus] = useState("");
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const streamRef = useRef<HTMLImageElement | null>(null);

  const running = job?.status === "processing" || job?.status === "queued";

  async function refreshHealth() {
    setBackendOnline(await fetchHealth());
  }

  useEffect(() => {
    refreshHealth();
  }, []);

  useEffect(() => {
    if (!job || (job.status !== "processing" && job.status !== "queued")) {
      return;
    }

    const jobId = job.id;
    const timer = window.setInterval(async () => {
      try {
        const updated = await fetchJob(jobId);
        setJob(updated);
        const records = await fetchViolations();
        setViolations(records.filter((record) => record.job_id === jobId));
        if (updated.status === "completed" || updated.status === "failed") {
          setStatus(updated.message ?? "Live session ended.");
        }
      } catch {
        // keep the last known state; health indicator covers backend loss
      }
    }, 1500);

    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  async function handleStart(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const source = mode === "webcam" ? webcamIndex.trim() : rtspUrl.trim();
    if (!source) {
      setStatus(mode === "webcam" ? "Enter a webcam index (usually 0)." : "Enter an RTSP URL.");
      return;
    }

    setStarting(true);
    setStatus("Connecting to live source...");
    setViolations([]);
    try {
      const created = await startLiveAnalysis(source);
      setJob(created);
      setStatus("Live session started.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not start the live session");
    } finally {
      setStarting(false);
    }
  }

  async function handleStop() {
    if (!job) {
      return;
    }
    setStopping(true);
    setStatus("Stopping live session...");
    try {
      await stopLiveAnalysis(job.id);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not stop the live session");
    } finally {
      setStopping(false);
    }
  }

  const streamSrc = useMemo(
    () => (job && running ? `${liveStreamUrl(job.id)}?t=${job.id}` : null),
    [job?.id, running]
  );

  return (
    <div className="console-page">
      <header className="console-header">
        <div>
          <span className="eyebrow">Live</span>
          <h1>Live Monitor</h1>
          <p>Analyze a webcam or RTSP camera feed in real time. Sessions are recorded for replay.</p>
        </div>
        <div className="console-status">
          <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
          <span>{backendOnline ? "Backend online" : backendOnline === false ? "Backend offline" : "Checking backend"}</span>
          <button className="icon-button" type="button" onClick={refreshHealth} aria-label="Refresh backend status">
            <RefreshCcw size={16} />
          </button>
        </div>
      </header>

      <section className="ops-layout live-layout">
        <aside className="source-panel">
          <div className="panel-heading">
            <h2>Camera Source</h2>
            <span className="pill">{running ? "Live" : "Idle"}</span>
          </div>

          <form onSubmit={handleStart}>
            <div className="live-mode-switch" role="radiogroup" aria-label="Source type">
              <button
                type="button"
                className={mode === "webcam" ? "active" : ""}
                onClick={() => setMode("webcam")}
                disabled={running}
              >
                <Camera size={15} />
                Webcam
              </button>
              <button
                type="button"
                className={mode === "rtsp" ? "active" : ""}
                onClick={() => setMode("rtsp")}
                disabled={running}
              >
                <Video size={15} />
                RTSP
              </button>
            </div>

            {mode === "webcam" ? (
              <label className="live-field">
                <span>Device index</span>
                <input
                  type="number"
                  min="0"
                  max="9"
                  value={webcamIndex}
                  onChange={(event) => setWebcamIndex(event.target.value)}
                  disabled={running}
                />
                <small>0 is the default camera on this machine.</small>
              </label>
            ) : (
              <label className="live-field">
                <span>RTSP URL</span>
                <input
                  type="text"
                  placeholder="rtsp://user:pass@camera-ip:554/stream"
                  value={rtspUrl}
                  onChange={(event) => setRtspUrl(event.target.value)}
                  disabled={running}
                />
                <small>Credentials are stripped from the saved job name.</small>
              </label>
            )}

            <div className="source-actions">
              {running ? (
                <button className="button danger full" type="button" onClick={handleStop} disabled={stopping}>
                  {stopping ? <Loader2 className="spin" size={17} /> : <Square size={16} />}
                  {stopping ? "Stopping" : "Stop Session"}
                </button>
              ) : (
                <button className="button full" type="submit" disabled={starting || backendOnline === false}>
                  {starting ? <Loader2 className="spin" size={17} /> : <Radio size={17} />}
                  {starting ? "Connecting" : "Go Live"}
                </button>
              )}
            </div>
          </form>

          <div className="source-message" aria-live="polite">{status || "Ready"}</div>

          {job && !running ? (
            <div className="summary-actions">
              {job.source_video ? (
                <Link className="button secondary full" href={`/jobs/${job.id}`}>
                  <PlayCircle size={16} />
                  Open Session Replay
                </Link>
              ) : null}
              <Link className="button secondary full" href="/violations">
                Review Violations
              </Link>
            </div>
          ) : null}

          <p className="source-hint">
            The stream shows detections as they happen. Evidence, plates, and the recorded video land in the same
            review flow as uploaded footage.
          </p>
        </aside>

        <main className="viewer-panel">
          <div className="viewer-toolbar">
            <div className="panel-title">
              <Radio size={16} className={running ? "live-pulse" : ""} />
              <h2>{running ? "Live Detection Feed" : "Feed Preview"}</h2>
            </div>
            {job ? (
              <span className={`pill ${job.status === "failed" ? "failed" : running ? "processing" : "completed"}`}>
                {running ? "streaming" : job.status}
              </span>
            ) : null}
          </div>

          <div className="preview-stage">
            {streamSrc ? (
              <img ref={streamRef} className="live-stream" src={streamSrc} alt="Live annotated camera stream" />
            ) : job?.preview_image ? (
              <img className="live-stream" src={mediaUrl(job.preview_image)} alt="Last annotated frame of the session" />
            ) : (
              <div className="empty-preview">
                <Radio size={42} />
                <span>No live session running</span>
              </div>
            )}
          </div>

          {job ? (
            <div className="frame-strip">
              <span>{running ? "LIVE" : "ENDED"}</span>
              <span className="live-strip-message">{job.message ?? ""}</span>
              <span>
                {formatDuration(job.elapsed_seconds)} | {job.violation_count} violation(s) | {job.sampled_frames} samples
              </span>
            </div>
          ) : null}
        </main>

        <aside className="telemetry-panel">
          <div className="panel-title">
            <Radio size={16} />
            <h2>Session</h2>
          </div>
          <div className="metric-tiles">
            <div className="metric-tile">
              <span>Elapsed</span>
              <strong>{job ? formatDuration(job.elapsed_seconds) : "-"}</strong>
            </div>
            <div className="metric-tile">
              <span>Frames</span>
              <strong>{job?.current_frame || "-"}</strong>
            </div>
            <div className="metric-tile">
              <span>Violations</span>
              <strong>{job?.violation_count ?? "-"}</strong>
            </div>
            <div className="metric-tile">
              <span>Samples</span>
              <strong>{job?.sampled_frames ?? "-"}</strong>
            </div>
          </div>

          <div className="current-summary">
            <div className="panel-title">
              <h2>Latest Evidence</h2>
            </div>
            {violations.length ? (
              <div className="live-evidence-list">
                {violations.slice(0, 4).map((violation) => (
                  <article className="live-evidence-item" key={violation.id}>
                    <img src={mediaUrl(violation.evidence_image)} alt="Live violation evidence frame" />
                    <span>
                      <strong>{plateLabel(violation)}</strong>
                      <small>Frame {violation.frame_number ?? "-"} | {Math.round(violation.helmet_confidence * 100)}%</small>
                    </span>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted">No violations captured this session.</p>
            )}
            {violations.length ? (
              <Link href="/violations" className="button secondary full">
                Open Review Queue
              </Link>
            ) : null}
          </div>
        </aside>
      </section>
    </div>
  );
}

function plateLabel(violation: Violation) {
  const text = violation.plate_text?.trim();
  if (text) {
    return text;
  }
  return violation.plate_image ? "Unreadable plate" : "Plate not captured";
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
