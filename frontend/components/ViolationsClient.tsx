"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ChevronDown, Clock3, Download, PlayCircle, RefreshCcw, Search, Trash2, X } from "lucide-react";

import { clearJobs, deleteViolation, fetchJobs, fetchViolations, Job, mediaUrl, reviewViolation, Violation } from "@/lib/api";

export function ViolationsClient() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState<Date | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "pending" | "confirmed" | "false_positive">("all");
  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);
  const [selectedPlate, setSelectedPlate] = useState<Violation | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const [collapsedJobIds, setCollapsedJobIds] = useState<string[]>([]);

  async function loadReviewData() {
    setError(null);
    try {
      const [jobData, violationData] = await Promise.all([fetchJobs(), fetchViolations()]);
      setJobs(jobData);
      setViolations(violationData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load violation records");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setNow(new Date());
    loadReviewData();
    const dataTimer = window.setInterval(loadReviewData, 5000);
    const clockTimer = window.setInterval(() => setNow(new Date()), 1000);
    return () => {
      window.clearInterval(dataTimer);
      window.clearInterval(clockTimer);
    };
  }, []);

  const jobsById = useMemo(() => new Map(jobs.map((job) => [job.id, job])), [jobs]);

  const rows = useMemo(
    () =>
      violations.filter((violation) => {
        const normalizedQuery = query.trim().toLowerCase();
        const job = jobsById.get(violation.job_id);
        const matchesQuery =
          !normalizedQuery ||
          plateLabel(violation).toLowerCase().includes(normalizedQuery) ||
          (job?.filename ?? "").toLowerCase().includes(normalizedQuery) ||
          (job?.message ?? "").toLowerCase().includes(normalizedQuery) ||
          violation.helmet_status.toLowerCase().includes(normalizedQuery) ||
          violation.job_id.toLowerCase().includes(normalizedQuery) ||
          String(violation.track_id ?? "").includes(normalizedQuery) ||
          formatRecordTime(violation.detected_at).toLowerCase().includes(normalizedQuery);
        const matchesStatus = statusFilter === "all" || reviewState(violation) === statusFilter;
        return matchesQuery && matchesStatus;
      }),
    [jobsById, violations, query, statusFilter]
  );

  const groupedRows = useMemo(
    () =>
      groupViolationsByJob(rows, jobsById).sort((a, b) => {
        const aTime = new Date(a.job?.created_at ?? a.latestDetectedAt).getTime();
        const bTime = new Date(b.job?.created_at ?? b.latestDetectedAt).getTime();
        return bTime - aTime;
      }),
    [jobsById, rows]
  );

  return (
    <div className="history-page violations-page">
      <header className="console-header">
        <div>
          <span className="eyebrow">Review Queue</span>
          <h1>Violations</h1>
          <p>Confirm detections, mark false positives, and export reviewed evidence.</p>
        </div>
        <div className="header-actions">
          <div className="console-clock">
            <time dateTime={now?.toISOString()}>
              <span>{now ? formatClockDate(now) : "Loading date"}</span>
              <strong>{now ? formatClockTime(now) : "--:--:--"}</strong>
            </time>
          </div>
          <button className="button secondary" type="button" onClick={loadReviewData}>
            <RefreshCcw size={16} />
            Refresh
          </button>
          <button className="button secondary" type="button" onClick={exportCsv} disabled={!rows.length}>
            <Download size={16} />
            Export CSV
          </button>
          <button className="button danger" type="button" onClick={clearAllRecords} disabled={clearing || !violations.length}>
            <Trash2 size={16} />
            {clearing ? "Clearing" : "Clear Records"}
          </button>
        </div>
      </header>

      <section className="content-card violation-summary-card" aria-label="Violation summary">
        <div>
          <h2>Violation Records</h2>
          <p>Recent helmet violations detected across completed analyses.</p>
        </div>
        <span className="pill warning">{violations.length} total</span>
      </section>

      {error ? <div className="notice danger" role="alert">{error}</div> : null}

      <section className="content-card violation-history">
        <div className="section-title">
          <div>
            <h2>Review Queue</h2>
            <p className="muted">Search by plate, date, status, filename, or job id.</p>
          </div>
          {loading ? <span className="pill processing">loading</span> : null}
        </div>

            <div className="violation-controls">
              <label className="violation-search">
                <Search size={16} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search plate, filename, date, status, job" />
              </label>
              <div className="violation-filter-chips" role="group" aria-label="Violation status filter">
                {(["all", "pending", "confirmed", "false_positive"] as const).map((filter) => (
                  <button className={statusFilter === filter ? "active" : ""} key={filter} type="button" onClick={() => setStatusFilter(filter)}>
                    {statusFilterLabel(filter)}
                  </button>
                ))}
              </div>
            </div>
            <div className="status-explainer">
              <span><i className="status-dot high" />Pending records still need a human decision.</span>
              <span><i className="status-dot review" />Use Confirm or False positive to finalize each detection.</span>
            </div>

            {groupedRows.length ? (
              <div className="violation-job-groups">
                {groupedRows.map((group) => {
                  const collapsed = collapsedJobIds.includes(group.jobId);
                  return (
                    <article className="violation-job-group" key={group.jobId}>
                      <header className="violation-job-header">
                        <button className="job-collapse-button" type="button" onClick={() => toggleJobGroup(group.jobId)} aria-expanded={!collapsed}>
                          <ChevronDown size={17} className={collapsed ? "collapsed" : ""} />
                          <span>
                            <strong>{group.job?.filename ?? `Job ${shortJobId(group.jobId)}`}</strong>
                            <small>{jobSubtitle(group)}</small>
                          </span>
                        </button>
                        <div className="violation-job-actions">
                          <span className="pill warning">{group.violations.length} violations</span>
                          {group.job ? <span className={`pill ${group.job.status}`}>{group.job.status}</span> : null}
                          <Link className="button secondary" href={`/jobs/${group.jobId}`}>
                            <PlayCircle size={16} />
                            Replay Job
                          </Link>
                          <button className="button secondary" type="button" onClick={() => exportJobCsv(group)}>
                            <Download size={16} />
                            Export Job
                          </button>
                          <button
                            className="button secondary"
                            type="button"
                            onClick={() => confirmPendingForGroup(group.violations)}
                            disabled={!group.violations.some((violation) => reviewState(violation) === "pending")}
                          >
                            Confirm Pending
                          </button>
                        </div>
                      </header>

                      {!collapsed ? (
                        <div className="violation-table" role="table" aria-label={`${group.job?.filename ?? group.jobId} violations`}>
                          <div className="violation-table-head" role="row">
                            <span>Snapshot</span>
                            <span>Plate OCR</span>
                            <span>Plate Crop</span>
                            <span>Frame</span>
                            <span>Status</span>
                            <span>Decision</span>
                            <span>Replay</span>
                            <span>Delete</span>
                          </div>

                          {group.violations.map((violation) => (
                            <ViolationRow
                              deletingId={deletingId}
                              key={violation.id}
                              onApplyReview={applyReview}
                              onInspectEvidence={setSelectedViolation}
                              onInspectPlate={setSelectedPlate}
                              onRemove={removeRecord}
                              violation={violation}
                            />
                          ))}
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            ) : (
              <div className="violation-empty">
                <Clock3 size={34} />
                <strong>No violation records</strong>
                <span>Run an analysis to populate this table.</span>
              </div>
            )}

            <footer className="violation-total">Showing {rows.length} of {violations.length} violation(s) across {groupedRows.length} job(s)</footer>
      </section>

      {selectedViolation ? (
        <div className="evidence-modal" role="dialog" aria-modal="true" aria-label="Violation evidence">
          <button className="evidence-modal-backdrop" type="button" onClick={() => setSelectedViolation(null)} aria-label="Close evidence inspector" />
          <section className="evidence-modal-panel">
            <header>
              <div>
                <h2>{plateLabel(selectedViolation)}</h2>
                <p>{formatRecordTime(selectedViolation.detected_at)}</p>
              </div>
              <button className="icon-button" type="button" onClick={() => setSelectedViolation(null)} aria-label="Close evidence inspector">
                <X size={18} />
              </button>
            </header>
            <img src={mediaUrl(selectedViolation.evidence_image)} alt="Large violation evidence" />
            <dl className="evidence-details">
              <div>
                <dt>Status</dt>
                <dd>{statusLabel(selectedViolation)}</dd>
              </div>
              <div>
                <dt>Confidence</dt>
                <dd>{Math.round(selectedViolation.helmet_confidence * 100)}%</dd>
              </div>
              <div>
                <dt>Frame</dt>
                <dd>{selectedViolation.frame_number ?? "-"}</dd>
              </div>
              <div>
                <dt>Track</dt>
                <dd>{selectedViolation.track_id ?? "-"}</dd>
              </div>
              <div>
                <dt>Plate Crop</dt>
                <dd>{selectedViolation.plate_image ? "Captured below" : "Not captured"}</dd>
              </div>
              <div>
                <dt>OCR Confidence</dt>
                <dd>{plateConfidenceLabel(selectedViolation)}</dd>
              </div>
            </dl>
            {selectedViolation.plate_image ? (
              <div className="plate-crop-panel">
                <span>Plate crop</span>
                <button className="plate-crop-preview" type="button" onClick={() => setSelectedPlate(selectedViolation)}>
                  <img src={mediaUrl(selectedViolation.plate_image)} alt="Detected license plate crop" />
                </button>
              </div>
            ) : null}
            <div className="evidence-modal-actions">
              <Link
                className="button secondary"
                href={`/jobs/${selectedViolation.job_id}${selectedViolation.frame_number === null ? "" : `?frame=${selectedViolation.frame_number}`}`}
              >
                <PlayCircle size={16} />
                Replay Moment
              </Link>
              <button className="button secondary" type="button" onClick={() => applyReview(selectedViolation, "confirmed")}>
                Confirm Violation
              </button>
              <button className="button secondary" type="button" onClick={() => applyReview(selectedViolation, "false_positive")}>
                Mark False Positive
              </button>
              <button className="button danger" type="button" onClick={() => removeRecord(selectedViolation)}>
                <Trash2 size={16} />
                Delete Record
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {selectedPlate?.plate_image ? (
        <div className="evidence-modal" role="dialog" aria-modal="true" aria-label="Plate crop preview">
          <button className="evidence-modal-backdrop" type="button" onClick={() => setSelectedPlate(null)} aria-label="Close plate crop preview" />
          <section className="evidence-modal-panel plate-modal-panel">
            <header>
              <div>
                <h2>{plateLabel(selectedPlate)}</h2>
                <p>{plateConfidenceLabel(selectedPlate) === "-" ? "OCR confidence unavailable" : `OCR confidence ${plateConfidenceLabel(selectedPlate)}`}</p>
              </div>
              <button className="icon-button" type="button" onClick={() => setSelectedPlate(null)} aria-label="Close plate crop preview">
                <X size={18} />
              </button>
            </header>
            <img className="plate-modal-image" src={mediaUrl(selectedPlate.plate_image)} alt="Large detected license plate crop" />
          </section>
        </div>
      ) : null}
    </div>
  );

  function exportCsv() {
    downloadViolationCsv("saferide-violations.csv", rows, jobsById);
  }

  function exportJobCsv(group: ViolationJobGroup) {
    downloadViolationCsv(`saferide-${group.job?.filename ?? group.jobId}-violations.csv`, group.violations, jobsById);
  }

  function toggleJobGroup(jobId: string) {
    setCollapsedJobIds((current) => (current.includes(jobId) ? current.filter((id) => id !== jobId) : [...current, jobId]));
  }

  async function removeRecord(violation: Violation) {
    if (!window.confirm("Delete this violation record and evidence image?")) {
      return;
    }

    setDeletingId(violation.id);
    setError(null);
    try {
      await deleteViolation(violation.id);
      if (selectedViolation?.id === violation.id) {
        setSelectedViolation(null);
      }
      await loadReviewData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete violation");
    } finally {
      setDeletingId(null);
    }
  }

  async function applyReview(violation: Violation, reviewStatus: "confirmed" | "false_positive") {
    setError(null);
    try {
      const updated = await reviewViolation(violation.id, reviewStatus);
      setViolations((current) => current.map((record) => (record.id === updated.id ? updated : record)));
      setSelectedViolation((current) => (current?.id === updated.id ? updated : current));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update review decision");
    }
  }

  async function confirmPendingForGroup(groupViolations: Violation[]) {
    const pending = groupViolations.filter((violation) => reviewState(violation) === "pending");
    if (!pending.length || !window.confirm(`Confirm ${pending.length} pending violation(s) for this job?`)) {
      return;
    }

    setError(null);
    try {
      await Promise.all(pending.map((violation) => reviewViolation(violation.id, "confirmed")));
      await loadReviewData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not confirm pending violations");
    }
  }

  async function clearAllRecords() {
    if (!window.confirm("Delete all previous jobs, violation records, and generated media?")) {
      return;
    }

    setClearing(true);
    setError(null);
    try {
      await clearJobs();
      setSelectedViolation(null);
      setSelectedPlate(null);
      await loadReviewData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not clear records");
    } finally {
      setClearing(false);
    }
  }
}

type ViolationJobGroup = {
  jobId: string;
  job: Job | null;
  latestDetectedAt: string;
  violations: Violation[];
};

function ViolationRow({
  deletingId,
  onApplyReview,
  onInspectEvidence,
  onInspectPlate,
  onRemove,
  violation
}: {
  deletingId: string | null;
  onApplyReview: (violation: Violation, reviewStatus: "confirmed" | "false_positive") => void;
  onInspectEvidence: (violation: Violation) => void;
  onInspectPlate: (violation: Violation) => void;
  onRemove: (violation: Violation) => void;
  violation: Violation;
}) {
  return (
    <article className="violation-table-row" role="row">
      <button className="violation-snapshot" type="button" onClick={() => onInspectEvidence(violation)} aria-label="Inspect evidence snapshot">
        <img src={mediaUrl(violation.evidence_image)} alt="Violation evidence snapshot" />
      </button>
      <span className={`plate-chip ${violation.plate_text ? "" : "pending"}`}>{plateLabel(violation)}</span>
      {violation.plate_image ? (
        <button className="plate-preview interactive" type="button" onClick={() => onInspectPlate(violation)} aria-label="Inspect plate crop">
          <img src={mediaUrl(violation.plate_image)} alt="Detected license plate crop" />
        </button>
      ) : (
        <span className="plate-preview">Not captured</span>
      )}
      <span className="frame-cell">Frame {violation.frame_number ?? "-"}</span>
      <span className={`violation-status ${statusTone(violation)}`}>{statusLabel(violation)}</span>
      <span className="review-actions">
        <button type="button" onClick={() => onApplyReview(violation, "confirmed")} disabled={violation.review_status === "confirmed"}>
          Confirm
        </button>
        <button type="button" onClick={() => onApplyReview(violation, "false_positive")} disabled={violation.review_status === "false_positive"}>
          False positive
        </button>
      </span>
      <Link
        className="icon-button"
        href={`/jobs/${violation.job_id}${violation.frame_number === null ? "" : `?frame=${violation.frame_number}`}`}
        aria-label="Replay violation moment"
      >
        <PlayCircle size={16} />
      </Link>
      <button
        className="icon-button danger"
        type="button"
        onClick={() => onRemove(violation)}
        disabled={deletingId === violation.id}
        aria-label="Delete violation record"
      >
        <Trash2 size={16} />
      </button>
    </article>
  );
}

function groupViolationsByJob(rows: Violation[], jobsById: Map<string, Job>): ViolationJobGroup[] {
  const groups = new Map<string, Violation[]>();
  for (const violation of rows) {
    const records = groups.get(violation.job_id) ?? [];
    records.push(violation);
    groups.set(violation.job_id, records);
  }

  return [...groups.entries()].map(([jobId, groupViolations]) => {
    const sorted = [...groupViolations].sort((a, b) => {
      const frameA = a.frame_number ?? Number.MAX_SAFE_INTEGER;
      const frameB = b.frame_number ?? Number.MAX_SAFE_INTEGER;
      if (frameA !== frameB) {
        return frameA - frameB;
      }
      return new Date(a.detected_at).getTime() - new Date(b.detected_at).getTime();
    });
    return {
      jobId,
      job: jobsById.get(jobId) ?? null,
      latestDetectedAt: sorted.reduce((latest, violation) => (violation.detected_at > latest ? violation.detected_at : latest), sorted[0]?.detected_at ?? ""),
      violations: sorted
    };
  });
}

function downloadViolationCsv(filename: string, records: Violation[], jobsById: Map<string, Job>) {
  const header = ["Job", "Job ID", "Plate OCR", "Detected At", "Status", "Confidence", "Frame", "Track", "Evidence Image"];
  const lines = records.map((violation) => {
    const job = jobsById.get(violation.job_id);
    return [
      job?.filename ?? "",
      violation.job_id,
      plateLabel(violation),
      formatRecordTime(violation.detected_at),
      statusLabel(violation),
      Math.round(violation.helmet_confidence * 100),
      violation.frame_number ?? "",
      violation.track_id ?? "",
      mediaUrl(violation.evidence_image)
    ]
      .map(csvCell)
      .join(",");
  });
  const csv = [header.map(csvCell).join(","), ...lines].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = sanitizeFilename(filename);
  link.click();
  URL.revokeObjectURL(url);
}

function jobSubtitle(group: ViolationJobGroup) {
  const date = group.job?.created_at ?? group.latestDetectedAt;
  const formatted = date ? formatRecordTime(date) : "Unknown date";
  return `${shortJobId(group.jobId)} | ${formatted}`;
}

function shortJobId(jobId: string) {
  return jobId.slice(0, 8);
}

function sanitizeFilename(filename: string) {
  return filename.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-+|-+$/g, "") || "saferide-violations.csv";
}

function plateLabel(violation: Violation) {
  const text = violation.plate_text?.trim();
  if (text) {
    return text;
  }
  return violation.plate_image ? "Unreadable plate" : "Plate not captured";
}

function plateConfidenceLabel(violation: Violation) {
  return violation.plate_confidence === null ? "-" : `${Math.round(violation.plate_confidence * 100)}%`;
}

function statusLabel(violation: Violation) {
  if (violation.review_status === "confirmed") {
    return "Confirmed violation";
  }
  if (violation.review_status === "false_positive") {
    return "False positive";
  }
  return violation.helmet_confidence >= 0.65 ? "Pending - high confidence" : "Pending - needs review";
}

function statusTone(violation: Violation) {
  if (violation.review_status === "confirmed") {
    return "confirmed";
  }
  if (violation.review_status === "false_positive") {
    return "false-positive";
  }
  return violation.helmet_confidence >= 0.65 ? "high" : "review";
}

function reviewState(violation: Violation) {
  if (violation.review_status === "confirmed" || violation.review_status === "false_positive") {
    return violation.review_status;
  }
  return "pending";
}

function statusFilterLabel(value: "all" | "pending" | "confirmed" | "false_positive") {
  return {
    all: "All",
    pending: "Pending",
    confirmed: "Confirmed",
    false_positive: "False positive"
  }[value];
}

function csvCell(value: string | number) {
  return `"${String(value).replaceAll("\"", "\"\"")}"`;
}

function formatClockDate(date: Date) {
  return new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric"
  }).format(date);
}

function formatClockTime(date: Date) {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true
  }).format(date);
}

function formatRecordTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  const second = String(date.getSeconds()).padStart(2, "0");
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}
