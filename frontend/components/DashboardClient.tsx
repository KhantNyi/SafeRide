"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { AlertTriangle, FileSearch, PlayCircle, RefreshCcw, Search, Trash2, Upload } from "lucide-react";

import { clearJobs, deleteJob, fetchJobs, fetchViolations, Job, mediaUrl, Violation } from "@/lib/api";
import { StatCard } from "@/components/StatCard";

export function DashboardClient() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "violations" | "clear" | "failed">("all");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  async function loadData() {
    setError(null);
    try {
      const [jobData, violationData] = await Promise.all([fetchJobs(), fetchViolations()]);
      setJobs(jobData);
      setViolations(violationData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dashboard request failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
    const timer = window.setInterval(loadData, 5000);
    return () => window.clearInterval(timer);
  }, []);

  async function handleDeleteJob(job: Job) {
    if (!window.confirm(`Delete "${job.filename}" and its saved evidence?`)) {
      return;
    }

    setDeletingId(job.id);
    setError(null);
    try {
      await deleteJob(job.id);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete job");
    } finally {
      setDeletingId(null);
    }
  }

  async function handleClearJobs() {
    if (!jobs.length || !window.confirm("Delete all previous jobs, violation records, and generated media?")) {
      return;
    }

    setClearing(true);
    setError(null);
    try {
      await clearJobs();
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not clear jobs");
    } finally {
      setClearing(false);
    }
  }

  const completedJobs = useMemo(() => jobs.filter((job) => job.status === "completed").length, [jobs]);
  const activeJobs = useMemo(
    () => jobs.filter((job) => job.status === "queued" || job.status === "processing").length,
    [jobs]
  );
  const clearJobCount = useMemo(
    () => jobs.filter((job) => job.result === "no_violations" || (job.status === "completed" && job.violation_count === 0)).length,
    [jobs]
  );
  const filteredJobs = useMemo(
    () =>
      jobs.filter((job) => {
        const normalizedQuery = query.trim().toLowerCase();
        const matchesQuery =
          !normalizedQuery ||
          job.filename.toLowerCase().includes(normalizedQuery) ||
          (job.message ?? "").toLowerCase().includes(normalizedQuery);
        const matchesStatus =
          statusFilter === "all" ||
          (statusFilter === "active" && (job.status === "queued" || job.status === "processing")) ||
          (statusFilter === "violations" && job.violation_count > 0) ||
          (statusFilter === "clear" && (job.result === "no_violations" || (job.status === "completed" && job.violation_count === 0))) ||
          (statusFilter === "failed" && job.status === "failed");
        return matchesQuery && matchesStatus;
      }),
    [jobs, query, statusFilter]
  );
  const visibleViolations = useMemo(
    () =>
      violations.filter((violation) => {
        const normalizedQuery = query.trim().toLowerCase();
        return (
          !normalizedQuery ||
          plateLabel(violation).toLowerCase().includes(normalizedQuery) ||
          violation.job_id.toLowerCase().includes(normalizedQuery)
        );
      }),
    [violations, query]
  );

  return (
    <div className="history-page">
      <header className="console-header">
        <div>
          <span className="eyebrow">Overview</span>
          <h1>Operations Dashboard</h1>
          <p>Monitor analysis jobs, active processing, and recent evidence.</p>
        </div>
        <div className="header-actions">
          <button className="button secondary" type="button" onClick={loadData}>
            <RefreshCcw size={16} />
            Refresh
          </button>
          <button className="button danger" type="button" onClick={handleClearJobs} disabled={clearing || !jobs.length}>
            <Trash2 size={16} />
            {clearing ? "Clearing" : "Clear History"}
          </button>
          <Link className="button" href="/upload">
            <Upload size={18} />
            New Analysis
          </Link>
        </div>
      </header>

      <section className="stats-grid" aria-label="System summary">
        <StatCard label="Violations" value={violations.length} />
        <StatCard label="Completed Jobs" value={completedJobs} />
        <StatCard label="Active Jobs" value={activeJobs} />
        <StatCard label="Clear Results" value={clearJobCount} />
      </section>

      {error ? <div className="notice danger" role="alert">{error}</div> : null}

      <section className="review-controls" aria-label="Review filters">
        <label className="search-field">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search jobs, messages, plates" />
        </label>
        <div className="filter-chips" role="group" aria-label="Job status filter">
          {(["all", "active", "violations", "clear", "failed"] as const).map((filter) => (
            <button
              className={statusFilter === filter ? "active" : ""}
              key={filter}
              type="button"
              onClick={() => setStatusFilter(filter)}
            >
              {filterLabel(filter)}
            </button>
          ))}
        </div>
      </section>

      <div className="review-layout">
        <section className="content-card">
          <div className="section-title">
            <h2>Jobs</h2>
            {loading ? <span className="pill processing">loading</span> : null}
          </div>

          {!loading && filteredJobs.length === 0 ? <EmptyState text="No jobs match the current filters." /> : null}

          <div className="job-table">
            {filteredJobs.map((job) => (
              <article className="job-table-row" key={job.id}>
                <div className="job-main">
                  <strong>{job.filename}</strong>
                  <span>{job.message ?? "No status message."}</span>
                </div>
                <span className={`pill ${job.status}`}>{job.status}</span>
                <span className={`result-badge ${toneForJob(job)}`}>{labelForJob(job)}</span>
                <span className="job-number">{job.violation_count}</span>
                <span className="job-date">{new Date(job.created_at).toLocaleString()}</span>
                <Link
                  className={`icon-button ${job.source_video ? "" : "disabled"}`}
                  href={job.source_video ? `/upload?job=${job.id}` : "#"}
                  aria-disabled={!job.source_video}
                  aria-label={`Open playback for ${job.filename}`}
                >
                  <PlayCircle size={16} />
                </Link>
                <button
                  className="icon-button danger"
                  type="button"
                  onClick={() => handleDeleteJob(job)}
                  disabled={deletingId === job.id}
                  aria-label={`Delete ${job.filename}`}
                >
                  <Trash2 size={16} />
                </button>
              </article>
            ))}
          </div>
        </section>

        <section className="content-card">
          <div className="section-title">
            <h2>Recent Evidence</h2>
            <span className="pill">{visibleViolations.length} records</span>
          </div>

          {!loading && visibleViolations.length === 0 ? <EmptyState text="No saved violation evidence matches the current search." /> : null}

          <div className="evidence-feed">
            {visibleViolations.map((violation) => (
              <article className="evidence-feed-card" key={violation.id}>
                <img src={mediaUrl(violation.evidence_image)} alt="Saved traffic evidence frame" />
                <div>
                  <strong>{plateLabel(violation)}</strong>
                  <span>{new Date(violation.detected_at).toLocaleString()}</span>
                  <div className="violation-meta">
                    <span className="pill failed">{violation.helmet_status.replaceAll("_", " ")}</span>
                    <span className="pill warning">Frame {violation.frame_number ?? "-"}</span>
                    <span className="pill">{Math.round(violation.helmet_confidence * 100)}%</span>
                  </div>
                  <Link className="inline-review-link" href="/violations">
                    <AlertTriangle size={15} />
                    Open violation table
                  </Link>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="empty-state">
      <FileSearch size={34} />
      <span>{text}</span>
    </div>
  );
}

function labelForJob(job: Job) {
  if (job.status === "queued" || job.status === "processing") {
    return "Analyzing";
  }
  if (job.status === "failed") {
    return "Failed";
  }
  return job.violation_count > 0 ? "Violation" : "Clear";
}

function toneForJob(job: Job) {
  if (job.status === "queued" || job.status === "processing") {
    return "neutral";
  }
  if (job.status === "failed" || job.violation_count > 0) {
    return "danger";
  }
  return "clear";
}

function plateLabel(violation: Violation) {
  const text = violation.plate_text?.trim();
  if (text) {
    return text;
  }
  return violation.plate_image ? "Unreadable plate" : "Plate not captured";
}

function filterLabel(filter: "all" | "active" | "violations" | "clear" | "failed") {
  return {
    all: "All",
    active: "Active",
    violations: "Violations",
    clear: "Clear",
    failed: "Failed"
  }[filter];
}
