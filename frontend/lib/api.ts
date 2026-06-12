export type Job = {
  id: string;
  filename: string;
  status: string;
  message: string | null;
  progress: number;
  current_frame: number;
  total_frames: number;
  sampled_frames: number;
  violation_count: number;
  elapsed_seconds: number;
  processing_fps: number;
  eta_seconds: number;
  preview_image: string | null;
  source_video: string | null;
  result: string | null;
  created_at: string;
  updated_at: string;
};

export type DetectionBox = {
  label: string;
  confidence: number;
  xyxy: [number, number, number, number];
};

export type DetectionAssociation = {
  track_id: number | null;
  track_hits: number;
  helmet_status: string;
  association_score: number;
  person_box: DetectionBox | null;
  motorcycle_box: DetectionBox | null;
  helmet_box: DetectionBox | null;
  plate_box: DetectionBox | null;
};

export type DetectionFrame = {
  frame_number: number;
  timestamp: number;
  width: number;
  height: number;
  people: DetectionBox[];
  motorcycles: DetectionBox[];
  helmets: DetectionBox[];
  no_helmets: DetectionBox[];
  plates: DetectionBox[];
  associations: DetectionAssociation[];
};

export type Violation = {
  id: string;
  job_id: string;
  detected_at: string;
  helmet_status: string;
  helmet_confidence: number;
  plate_text: string | null;
  plate_confidence: number | null;
  evidence_image: string;
  plate_image: string | null;
  frame_number: number | null;
  track_id: number | null;
  review_status: string;
};

export type DetectionSettings = {
  object_confidence: number;
  helmet_confidence: number;
  plate_confidence: number;
  sample_every_seconds: number;
  max_violations_per_video: number;
  enable_ocr: boolean;
};

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export function mediaUrl(path: string) {
  if (path.startsWith("http")) {
    return path;
  }
  return `${API_BASE}${path}`;
}

export async function fetchJobs(): Promise<Job[]> {
  const response = await fetch(`${API_BASE}/api/jobs`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load jobs");
  }
  return response.json();
}

export async function fetchHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export async function fetchSettings(): Promise<DetectionSettings> {
  const response = await fetch(`${API_BASE}/api/settings`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load settings");
  }
  return response.json();
}

export async function updateSettings(settings: DetectionSettings): Promise<DetectionSettings> {
  const response = await fetch(`${API_BASE}/api/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!response.ok) {
    throw new Error("Could not update settings");
  }
  return response.json();
}

export async function fetchJob(jobId: string): Promise<Job> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load job status");
  }
  return response.json();
}

export async function fetchDetections(jobId: string): Promise<DetectionFrame[]> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/detections`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load detection metadata");
  }
  const payload = (await response.json()) as { frames?: DetectionFrame[] };
  return payload.frames ?? [];
}

export async function fetchViolations(): Promise<Violation[]> {
  const response = await fetch(`${API_BASE}/api/violations`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load violations");
  }
  return response.json();
}

export async function deleteJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Could not delete job");
  }
}

export async function clearJobs(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/jobs`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Could not clear jobs");
  }
}

export async function deleteViolation(violationId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/violations/${violationId}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Could not delete violation");
  }
}

export async function reviewViolation(violationId: string, reviewStatus: "pending" | "confirmed" | "false_positive"): Promise<Violation> {
  const response = await fetch(`${API_BASE}/api/violations/${violationId}/review`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_status: reviewStatus })
  });
  if (!response.ok) {
    throw new Error("Could not update review decision");
  }
  return response.json();
}
