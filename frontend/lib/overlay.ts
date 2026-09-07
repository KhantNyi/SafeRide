import type { DetectionAssociation, DetectionBox, DetectionFrame } from "./api";

const MAX_GAP = 1.25;
const fields = ["person_box", "motorcycle_box", "helmet_box", "plate_box"] as const;
const boxKey = (box: DetectionBox) => `${box.label}:${box.xyxy.join(",")}`;

export function nearestDetection(frames: DetectionFrame[], time: number): DetectionFrame | null {
  let closest: DetectionFrame | null = null;
  let distance = Infinity;
  for (const frame of frames) {
    const gap = Math.abs(frame.timestamp - time);
    if (gap < distance) { closest = frame; distance = gap; }
  }
  return distance <= MAX_GAP ? closest : null;
}

function interpolate(a: DetectionBox, b: DetectionBox, fraction: number, base: DetectionBox): DetectionBox | null {
  if (a.label !== b.label) return null;
  const [x, y, right, bottom] = a.xyxy;
  const [nx, ny, nr, nb] = b.xyxy;
  const width = right - x, height = bottom - y;
  const nextWidth = nr - nx, nextHeight = nb - ny;
  if (Math.min(width, height, nextWidth, nextHeight) <= 0) return null;
  const ratio = (nextWidth * nextHeight) / (width * height);
  const movement = Math.hypot((nx + nr - x - right) / 2, (ny + nb - y - bottom) / 2);
  if (ratio < 0.25 || ratio > 4 || movement > Math.max(width, height, nextWidth, nextHeight) * 1.5) return null;
  return { ...base, xyxy: a.xyxy.map((value, i) => value + (b.xyxy[i] - value) * fraction) as DetectionBox["xyxy"] };
}

// Multiple occupants can share a motorcycle ID. Without occupant IDs, never
// invent a correspondence between their heads or bodies.
function uniqueAssociation(frame: DetectionFrame, id: number): DetectionAssociation | null {
  const matches = frame.associations.filter((a) => a.track_id === id);
  return matches.length === 1 ? matches[0] : null;
}

export function smoothDetection(frames: DetectionFrame[], time: number, base = nearestDetection(frames, time)): DetectionFrame | null {
  if (!base) return null;
  let before: DetectionFrame | undefined, after: DetectionFrame | undefined;
  for (const frame of frames) {
    if (frame.timestamp <= time && (!before || frame.timestamp > before.timestamp)) before = frame;
    if (frame.timestamp >= time && (!after || frame.timestamp < after.timestamp)) after = frame;
  }
  if (!before || !after || before === after || after.timestamp - before.timestamp > MAX_GAP ||
      before.width !== after.width || before.height !== after.height) return base;
  const fraction = (time - before.timestamp) / (after.timestamp - before.timestamp);
  const replacements = new Map<string, DetectionBox>();
  const associations = base.associations.map((association) => {
    if (association.track_id === null) return association;
    const a = uniqueAssociation(before!, association.track_id);
    const b = uniqueAssociation(after!, association.track_id);
    if (!a || !b || a.helmet_status !== b.helmet_status || !a.motorcycle_box || !b.motorcycle_box || !association.motorcycle_box) return association;
    if (!interpolate(a.motorcycle_box, b.motorcycle_box, fraction, association.motorcycle_box)) return association;
    const result = { ...association };
    for (const field of fields) {
      const start = a[field], end = b[field], original = association[field];
      if (!start || !end || !original) continue;
      const box = interpolate(start, end, fraction, original);
      if (box) { result[field] = box; replacements.set(boxKey(original), box); }
    }
    return result;
  });
  const replace = (boxes: DetectionBox[]) => boxes.map((box) => replacements.get(boxKey(box)) ?? box);
  return { ...base, associations, people: replace(base.people), motorcycles: replace(base.motorcycles),
    helmets: replace(base.helmets), no_helmets: replace(base.no_helmets), plates: replace(base.plates) };
}
