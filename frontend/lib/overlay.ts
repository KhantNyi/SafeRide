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
  return distance <= (closest?.playback_generated ? .15 : MAX_GAP) ? closest : null;
}

function interpolate(a: DetectionBox, b: DetectionBox, fraction: number, base: DetectionBox, anchorMotion: [number, number] = [0, 0]): DetectionBox | null {
  if (a.label !== b.label) return null;
  const [x, y, right, bottom] = a.xyxy;
  const [nx, ny, nr, nb] = b.xyxy;
  const width = right - x, height = bottom - y;
  const nextWidth = nr - nx, nextHeight = nb - ny;
  if (Math.min(width, height, nextWidth, nextHeight) <= 0) return null;
  const ratio = (nextWidth * nextHeight) / (width * height);
  const movement = Math.hypot((nx + nr - x - right) / 2 - anchorMotion[0], (ny + nb - y - bottom) / 2 - anchorMotion[1]);
  if (ratio < 0.25 || ratio > 4 || movement > Math.max(width, height, nextWidth, nextHeight) * 1.5) return null;
  return { ...base, xyxy: a.xyxy.map((value, i) => value + (b.xyxy[i] - value) * fraction) as DetectionBox["xyxy"] };
}

// Multiple occupants can share a motorcycle ID. Without occupant IDs, never
// invent a correspondence between their heads or bodies.
function uniqueAssociation(frame: DetectionFrame, id: number): DetectionAssociation | null {
  const matches = (frame.tracking_associations ?? frame.associations).filter((a) => a.track_id === id);
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
  if (base.playback_generated) {
    // Visual-only identities never participate in violation tracking.
    for (const group of ["people", "motorcycles", "helmets", "no_helmets", "plates"] as const) {
      for (const box of base[group]) {
        if (!box.playback_id) continue;
        const a = before[group].find((candidate) => candidate.playback_id === box.playback_id);
        const b = after[group].find((candidate) => candidate.playback_id === box.playback_id);
        if (a && b && after.timestamp - before.timestamp <= .2) {
          replacements.set(boxKey(box), { ...box, xyxy: a.xyxy.map((v, i) => v + (b.xyxy[i] - v) * fraction) as DetectionBox["xyxy"] });
        }
      }
    }
    const result = { ...base };
    for (const group of ["people", "motorcycles", "helmets", "no_helmets", "plates"] as const) {
      result[group] = base[group].map((box) => replacements.get(boxKey(box)) ?? box);
    }
    result.associations = base.associations.map((association) => {
      const updated = { ...association };
      for (const field of fields) {
        const box = association[field];
        if (box) updated[field] = replacements.get(boxKey(box)) ?? box;
      }
      return updated;
    });
    return result;
  }
  // Bikes retain their own identity even when occupants or helmet labels change.
  for (const bike of base.motorcycles) {
    if (bike.track_id == null) continue;
    const starts = before.motorcycles.filter((box) => box.track_id === bike.track_id);
    const ends = after.motorcycles.filter((box) => box.track_id === bike.track_id);
    if (starts.length !== 1 || ends.length !== 1) continue;
    const box = interpolate(starts[0], ends[0], fraction, bike);
    if (box) replacements.set(boxKey(bike), box);
  }
  (base.tracking_associations ?? base.associations).forEach((association) => {
    if (association.track_id === null) return association;
    const a = uniqueAssociation(before!, association.track_id);
    const b = uniqueAssociation(after!, association.track_id);
    if (!a || !b || !a.motorcycle_box || !b.motorcycle_box || !association.motorcycle_box) return association;
    if (!interpolate(a.motorcycle_box, b.motorcycle_box, fraction, association.motorcycle_box)) return association;
    const startBike = a.motorcycle_box.xyxy, endBike = b.motorcycle_box.xyxy;
    const anchorMotion: [number, number] = [
      (endBike[0] + endBike[2] - startBike[0] - startBike[2]) / 2,
      (endBike[1] + endBike[3] - startBike[1] - startBike[3]) / 2,
    ];
    for (const field of fields) {
      if (field === "helmet_box" && a.helmet_status !== b.helmet_status) continue;
      const start = a[field], end = b[field], original = association[field];
      if (!start || !end || !original) continue;
      const box = interpolate(start, end, fraction, original, field === "motorcycle_box" ? [0, 0] : anchorMotion);
      if (box) replacements.set(boxKey(original), box);
    }
  });
  const replace = (boxes: DetectionBox[]) => boxes.map((box) => replacements.get(boxKey(box)) ?? box);
  const replaceAssociation = (association: DetectionAssociation): DetectionAssociation => {
    const result = { ...association };
    for (const field of fields) {
      const box = association[field];
      if (box) result[field] = replacements.get(boxKey(box)) ?? box;
    }
    return result;
  };
  const associations = base.associations.map(replaceAssociation);
  return { ...base, associations, people: replace(base.people), motorcycles: replace(base.motorcycles),
    ...(base.tracking_associations ? { tracking_associations: base.tracking_associations.map(replaceAssociation) } : {}),
    helmets: replace(base.helmets), no_helmets: replace(base.no_helmets), plates: replace(base.plates) };
}
