import assert from "node:assert/strict";
import { test } from "node:test";
import { nearestDetection, smoothDetection } from "./overlay";
import type { DetectionFrame, DetectionBox } from "./api";

test("visual playback uses only its own identities and expires stale estimates", () => {
  const a = frame(0, 100), b = frame(.1, 120, 9);
  for (const f of [a, b]) { f.playback_generated = true; f.motorcycles[0].playback_id = "visual:1"; }
  assert.equal(smoothDetection([a, b], .05)!.motorcycles[0].xyxy[0], 110);
  assert.equal(nearestDetection([a, b], .3), null);
  b.motorcycles[0].playback_id = "visual:2";
  assert.equal(smoothDetection([a, b], .05)!.motorcycles[0].xyxy[0], 100);
});

function frame(time: number, x: number, id = 1): DetectionFrame {
  const bike: DetectionBox = { label: "motorcycle", confidence: .8, xyxy: [x, 100, x + 100, 300] };
  return { timestamp: time, frame_number: time * 30, width: 1080, height: 1920,
    people: [], motorcycles: [bike], helmets: [], no_helmets: [], plates: [],
    associations: [{ track_id: id, track_hits: 3, helmet_status: "no_helmet", association_score: .8,
      motorcycle_box: bike, person_box: null, helmet_box: null, plate_box: null }] };
}

test("interpolates matching tracks and keeps observations immutable", () => {
  const a = frame(0, 100), b = frame(1, 200);
  const snapshot = JSON.stringify([a, b]);
  const output = smoothDetection([a, b], .5)!;
  assert.equal(output.motorcycles[0].xyxy[0], 150);
  assert.equal(output.associations[0].motorcycle_box!.xyxy[0], 150);
  assert.equal(output.frame_number, a.frame_number);
  assert.equal(JSON.stringify([a, b]), snapshot);
});

test("preserves exact observed boxes and does not extrapolate", () => {
  const a = frame(0, 100), b = frame(1, 200);
  assert.equal(smoothDetection([a, b], 0), a);
  assert.equal(smoothDetection([a, b], 1), b);
  assert.equal(smoothDetection([a, b], 1.1), b);
  assert.equal(nearestDetection([a, b], 3), null);
});

test("does not bridge changed IDs, large gaps, or implausible motion", () => {
  const a = frame(0, 100);
  for (const b of [frame(1, 200, 2), frame(2, 200), frame(1, 1000)]) {
    assert.deepEqual(smoothDetection([a, b], .4)!.motorcycles, a.motorcycles);
  }
});

test("does not invent occupant matches or smooth untracked objects", () => {
  const a = frame(0, 100), b = frame(1, 200);
  b.associations.push({ ...b.associations[0] });
  assert.deepEqual(smoothDetection([a, b], .4)!.motorcycles, a.motorcycles);
  b.associations = [];
  assert.deepEqual(smoothDetection([a, b], .4)!.motorcycles, a.motorcycles);
});

test("smooths helmeted riders at one-second sampling without adding violation associations", () => {
  const frames = [frame(0, 100), frame(1, 200)];
  for (const f of frames) {
    const x = f.motorcycles[0].xyxy[0];
    const person: DetectionBox = { label: "person", confidence: .9, xyxy: [x, 0, x + 60, 180] };
    const head: DetectionBox = { label: "with helmet", confidence: .9, xyxy: [x, 0, x + 40, 40] };
    f.people = [person]; f.helmets = [head];
    f.motorcycles[0].track_id = 1;
    f.tracking_associations = [{ ...f.associations[0], helmet_status: "with_helmet", person_box: person, helmet_box: head }];
    f.associations = [];
  }
  const snapshot = JSON.stringify(frames);
  const result = smoothDetection(frames, .5)!;
  assert.equal(result.people[0].xyxy[0], 150);
  assert.equal(result.helmets[0].xyxy[0], 150);
  assert.equal(result.motorcycles[0].xyxy[0], 150);
  assert.deepEqual(result.associations, []);
  assert.equal(JSON.stringify(frames), snapshot);
});

test("smooths bikes without helmet observations and skips ambiguous bike identities", () => {
  const a = frame(0, 100), b = frame(1, 200);
  for (const f of [a, b]) { f.motorcycles[0].track_id = 1; f.associations = []; }
  assert.equal(smoothDetection([a, b], .5)!.motorcycles[0].xyxy[0], 150);
  b.motorcycles.push({ ...b.motorcycles[0] });
  assert.equal(smoothDetection([a, b], .5)!.motorcycles[0].xyxy[0], 100);
});

test("shared occupants prevent body smoothing but not motorcycle smoothing", () => {
  const a = frame(0, 100), b = frame(1, 200);
  for (const f of [a, b]) {
    f.motorcycles[0].track_id = 1;
    const person: DetectionBox = { label: "person", confidence: .9, xyxy: [f.motorcycles[0].xyxy[0], 0, 300, 180] };
    f.people = [person];
    f.tracking_associations = [{ ...f.associations[0], person_box: person }, { ...f.associations[0], helmet_status: "with_helmet" }];
  }
  const result = smoothDetection([a, b], .5)!;
  assert.equal(result.motorcycles[0].xyxy[0], 150);
  assert.deepEqual(result.people, a.people);
  assert.equal(result.associations[0].motorcycle_box!.xyxy[0], 150);
});

test("helmet classification changes do not freeze bodies or interpolate helmet labels", () => {
  const a = frame(0, 100), b = frame(1, 200);
  for (const f of [a, b]) {
    const x = f.motorcycles[0].xyxy[0];
    f.people = [{ label: "person", confidence: .8, xyxy: [x, 0, x + 60, 180] }];
    const head: DetectionBox = { label: f === a ? "without helmet" : "with helmet", confidence: .8, xyxy: [x, 0, x + 40, 40] };
    f.tracking_associations = [{ ...f.associations[0], person_box: f.people[0], helmet_box: head,
      helmet_status: f === a ? "no_helmet" : "with_helmet" }];
    if (f === a) f.no_helmets = [head]; else f.helmets = [head];
  }
  const result = smoothDetection([a, b], .5)!;
  assert.equal(result.people[0].xyxy[0], 150);
  assert.deepEqual(result.no_helmets, a.no_helmets);
});
