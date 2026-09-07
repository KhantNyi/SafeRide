import assert from "node:assert/strict";
import { test } from "node:test";
import { nearestDetection, smoothDetection } from "./overlay";
import type { DetectionFrame, DetectionBox } from "./api";

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
