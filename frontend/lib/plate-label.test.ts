import assert from "node:assert/strict";
import { test } from "node:test";
import { plateLabel } from "./plate-label";
import type { Violation } from "./api";

test("labels uncertainty while preserving legacy and manual text", () => {
  const plate = { plate_text: "1กข 1234", plate_image: "/plate.jpg" } as Violation;
  assert.equal(plateLabel({ ...plate, plate_ocr_status: "uncertain" }), "1กข 1234 (uncertain)");
  assert.equal(plateLabel({ ...plate, plate_ocr_status: "read" }), "1กข 1234");
  assert.equal(plateLabel(plate), "1กข 1234");
  assert.equal(plateLabel({ ...plate, plate_text: null }), "Unreadable plate");
  assert.equal(plateLabel({ ...plate, plate_text: null, plate_image: null }), "Plate not captured");
});
