import type { Violation } from "./api";

export function plateLabel(violation: Violation): string {
  const text = violation.plate_text?.trim();
  if (text) return violation.plate_ocr_status === "uncertain" ? `${text} (uncertain)` : text;
  return violation.plate_image ? "Unreadable plate" : "Plate not captured";
}
