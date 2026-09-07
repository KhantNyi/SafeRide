# Plate OCR selection and uncertainty

Plate crops are buffered before violation confirmation and during the existing
collection window. The defaults still retain five crops and run OCR on at most
three. Each selected crop uses the existing original, resized, and thresholded
variants; no sharpening or extra inference attempts were added.

Selection always retains the highest quality candidate first. Subsequent choices
receive a diversity bonus of at most 0.10: 60% from time separation (saturating at
0.5 seconds) and 40% from a small grayscale appearance descriptor. Diversity is
relative to the closest already selected candidate. This helps avoid spending
the entire OCR budget on adjacent, nearly identical views without promoting
clearly inferior crops. Source FPS converts observation frames to seconds for
both uploads and live sessions. The descriptor is for view selection, not proof
of plate ownership. Selection adds small CPU work; end-to-end performance has
not been benchmarked.

Joined and split registration readings are normalized before character voting:
`1กข1234` and `1กข 1234` both become `1กข 1234`. Leading zeros are preserved.

New automatic records include `plate_ocr_status`:

- `read`: a complete prefix and number with voted confidence at least 0.5,
  supported in full (including province if present) by at least two crop readings
  with confidence at least 0.5. Supporting readings must carry at least two-thirds
  of all reading weight.
- `uncertain`: text exists but lacks that support, including partial readings,
  single observations, weak readings, and contradictory or synthesized strings.
  The text is retained and shown with an `(uncertain)` suffix for review.
- `unreadable`: no OCR text was produced.

These are conservative, uncalibrated heuristics, not measured probabilities of
correct registration or ownership. Repeated OCR errors can still agree. The
underlying confidence score is preserved; it is not inflated to reflect status.
Helmet review decisions remain separate from OCR reliability.

The additive database migration leaves existing and manual records with null
status; their historical text is not rewritten or reassessed. Reprocess a video
to apply the new crop selection and OCR logic. The original saved crop remains
available even when its text is uncertain.

Backend regression coverage:
`python -m unittest discover -s backend/tests -v`
