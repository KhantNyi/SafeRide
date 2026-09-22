train-video-footage - SafeRide YOLO training annotations
Prepared 2026-09-16 from all 24 MOV clips in Downloads/Sp vd.

CONTENTS
images/         53 clean, contextual motorcycle crops: TRAINING INPUT.
labels/         53 matching YOLO bounding-box text files: TRAINING LABELS.
annotated/      Boxed previews for visual review; do not train on these.
source_frames/  53 uncropped original video frames for reference only.
review.html    Browse every clean crop and its boxed preview.
review/        Contact sheets and per-candidate annotation decisions.
manifest.json  Clip, zero-based frame number, timestamp, exact crop coordinates,
               image hash, split group, final pixel boxes and annotation origin.
summary.json   Counts and annotation limitations.
classes.txt    Canonical SafeRide class order.
notes.json     Category IDs for compatibility with existing raw exports.

LABEL FORMAT
Each row: class_id center_x center_y width height, normalized to 0..1
relative to the matching image in images/ (NOT the uncropped source frame).

0 Helmet
1 License Plate
2 Motorcycle
3 No Helmet
4 No License Plate
5 Person

ANNOTATION METHOD
Selected 64 candidate frames using previously reviewed motorcycle passages and
clear helmeted-rider examples. Retained 53 after visual review; excluded 11
unhelpful, repetitive, distant or incomplete candidate views. Sampling is
event-focused, not every decoded frame and not a uniform frame rate.

V4 helmet/plate models and YOLO11s supplied initial boxes. An AI visual reviewer
checked images, removed false/duplicate proposals, corrected classes and boxes,
and added missed visible heads, people and plates. Motorcycle boxes include
visible mirrors/handlebars and wheels. Helmet/No Helmet boxes surround heads;
Person boxes surround individual visible people, including occluded riders.
No unseen body parts or fully hidden heads were invented. These are AI-reviewed
annotations, not independently human-validated ground truth.

Training images use contextual crops to keep clearly annotatable riders while
excluding distant or uncertain background heads. They retain original pixels
and scale; no synthetic enhancement or generated image content is used.
Relevant incidental visible vehicle plates within retained crops are labeled.
Source videos are unchanged. Full source frames are preserved separately.
The coordinates of any final box in source pixels can be reconstructed by
adding manifest.crop_xyxy[0:2] to its xyxy coordinates.

No License Plate has zero examples: an obscured, unreadable, or undetected plate
does not prove that a motorcycle has no plate. No OCR text transcription is
provided; License Plate labels identify plate regions only.

TRAINING USE
This is a raw dataset export, compatible with SafeRide's six-class source
format. No training was started and no model or existing dataset was replaced.
For specialist training, remap Helmet (0) / No Helmet (3) to the helmet model's
class IDs, and License Plate (1) to the plate model's class ID, as the existing
SafeRide preparation scripts do. Do not pass these six-class label IDs directly
to a two-class helmet model or one-class plate model.

No train/validation split is imposed. Keep all frames from one source clip or
motorcycle passage together when splitting; check overlap with prior datasets
from the same filming session. All these clips were used in the earlier Sp vd
benchmark. Once this set is used for training, those videos are NOT an independent
test set. Use new held-out footage to measure generalization.
