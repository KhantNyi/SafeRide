# SafeRide Models

Downloaded model weights live in this folder, but `.pt` and `.onnx` files are ignored by git because they are large binaries.

## Current Local Model Paths

- `models/yolo11s.pt` - COCO-pretrained Ultralytics YOLO11s for `person`, `motorcycle`, and general objects.
- `models/helmet-yolov8n.pt` - YOLOv8n helmet/no-helmet detector from Hugging Face `iam-tsr/yolov8n-helmet-detection`.
- `models/license-plate-yolo11n.pt` - YOLO11n license plate detector from Hugging Face `morsetechlab/yolov11-license-plate-detection`.

## Notes

- The YOLO/Ultralytics-family models use AGPL-3.0 licensing. This is fine for a school project, but cite the sources in the report.
- These are baseline models. Thai traffic footage may still need fine-tuning for best accuracy.
