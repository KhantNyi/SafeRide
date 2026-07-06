# SafeRide

SafeRide is a senior-project computer vision web app for detecting motorcycle helmet violations and capturing license plate evidence.

## Stack

- Frontend: Next.js, TypeScript
- Backend: FastAPI, SQLite
- CV pipeline: Ultralytics YOLO for motorcycle, helmet, and license plate detection
- First input mode: uploaded videos

## Local Setup

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Install ML packages for real detection:

```powershell
pip install -r backend\requirements-ml.txt
```

Run the backend in stable inference mode:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

During development, avoid watching generated evidence/model files:

```powershell
python -m uvicorn app.main:app --reload --reload-dir backend --app-dir backend --host 127.0.0.1 --port 8000
```

The current baseline model weights should be placed here:

```text
models/
  yolo11s.pt
  helmet-yolov8n.pt
  license-plate-yolo11n.pt
```

By default OCR is enabled so plate crops are read with EasyOCR during analysis. Disable it for a faster demo or to avoid OCR model initialization:

```powershell
$env:ENABLE_OCR="false"
```

Helmet detection runs on rider-focused crops around each motorcycle (batched into one model call), which gives distant riders much more effective resolution than a full-frame pass. Disable it to fall back to full-frame helmet inference at `HELMET_IMGSZ`:

```powershell
$env:HELMET_CROP_INFERENCE="false"
```

Object and plate detection still run full-frame at a pinned image size. Tune these if processing speed becomes more important than recall:

```powershell
$env:HELMET_CROP_IMGSZ="640"
$env:HELMET_IMGSZ="960"
$env:OBJECT_IMGSZ="960"
$env:PLATE_IMGSZ="960"
```

Videos process at full hardware speed. Real-time pacing only happens while the legacy MJPEG stream has a viewer, or when forced:

```powershell
$env:REALTIME_PREVIEW="true"
```

Violation quality gates (see `docs/confidence-settings.md` for the full list):

```powershell
$env:MIN_NO_HELMET_VOTES="2"        # no-helmet samples required per tracked rider
$env:PLATE_MIN_TRACK_SIGHTINGS="2"  # plate must co-travel with the rider's track
$env:ADAPTIVE_SAMPLING="true"       # densify sampling after a no-helmet detection
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

## MVP Flow

1. Upload a traffic or motorcycle video.
2. Backend creates a processing job.
3. YOLO samples detection frames while the backend writes detection metadata and annotated previews.
4. The web app plays the uploaded video directly with synchronized canvas overlays, plus Results and Evidence tabs.
5. If no no-helmet rider is found, the result shows "No violations detected."
6. If a no-helmet rider is found, evidence and license plate crops are saved.

## Evaluation

Measure pipeline accuracy against labeled clips (precision, recall, duplicate rate, plate capture, OCR accuracy):

```powershell
python scripts/evaluate.py scripts/eval-labels.example.json
```

Copy the example labels file and point it at your own clips with the frame ranges of real no-helmet riders. See `docs/system-design.md` (Evaluation section) for the format.

Human review decisions on the Violations page feed live precision metrics at:

```text
GET http://127.0.0.1:8000/api/metrics/review
```
