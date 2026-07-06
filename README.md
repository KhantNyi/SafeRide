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

### GPU acceleration

Inference device is picked automatically (`MODEL_DEVICE=auto`): CUDA on NVIDIA machines, MPS on Apple Silicon, CPU otherwise. OCR follows CUDA availability unless `OCR_GPU` is set explicitly.

On Windows with an NVIDIA GPU, replace the default CPU PyTorch with the CUDA build (one-time, large download):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps
```

Force a specific device if needed:

```powershell
$env:MODEL_DEVICE="cpu"   # or cuda / mps
```

### macOS (Apple Silicon) setup

The same code runs on M-series Macs. Use bash equivalents for setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt -r backend/requirements-ml.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

YOLO inference auto-selects MPS (Apple GPU); OCR stays on CPU (EasyOCR's GPU path is CUDA-only). For webcam live sessions, grant camera permission to your terminal app when macOS prompts.

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

## Live Ingestion

The `/live` page analyzes a webcam or RTSP camera in real time. Pick Webcam (device index, usually `0`) or paste an RTSP URL, then Go Live. The annotated detection stream renders live, violations flow into the same review queue, and the whole session is recorded to `data/uploads` so it replays like an uploaded video. Sessions stop on demand, on source loss, or at the `LIVE_MAX_SECONDS` safety limit (default 900 s).

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
