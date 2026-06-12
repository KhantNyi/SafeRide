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

By default OCR is disabled so the first real detector does not download OCR model files during a demo. Enable it with:

```powershell
$env:ENABLE_OCR="true"
```

Helmet inference is pinned to a larger image size by default because small road-scene riders can be missed at the model's implicit default size. Tune these if processing speed becomes more important than recall:

```powershell
$env:HELMET_IMGSZ="960"
$env:OBJECT_IMGSZ="960"
$env:PLATE_IMGSZ="960"
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
3. YOLO samples detection frames while the backend streams a paced live preview from the uploaded video.
4. The web app keeps one MJPEG live stream connected plus Results and Evidence tabs.
5. If no no-helmet rider is found, the result shows "No violations detected."
6. If a no-helmet rider is found, evidence and license plate crops are saved.
