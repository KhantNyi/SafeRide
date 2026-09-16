# PaddleOCR Thai evaluation setup

PaddleOCR is installed separately from the running SafeRide backend at
`.cache/paddleocr-venv`. SafeRide can now select either EasyOCR or PaddleOCR;
the local `.env` selects PaddleOCR for new processing.

## Test through SafeRide

1. Open http://localhost:3000/upload and refresh the page.
2. Under Settings, enable OCR and select **PaddleOCR (Thai PP-OCRv5)** in
   **Plate OCR engine**. Click **Apply Settings**.
3. Upload a video and start the analysis. Read the resulting plates in Results
   or Evidence. Existing jobs are not reprocessed.
4. To compare, select **EasyOCR (Thai)**, apply settings and upload the same
   video again, keeping the other settings equal.

Full frame and 0.5-second sample interval are currently configured. PaddleOCR
runs on CPU. The first read starts a persistent local worker, adding startup
time; later crops reuse that worker. Results remain subject to SafeRide's
existing multi-frame agreement checks, so partial numbers can remain uncertain.

The app worker uses `PP-OCRv5_mobile_det` to find and crop text lines, then
`th_PP-OCRv5_mobile_rec` to recognize each line. The app handles whole plate
crops automatically; manual row cropping is only needed for the standalone
recognizer command below. The worker communicates through local process pipes.
Runtime errors fail the job visibly rather than falling back silently to
EasyOCR. Diagnostics are in `.cache/paddleocr-worker.log`.

### Integration validation

- 39 backend tests passed; frontend TypeScript check passed.
- Upload page returned HTTP 200 and includes the engine selector.
- Fresh IMG_6805.MOV job `e07787bc97544acc98a177be304338ca` completed with
  PaddleOCR: 75 sampled frames, 3 violations, 47.8 seconds including cold start.
- All three OCR readings remained uncertain. No accuracy improvement is claimed.
- Settings, job and readings are recorded in
  `analysis/paddleocr-setup/app-test-job.json`.

## Components

- PaddleOCR 3.7.0
- PaddlePaddle 3.3.0, CPU runtime
- Thai recognition model: `th_PP-OCRv5_mobile_rec`
- Text-line detection model: `PP-OCRv5_mobile_det`
- Model cache: `.cache/paddlex`

The standalone recognition model recognizes individual text lines.
This installation does not change YOLO v4 or earlier weights.

## Reproduce

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m venv .cache\paddleocr-venv
.\.cache\paddleocr-venv\Scripts\python.exe -m pip install -r backend\requirements-paddleocr.txt
.\.cache\paddleocr-venv\Scripts\python.exe scripts\check_paddleocr.py
```

The first check downloads the Thai model. Subsequent runs use the local cache.
The default check creates a clear synthetic `3583` image and performs inference.
This verifies installation, not accuracy on the user's real plate images.
Results are saved under `analysis/paddleocr-setup`.

Verified on 2026-09-16: dependency check passed, the model loaded from its
project-local cache, and a tightly cropped synthetic `3583` line was recognized
as `3583` (score 0.99998). A generously padded version was misread as `3` with
and without oneDNN acceleration. This demonstrates sensitivity to crop margins;
real plate accuracy is still unmeasured. The check script disables oneDNN.

To read a real single-row crop:

```powershell
.\.cache\paddleocr-venv\Scripts\python.exe scripts\check_paddleocr.py "path\to\number-row.png"
```

Compare exact digit and full-registration accuracy against EasyOCR on the same
manually verified plate crops, and measure processing time. This integration
does not establish which engine is more accurate. GPU acceleration has not
been configured in this environment.

References: [official installation](https://www.paddlepaddle.org.cn/documentation/docs/install/pip/windows-pip_en.html),
[Thai model](https://huggingface.co/PaddlePaddle/th_PP-OCRv5_mobile_rec).
