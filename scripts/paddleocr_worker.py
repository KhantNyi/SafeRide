"""JSON-lines worker for SafeRide's isolated PaddleOCR environment."""
import base64
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache/paddlex"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
protocol = sys.stdout
sys.stdout = sys.stderr


def create_model():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="th_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=320,
        text_det_limit_type="min",
        device="cpu",
        cpu_threads=4,
        enable_mkldnn=False,
    )


def main():
    import cv2
    import numpy as np

    model = create_model()
    for request in sys.stdin:
        try:
            data = json.loads(request)
            image = cv2.imdecode(np.frombuffer(base64.b64decode(data["image"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid plate crop")
            lines = []
            for result in model.predict(image):
                for polygon, text, score in zip(result["rec_polys"], result["rec_texts"], result["rec_scores"]):
                    lines.append([np.asarray(polygon).tolist(), str(text), float(score)])
            response = {"lines": lines}
        except Exception as exc:
            traceback.print_exc()
            response = {"error": f"PaddleOCR failed: {exc}"}
        protocol.write("SAFERIDE_OCR:" + json.dumps(response, ensure_ascii=True) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
