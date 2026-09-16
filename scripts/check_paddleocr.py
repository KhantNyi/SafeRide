"""Download/check the Thai recognizer, or recognize a supplied text-line crop."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache" / "paddlex"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="A cropped single line of text")
    args = parser.parse_args()
    import importlib.metadata
    from PIL import Image, ImageDraw, ImageFont
    from paddleocr import TextRecognition

    output_dir = ROOT / "analysis" / "paddleocr-setup"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.image
    if not input_path:
        input_path = str(output_dir / "synthetic-digits.png")
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 64)
        left, top, right, bottom = font.getbbox("3583")
        image = Image.new("RGB", (right - left + 12, bottom - top + 12), "white")
        ImageDraw.Draw(image).text((6 - left, 6 - top), "3583", font=font, fill="black")
        image.save(input_path)
    model = TextRecognition(model_name="th_PP-OCRv5_mobile_rec", device="cpu", enable_mkldnn=False)
    results = []
    for result in model.predict(input=input_path, batch_size=1):
        result.save_to_json(str(output_dir))
        results.append({"text": result["rec_text"], "score": float(result["rec_score"])})
    report = {
        "versions": {name: importlib.metadata.version(name) for name in ("paddleocr", "paddlepaddle", "paddlex")},
        "model": "th_PP-OCRv5_mobile_rec",
        "device": "cpu",
        "input": input_path,
        "synthetic": args.image is None,
        "results": results,
        "note": "Synthetic smoke test only; not a real-plate accuracy benchmark.",
    }
    (output_dir / "check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.image is None and (not results or results[0]["text"] != "3583"):
        raise SystemExit("Synthetic digit recognition check failed")


if __name__ == "__main__":
    main()
