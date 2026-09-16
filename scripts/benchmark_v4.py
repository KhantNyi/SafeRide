"""Run an isolated full-frame v4 video benchmark at a fixed 0.5s interval."""
import argparse
import hashlib
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "training/.ultralytics")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def model_inventory():
    paths = list((ROOT / "models").glob("*.pt"))
    paths += list((ROOT / "training/runs").glob("*/weights/*.pt"))
    paths += list((ROOT / "training/round3").rglob("*.pt"))
    paths += list((ROOT / "training/round4/models").glob("*.pt"))
    return {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}


def write_report(out, data):
    clips = data["clips"]
    records = [v for c in clips for v in c["violations"]]
    totals = dict(clips=len(clips), statuses=dict(Counter(c["job"]["status"] for c in clips)),
        video_seconds=sum(c["seconds"] for c in clips), wall_seconds=sum(c["wall_seconds"] for c in clips),
        sampled_frames=sum(c["job"]["sampled_frames"] for c in clips),
        violation_records=len(records), clips_with_records=sum(bool(c["violations"]) for c in clips),
        records_with_plate_crop=sum(bool(v["plate_image"]) for v in records),
        records_with_ocr_text=sum(bool(v["plate_text"]) for v in records),
        plate_statuses=dict(Counter(v.get("plate_ocr_status") or "unknown" for v in records)))
    data["totals"] = totals
    (out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Sp vd: v4 full-frame video benchmark", "",
        "Settings: helmet-v4 + plate-v4; existing YOLO11s object detector; full-frame detection; fixed 0.5-second sampling (adaptive sampling off); OCR enabled. Image sizes and confidence thresholds are recorded in results.json.", "",
        "These are raw model outputs. For a completed visual review, see accuracy-report.md when present. Raw record counts alone do not establish accuracy or OCR correctness.", "",
        "## Summary", "", *[f"- {key}: {value}" for key,value in totals.items()], "",
        "## Per-video results", "", "| Video | Duration (s) | Analyzed frames | Records | Plate crops | OCR text | Processing (s) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    pages = ['<!doctype html><html><head><meta charset="utf-8"><title>SafeRide v4 — Sp vd</title><style>body{font:16px system-ui;background:#eef1f5;margin:30px;color:#182437}article{display:inline-block;vertical-align:top;background:white;padding:16px;margin:8px;width:340px;border-radius:10px}img{max-width:100%;max-height:500px}h2{margin-top:40px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #bbc}</style></head><body><h1>SafeRide v4 — Sp vd</h1><p>Full frame · fixed 0.5-second interval · OCR enabled. Outputs are not ground truth.</p>']
    for c in clips:
        vs = c["violations"]
        lines.append(f'| {c["clip"]} | {c["seconds"]:.1f} | {c["job"]["sampled_frames"]} | {len(vs)} | {sum(bool(v["plate_image"]) for v in vs)} | {sum(bool(v["plate_text"]) for v in vs)} | {c["wall_seconds"]:.1f} | {c["job"]["status"]} |')
        pages.append(f'<h2>{html.escape(c["clip"])}</h2><p>{len(vs)} records; {c["job"]["sampled_frames"]} frames analyzed; {c["wall_seconds"]:.1f}s processing</p>')
        if not vs:
            pages.append('<p>No violation records generated.</p>')
        for v in vs:
            evidence = v["evidence_image"].replace("/media/", "media/")
            plate = (v["plate_image"] or "").replace("/media/", "media/")
            pages.append(f'<article><p>Frame {v["frame_number"]} · track {v["track_id"]} · helmet confidence {v["helmet_confidence"]:.3f}</p><a href="{html.escape(evidence)}"><img loading="lazy" src="{html.escape(evidence)}"></a><p>OCR: {html.escape(v["plate_text"] or "none")} · {html.escape(v.get("plate_ocr_status") or "unknown")}</p>')
            if plate:
                pages.append(f'<a href="{html.escape(plate)}"><img loading="lazy" src="{html.escape(plate)}"></a>')
            pages.append('</article>')
    lines += ["", "## Preservation", "", f"Model checksums unchanged after benchmark: {data.get('model_weights_unchanged', 'check pending')}", "", "Previous training reports and application configuration were preserved. Evidence is in evidence.html; detailed settings, jobs and records are in results.json. Media and benchmark.db belong only to this test.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "evidence.html").write_text("\n".join(pages) + '</body></html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from app.core.config import settings
    settings.helmet_model_path = ROOT / "training/round4/models/helmet-v4-best.pt"
    settings.plate_model_path = ROOT / "training/round4/models/license-plate-v4-best.pt"
    settings.sample_every_seconds = 0.5
    settings.adaptive_sampling = False
    settings.helmet_crop_inference = False
    settings.enable_ocr = True
    settings.realtime_preview = False
    settings.model_device = "0"
    settings.data_dir = out / "media"
    for field, folder in [("upload_dir", "uploads"), ("evidence_dir", "evidence"), ("plate_dir", "plates"), ("preview_dir", "previews"), ("metadata_dir", "metadata")]:
        setattr(settings, field, settings.data_dir / folder)
        getattr(settings, field).mkdir(parents=True)
    settings.database_path = out / "benchmark.db"
    import cv2
    from app.core.database import init_db, get_connection
    from app.services.pipeline import process_uploaded_video, enable_capture_orientation_auto
    from app.services.repository import create_job, get_job
    init_db()
    before = model_inventory()
    data = dict(source=str(args.source.resolve()), settings=settings.model_dump(mode="json"), models_before=before, clips=[])
    videos = sorted(p for p in args.source.iterdir() if p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"})
    if not videos:
        raise RuntimeError("No videos found")
    for index, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        enable_capture_orientation_auto(cap)
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        item = dict(clip=video.name, seconds=frames/fps, fps=fps, frames=frames, width=int(cap.get(3)), height=int(cap.get(4)))
        cap.release()
        job_id = uuid4().hex
        create_job(job_id, video.name, str(video))
        start = perf_counter()
        process_uploaded_video(job_id, str(video))
        item["wall_seconds"] = perf_counter() - start
        item["job"] = get_job(job_id)
        with get_connection() as conn:
            item["violations"] = [dict(r) for r in conn.execute("SELECT * FROM violations WHERE job_id=? ORDER BY frame_number", (job_id,))]
        data["clips"].append(item)
        write_report(out, data)
        print(f'{index}/{len(videos)} {video.name}: {item["job"]["status"]}, {len(item["violations"])} records, {item["wall_seconds"]:.1f}s', flush=True)
    data["model_weights_unchanged"] = model_inventory() == before
    write_report(out, data)
    assert data["model_weights_unchanged"], "Unexpected model file change"
    assert all(c["job"]["status"] == "completed" for c in data["clips"]), "Some videos failed"
    print("COMPLETE", json.dumps(data["totals"]), flush=True)


if __name__ == "__main__":
    main()
