"""Isolated full-pipeline regression check; leaves app settings and data alone."""
import json
import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    from app.core.config import settings
    out = ROOT / "analysis/overlap-fix/validated-pipeline"
    out.mkdir(parents=True, exist_ok=False)
    settings.data_dir = out / "media"
    for field, folder in [("upload_dir", "uploads"), ("evidence_dir", "evidence"),
                          ("plate_dir", "plates"), ("preview_dir", "previews"),
                          ("metadata_dir", "metadata")]:
        setattr(settings, field, settings.data_dir / folder)
        getattr(settings, field).mkdir(parents=True)
    settings.database_path = out / "test.db"
    settings.helmet_crop_inference = False
    settings.ocr_engine = "paddleocr"
    settings.enable_ocr = True
    settings.adaptive_sampling = True
    settings.realtime_preview = False
    from app.core.database import init_db, get_connection
    from app.services import pipeline
    from app.services.repository import create_job, get_job
    init_db()
    results = []
    for version, name, interval in [(4, "IMG_6665", .5), (4, "IMG_6665", 1.),
                                    (4, "IMG_6805", .5), (5, "IMG_6665", .5)]:
        settings.helmet_model_path = ROOT / f"training/round{version}/models/helmet-v{version}-best.pt"
        settings.plate_model_path = ROOT / f"training/round{version}/models/license-plate-v{version}-best.pt"
        pipeline._helmet_model = None
        pipeline._plate_model = None
        settings.sample_every_seconds = interval
        source = Path(r"C:\Users\ADMIN\Downloads\Sp vd") / (name + ".mov")
        job_id = uuid4().hex
        create_job(job_id, source.name, str(source))
        start = perf_counter()
        pipeline.process_uploaded_video(job_id, str(source))
        with get_connection() as conn:
            records = [dict(row) for row in conn.execute(
                "SELECT * FROM violations WHERE job_id=? ORDER BY frame_number", (job_id,))]
        result = dict(version=version, clip=source.name, interval=interval,
                      settings=settings.model_dump(mode="json"), job=get_job(job_id),
                      violations=records, wall_seconds=perf_counter()-start)
        results.append(result)
        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(version, name, interval, result["job"]["status"], len(records), flush=True)
    assert all(r["job"]["status"] == "completed" for r in results)


if __name__ == "__main__":
    main()
