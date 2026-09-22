"""Isolated benchmark of unlabeled September clips; no accuracy scoring."""
import json
import argparse
import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.core.config import settings

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--v3', action='store_true', help='Use round3 specialist weights and isolated v3 outputs')
parser.add_argument('--output', help='Separate output subdirectory, preserving previous benchmarks')
parser.add_argument('--intervals', type=float, nargs='+', help='Run these intervals with OCR disabled')
parser.add_argument('--clips', nargs='+', help='Restrict to named clips for regression investigation')
args = parser.parse_args()
if args.v3:
    OUT = OUT / 'v3'
    OUT.mkdir(exist_ok=True)
    settings.helmet_model_path = ROOT / 'training/round3/models/helmet-v3-best.pt'
    settings.plate_model_path = ROOT / 'training/round3/models/license-plate-v3-best.pt'
if args.output:
    OUT = Path(__file__).resolve().parent / args.output
    OUT.mkdir(parents=True, exist_ok=True)
SOURCE = Path('C:/Users/ADMIN/Downloads/Sp data 2 september')
settings.data_dir = OUT / 'media'
for field, folder in [('upload_dir', 'uploads'), ('evidence_dir', 'evidence'), ('plate_dir', 'plates'), ('preview_dir', 'previews'), ('metadata_dir', 'metadata')]:
    setattr(settings, field, settings.data_dir / folder)
    getattr(settings, field).mkdir(parents=True, exist_ok=True)
settings.database_path = OUT / 'benchmark.db'
settings.realtime_preview = False

import cv2
from app.core.database import init_db, get_connection
from app.services.pipeline import process_uploaded_video
from app.services.repository import create_job, get_job

def main():
    init_db()
    inventory = []
    for video in sorted(SOURCE.glob('*.MOV')):
        if args.clips and video.name not in args.clips:
            continue
        cap = cv2.VideoCapture(str(video))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        inventory.append(dict(clip=video.name, seconds=frames/fps, fps=fps, frames=frames, width=int(cap.get(3)), height=int(cap.get(4))))
        cap.release()
    results = {'inventory': inventory, 'settings': settings.model_dump(mode='json'), 'runs': []}
    runs = [(i, False) for i in args.intervals] if args.intervals else [(1.0, False), (0.5, False), (0.25, False), (0.5, True)]
    for interval, ocr in runs:
        settings.sample_every_seconds = interval
        settings.enable_ocr = ocr
        run = {'interval': interval, 'ocr': ocr, 'clips': []}
        results['runs'].append(run)
        started = perf_counter()
        for item in inventory:
            video = SOURCE / item['clip']
            job_id = uuid4().hex
            create_job(job_id, video.name, str(video))
            begin = perf_counter()
            process_uploaded_video(job_id, str(video))
            wall = perf_counter() - begin
            with get_connection() as conn:
                violations = [dict(r) for r in conn.execute('SELECT * FROM violations WHERE job_id=? ORDER BY frame_number', (job_id,))]
            job = get_job(job_id)
            run['clips'].append({'clip':video.name,'wall_seconds':wall,'job':job,'violations':violations})
            print(f"interval={interval} OCR={ocr} {video.name}: {job['status']}, {len(violations)} records, {wall:.1f}s", flush=True)
            (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        run['wall_seconds'] = perf_counter() - started
        (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"RUN COMPLETE: {interval}, OCR={ocr}, {run['wall_seconds']:.1f}s", flush=True)

if __name__ == '__main__':
    main()
