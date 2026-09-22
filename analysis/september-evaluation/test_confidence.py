"""Brief isolated threshold comparison; does not change app configuration."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.core.config import settings

if __name__ == '__main__':
    settings.helmet_crop_inference = False
    settings.plate_confidence = 0.30
    settings.adaptive_sampling = True
    for name, helmet, objects in [
        ('baseline', .35, .35),
        ('helmet25', .25, .35),
        ('helmet15', .15, .35),
        ('helmet25-object25', .25, .25),
    ]:
        settings.helmet_confidence = helmet
        settings.object_confidence = objects
        sys.argv = ['run_benchmark.py', '--v3', '--output', f'confidence-check/{name}',
                    '--intervals', '0.5', '--clips', 'IMG_7087.MOV', 'IMG_7089.MOV', 'IMG_7077.MOV']
        runpy.run_path(str(Path(__file__).with_name('run_benchmark.py')), run_name='__main__')
