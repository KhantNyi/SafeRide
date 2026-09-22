"""Continue v5 on street-level data; preserve all earlier weights and app settings."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'training/round6'
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'training/.ultralytics')
os.environ['WANDB_MODE'] = 'disabled'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['YOLO_OFFLINE'] = 'true'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import torch
    from ultralytics import YOLO
    import ultralytics.utils.checks
    ultralytics.utils.checks.check_pip_update_available = lambda: None
    assert torch.cuda.is_available(), 'CUDA required for this run'
    assert not (OUT/'runs').exists(), 'Refusing to overwrite training'
    previous = json.loads((OUT/'prior-model-inventory.json').read_text())
    source_inventory = json.loads((OUT/'source-inventory.json').read_text())
    results, artifacts = {}, {}
    started = perf_counter()
    for task, filename in [('helmet','helmet'),('plate','license-plate')]:
        source = ROOT/f'training/round5/models/{filename}-v5-best.pt'
        data = OUT/f'datasets/{task}/dataset.yaml'
        results[task] = {}
        for version in ('v5','v6'):
            weights = source
            if version == 'v6':
                print(f'TRAINING {task} FROM {source}',flush=True)
                YOLO(str(source)).train(data=str(data), epochs=100, patience=25,
                    imgsz=960,batch=8,device=0,workers=0,cache='ram',
                    project=str(OUT/'runs'),name=f'{task}-v6',exist_ok=False,
                    seed=0,deterministic=True,plots=True,save=True,
                    optimizer='AdamW',lr0=.0005,lrf=.01,amp=True,
                    mosaic=.5,close_mosaic=10,scale=.35)
                weights = OUT/f'models/{filename}-v6-best.pt'
                weights.parent.mkdir(exist_ok=True)
                for checkpoint in ('best','last'):
                    shutil.copy2(OUT/f'runs/{task}-v6/weights/{checkpoint}.pt',
                                 OUT/f'models/{filename}-v6-{checkpoint}.pt')
                print(f'EXPORTED {weights}',flush=True)
            configs = [('monitor_seen_by_base',data)] + [
                (f'former_round{n}_val_NOW_TRAINING',ROOT/f'training/round{n}/datasets/{task}/dataset.yaml') for n in (3,4,5)]
            results[task][version] = {}
            for name, config in configs:
                metrics = YOLO(str(weights)).val(data=str(config),split='val',
                    imgsz=960,batch=8,device=0,workers=0,project=str(OUT/'evaluation'),
                    name=f'{task}-{version}-{name}',plots=True)
                results[task][version][name] = {k:float(v) for k,v in metrics.results_dict.items()}
                (OUT/'metrics.json').write_text(json.dumps(results,indent=2))
            if version == 'v6':
                model = YOLO(str(weights))
                sample = next((OUT/f'datasets/{task}/images/train').glob('*IMG_6712_f000300*'))
                prediction = model.predict(str(sample),device=0,imgsz=960,verbose=False)
                artifacts[task] = dict(weights=str(weights),sha256=sha(weights),
                    initialized_from=str(source),source_sha256=sha(source),
                    matches_best=sha(weights)==sha(OUT/f'runs/{task}-v6/weights/best.pt'),
                    classes=model.names,inference_sample=str(sample),inference_boxes=len(prediction[0].boxes))
                assert artifacts[task]['matches_best']
                (OUT/'artifact-checks.json').write_text(json.dumps(artifacts,indent=2))
    assert all(sha(ROOT/p)==digest for p,digest in previous.items()), 'Earlier checkpoint changed'
    assert all(sha(ROOT/p)==digest for p,digest in source_inventory.items()), 'Source dataset changed'
    assert sha(ROOT/'.env') == (OUT/'app-env-sha256.txt').read_text(), 'App configuration changed'
    complete = dict(metrics=results,artifacts=artifacts,wall_seconds=perf_counter()-started,
                    prior_weights_unchanged=len(previous),source_files_unchanged=len(source_inventory),
                    app_configuration_unchanged=True,
                    evaluation_limit='No independent unseen test; monitoring seen by base, earlier validation reused for training.')
    (OUT/'COMPLETE.json').write_text(json.dumps(complete,indent=2))
    print('COMPLETE: v6 exports verified and old artifacts preserved',flush=True)


if __name__ == '__main__':
    main()
