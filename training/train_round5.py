"""Fine-tune v4 specialists, evaluate v4/v5 identically, preserve older weights."""
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training/round5"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "training/.ultralytics")
os.environ["WANDB_MODE"] = "disabled"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["YOLO_OFFLINE"] = "true"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from ultralytics import YOLO
    import ultralytics.utils.checks
    ultralytics.utils.checks.check_pip_update_available = lambda: None
    assert not (OUT/'runs').exists(), 'Refusing to overwrite a training run'
    inventory=json.loads((OUT/'prior-model-inventory.json').read_text())
    results={}
    for task,filename in [('helmet','helmet'),('plate','license-plate')]:
        source=ROOT/f'training/round4/models/{filename}-v4-best.pt'
        data=OUT/f'datasets/{task}/dataset.yaml'
        configs=[('new_holdout',data),('round4_holdout',ROOT/f'training/round4/datasets/{task}/dataset.yaml'),('round3_holdout',ROOT/f'training/round3/datasets/{task}/dataset.yaml')]
        results[task]={}
        for version in ('v4','v5'):
            weights=source
            if version=='v5':
                print(f'TRAINING {task} FROM {source}',flush=True)
                YOLO(str(source)).train(data=str(data),epochs=100,patience=30,imgsz=640,
                    batch=16,device=0,workers=0,project=str(OUT/'runs'),name=f'{task}-v5',
                    seed=0,deterministic=True,plots=True,save=True,exist_ok=False,
                    optimizer='AdamW',lr0=.001,lrf=.01,amp=True)
                weights=OUT/f'models/{filename}-v5-best.pt'
                weights.parent.mkdir(exist_ok=True)
                shutil.copy2(OUT/f'runs/{task}-v5/weights/best.pt',weights)
                print(f'EXPORTED {weights}',flush=True)
            results[task][version]={}
            for name,config in configs:
                metrics=YOLO(str(weights)).val(data=str(config),split='val',imgsz=640,
                    batch=16,device=0,workers=0,project=str(OUT/'evaluation'),
                    name=f'{task}-{version}-{name}',plots=True)
                results[task][version][name]={k:float(v) for k,v in metrics.results_dict.items()}
                (OUT/'metrics.json').write_text(json.dumps(results,indent=2))
        model=YOLO(str(weights))
        sample=next(p for p in sorted((OUT/f'datasets/{task}/images/val').iterdir())
                    if (OUT/f'datasets/{task}/labels/val'/p.with_suffix('.txt').name).read_text().strip())
        prediction=model.predict(str(sample),device=0,imgsz=640,verbose=False)
        checks={'weights':str(weights),'sha256':sha(weights),'matches_best':sha(weights)==sha(OUT/f'runs/{task}-v5/weights/best.pt'),
                'classes':model.names,'inference_sample':str(sample),'inference_boxes':len(prediction[0].boxes)}
        (OUT/f'{task}-artifact-check.json').write_text(json.dumps(checks,indent=2))
    assert all((ROOT/p).exists() and sha(ROOT/p)==value for p,value in inventory.items()), 'Previous weight changed'
    (OUT/'COMPLETE.json').write_text(json.dumps({'metrics':results,'prior_weights_unchanged':len(inventory)},indent=2))
    print('COMPLETE: v5 exports verified; all prior weights unchanged',flush=True)


if __name__=='__main__': main()
