"""Fine-tune the two v3 detectors and compare them on fixed validation sets."""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training/round4"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "training/.ultralytics")
os.environ["WANDB_MODE"] = "disabled"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def main():
    from ultralytics import YOLO

    results = {}
    for task, filename in [("helmet", "helmet"), ("plate", "license-plate")]:
        source = ROOT / f"training/round3/models/{filename}-v3-best.pt"
        data = OUT / f"datasets/{task}/dataset.yaml"
        results[task] = {}
        for version, weights in [("v3", source), ("v4", OUT / f"runs/{task}-v4/weights/best.pt")]:
            if version == "v4":
                YOLO(str(source)).train(data=str(data), epochs=100, patience=30, imgsz=640,
                    batch=16, device=0, workers=0, project=str(OUT / "runs"),
                    name=f"{task}-v4", seed=0, deterministic=True, plots=True,
                    save=True, exist_ok=False)
            results[task][version] = {}
            for split_name, config in [("new_holdout", data), ("round3_holdout", ROOT / f"training/round3/datasets/{task}/dataset.yaml")]:
                metrics = YOLO(str(weights)).val(data=str(config), split="val", imgsz=640,
                    batch=16, device=0, workers=0, project=str(OUT / "evaluation"),
                    name=f"{task}-{version}-{split_name}", plots=True)
                results[task][version][split_name] = {k: float(v) for k,v in metrics.results_dict.items()}
            (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
        dest = OUT / "models" / f"{filename}-v4-best.pt"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(OUT / f"runs/{task}-v4/weights/best.pt", dest)
        print(f"EXPORTED {dest}", flush=True)
    (OUT / "COMPLETE.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
