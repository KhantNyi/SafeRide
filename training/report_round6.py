"""Summarize completed round 6 with explicit evaluation limitations."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'training/round6'


def main():
    audit=json.loads((OUT/'data-audit.json').read_text(encoding='utf-8'))
    complete=json.loads((OUT/'COMPLETE.json').read_text())
    videos=json.loads((OUT/'video-checks/results.json').read_text(encoding='utf-8'))
    lines=['# SafeRide round 6: street-level continuation from v5','',
        '## Outcome','',
        'Both specialist detectors were fine-tuned directly from v5 and exported as v6. '
        'Earlier checkpoints and source datasets were preserved. The training script left application '
        'configuration unchanged at completion. On resuming the final checks, the current configuration '
        'already selected both v6 specialists; it was preserved.','',
        '**This is the requested reuse experiment, not an independent generalization test.** '
        'Former street-level validation examples now contribute to training. The monitoring clips '
        'are excluded from round-6 gradient updates, but were seen by earlier models. '
        'Historical validation metrics are now overlapping/training diagnostics, not holdout scores.','',
        '**Verdict:** keep v6 as an experimental checkpoint. The familiar-video checks still produce '
        'the pink-helmet false violation, and IMG_6665 gains a duplicate record with the wrong motorcycle\'s plate. '
        'Higher diagnostic mAP does not establish a safer end-to-end replacement. Both training jobs '
        'had already finished in the background before the user offered to skip plate fine-tuning; '
        'no additional plate training was started on resumption.','',
        '## Dataset','',
        '- Inputs: street-level `train-data3`, `train-data4`, and `train-data5`, visually checked using contact sheets.',
        '- Excluded: all 325 overhead-view images in `train-data` and `train-data2`. '
        'The inherited v5 weights still contain earlier learning; this filter applies to round-6 input data.',
        f"- Merged output: `train-data6`, {audit['unique_images']} unique image-label pairs.",
        f"- Removed duplicate copies: {len(audit['duplicate_decisions'])}. Entries excluded for unresolved conflicting labels: {len(audit['exclusions'])}.",
        '- Newer reviewed crops take precedence over their older full-frame copies. '
        'IMG_7439 and IMG_7278 remain excluded from helmet training because their headwear labels are ambiguous; '
        'their plate labels may still train the plate detector.',
        '- Both helmets in `IMG_6712_f000300` are class Helmet and now belong to training.',
        '- Added usable former validation examples to existing street-level training data; '
        'did not train exclusively on old validation.',
        '- Monitoring clips (previously training): '+', '.join(audit['monitoring_groups'])+'.',
        '- Entire known clips remain together; exact pixel hashes and capture IDs do not cross the round-6 split. '
        'Monitoring footage shares the street/capture domain and is not a fresh recording session.',
        '- Person/motorcycle detection and OCR were not trained. Sources include previously model-assisted, '
        'AI-reviewed annotations; no independent human label audit is claimed.','',
        '| Detector | Training images | Former-validation images in training | Monitoring images | Training boxes |',
        '|---|---:|---:|---:|---|']
    for task,counts in audit['counts'].items():
        train=counts['train']; val=counts['val']
        boxes=', '.join(f'{k}: {v}' for k,v in train.items() if k not in ('images','former_validation_images'))
        lines.append(f"| {task} | {train['images']} | {train['former_validation_images']} | {val['images']} | {boxes} |")
    lines += ['', '## Training','',
        '- Initialization: `training/round5/models/helmet-v5-best.pt` and `license-plate-v5-best.pt`.',
        '- 100-epoch cap, patience 25, image size 960, batch 8, CUDA device 0, workers 0, '
        'seed 0, deterministic mode, AMP, AdamW, initial learning rate 0.0005, final multiplier 0.01, '
        'mosaic probability 0.5, scale augmentation 0.35, close mosaic for last 10 scheduled epochs.',
        '- Resized training images cached in RAM to avoid repeatedly decoding large PNGs. '
        'An initial uncached attempt was stopped early and preserved under `analysis/round6-audit/uncached-initial-*`; '
        'the final run restarted from the same v5 checkpoints. RAM caching can limit strict determinism.',
        '- Training size changed from 640 in round 5 to 960 to match application inference. '
        'Both versions are evaluated at 960 here; these numbers are not directly comparable to the older 640-pixel report.',
        f"- Training/evaluation wall time: {complete['wall_seconds']/60:.1f} minutes."]
    for task in ('helmet','plate'):
        with (OUT/f'runs/{task}-v6/results.csv').open() as f:
            rows=list(csv.DictReader(f))
        lines.append(f'- {task}: {len(rows)} epochs completed; best and last checkpoints retained.')
    lines += ['', '## Matched diagnostic metrics','',
        '**No row below is an independent unseen-test accuracy estimate.**', '',
        '| Detector | Dataset | Version | Precision | Recall | mAP50 | mAP50-95 |',
        '|---|---|---|---:|---:|---:|---:|']
    for task,versions in complete['metrics'].items():
        for version,sets in versions.items():
            for name,metrics in sets.items():
                values=[metrics[k] for k in ('metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)')]
                lines.append(f'| {task} | {name} | {version} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '## Familiar-video diagnostic','',
        'Full-frame inference, 0.5-second base interval, adaptive sampling on, OCR disabled. '
        'Existing tracking/association code is identical for both versions. These videos now overlap training; '
        'record counts alone do not establish correctness. Evidence and detection metadata are preserved.', '',
        '| Video | Version | Records | Analyzed frames | Status |',
        '|---|---|---:|---:|---|']
    for r in videos:
        lines.append(f"| {r['clip']} | v{r['version']} | {len(r['violations'])} | {r['job']['sampled_frames']} | {r['job']['status']} |")
    lines += ['', 'Visual review: the v6 IMG_6712 record at frame 330 flags the pink-helmet passenger. '
        'In IMG_6665, v6 frames 522 and 618 flag the same first motorcycle twice; frame 522 saves '
        'its plate 3399, while frame 618 saves plate 3872 from the helmeted motorcycle behind it. '
        'V5 produces two records in that comparison. IMG_6805 keeps three records for both versions. '
        'These observations do not constitute exhaustive ground-truth scoring.']
    pink=json.loads((OUT/'pink-helmet-check/results.json').read_text())
    lines += ['', '### Pink helmet: fixed-frame check', '',
        'The additional near-view training example did **not** resolve the distant pink-helmet error. '
        'Both versions recognize the helmet at frame 300 but predict no helmet at frames 318, 342 and 366. '
        'This test predicts on full frames at 960 pixels and confidence 0.35. '
        'It isolates detector behavior; OCR and tracking are not involved. '
        'Approximate head regions come from the inspected earlier-run metadata.', '',
        '| Frame | Version | Pink-head detections |', '|---:|---|---|']
    for r in pink:
        detections=', '.join(f"{d['label']} ({d['confidence']:.1%})" for d in r['detections']) or 'none'
        lines.append(f"| {r['frame']} | v{r['version']} | {detections} |")
    lines += ['', '## Artifacts and preservation','',
        '- `training/round6/models/helmet-v6-best.pt`',
        '- `training/round6/models/license-plate-v6-best.pt`',
        '- Matching `*-v6-last.pt` exports and complete training run checkpoints are also retained.',
        f"- At training completion, verified unchanged: {complete['prior_weights_unchanged']} previous weight files, {complete['source_files_unchanged']} source image/label files, and `.env`.",
        '- Resumption checks again verified the earlier weights and v6 export hashes. Configuration had changed '
        'since training and now selects v6; the resumption did not overwrite it.',
        '- `training/round6/data-audit.json`: per-image provenance, split, exclusions and duplicate decisions.',
        '- `training/round6/metrics.json`, `artifact-checks.json`, `COMPLETE.json`: metrics and SHA-256 checks.',
        '- `training/round6/training.log`, `runs/`, `evaluation/`: logs, history, plots and checkpoints.',
        '- `training/round6/video-checks/results.json`: diagnostic jobs and evidence paths.',
        '- `training/round6/pink-helmet-check/`: fixed-frame predictions and images.',
        '', '## Reproduction','',
        'Scripts refuse to overwrite datasets or runs. From a fresh output location:', '',
        '```powershell', '.venv/Scripts/python.exe training/prepare_round6.py',
        '.venv/Scripts/python.exe training/train_round6.py',
        '.venv/Scripts/python.exe training/check_round6_videos.py',
        '.venv/Scripts/python.exe training/check_round6_helmet_views.py',
        '.venv/Scripts/python.exe training/report_round6.py', '```', '']
    (ROOT/'docs/model-training-round6.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote docs/model-training-round6.md')


if __name__=='__main__':
    main()
