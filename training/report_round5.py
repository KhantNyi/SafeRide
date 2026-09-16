"""Write the v5 training report after all training and artifact checks finish."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'training/round5'


def main():
    complete=json.loads((OUT/'COMPLETE.json').read_text())
    metrics=complete['metrics']
    audit=json.loads((OUT/'data-audit.json').read_text())
    lines=[
        '# SafeRide v5 training report', '',
        '## Outcome', '',
        'Both specialist detectors were fine-tuned from v4 and exported as v5. '
        'Helmet detection and plate-box localization improved on the new footage, '
        'but both detectors regressed in mAP50 on the older validation sets, especially '
        'the plate detector. V5 is not a general replacement for v4 based on these results. '
        'The app remains configured to v4; its PaddleOCR selection is unchanged.', '',
        '## Data preparation', '',
        '- Inputs: `C:/Users/ADMIN/Downloads/train-video-footage` (53 pairs) and '
        '`C:/Users/ADMIN/Downloads/train-data5` (54 pairs). Original folders were preserved.',
        '- Merged destination: `train-data5/`, with 103 unique pairs and 480 boxes '
        '(73 Helmet, 108 License Plate, 116 Motorcycle, 88 No Helmet, 9 No License Plate, 86 Person).',
        '- Four duplicate copies removed. For IMG_7438, the annotation containing the '
        'visible plate was retained over the conflicting No License Plate annotation. '
        'IMG_7439 duplicates were consolidated; its ambiguous headwear is excluded from helmet training. '
        'The redundant IMG_7440 annotation was also removed.',
        '- Fourteen EXIF-rotated photos normalized to upright pixels. Fifty retained still images '
        'cropped around all labeled objects with context; boxes transformed to match. '
        'PNG output avoids further lossy compression. Footage crops remain as previously reviewed.',
        '- All pairs passed decoding, finite-coordinate, positive-size, class-ID and boundary checks. '
        'All 54 new images were visually inspected using contact sheets, with closer inspection '
        'of duplicate/headwear cases. All 53 footage file hashes match the earlier reviewed export.',
        '- Fourteen inputs overlap earlier raw exports exactly after orientation normalization. '
        'IMG_7272 overlaps an earlier validation capture and is excluded from both specialist datasets.',
        '- IMG_7439 and IMG_7278 have ambiguous fabric-covered heads and are additionally excluded '
        'from the helmet dataset. Their plate labels remain usable.',
        '- Helmet dataset: 86 train images (61 Helmet, 73 No Helmet boxes), '
        '14 validation images (10 Helmet, 12 No Helmet boxes).',
        '- Plate dataset: 88 train images (94 plate boxes), 14 validation images (13 plate boxes).',
        '- Validation clips: '+', '.join(audit['validation_groups'])+'. '
        'Whole clips stay together. All new still images from the same road/capture session stay in training. '
        'No exact decoded image or known capture ID crosses the new split.',
        '- The six raw class IDs are preserved, but only Helmet/No Helmet and License Plate are '
        'remapped into the two specialists. The new still export lacks Person annotations; '
        'this is not a fully labeled Person dataset. General object detection and OCR were not trained.', '',
        '## Training', '',
        '- Initialized from `training/round4/models/helmet-v4-best.pt` and '
        '`training/round4/models/license-plate-v4-best.pt`.',
        '- RTX 4070 SUPER 12 GB; PyTorch 2.12.0+cu130; Ultralytics 8.3.52.',
        '- 100 epoch limit, patience 30, image size 640, batch 16, workers 0, '
        'seed 0, deterministic mode, mixed precision. AdamW, initial learning rate 0.001, final multiplier 0.01.',
        '- Fine-tuned on the prepared merged set; no separate historical replay dataset was appended.',
    ]
    for task in ('helmet','plate'):
        with (OUT/f'runs/{task}-v5/results.csv').open() as f:
            rows=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        best=max(rows,key=lambda r:.1*r['metrics/mAP50(B)']+.9*r['metrics/mAP50-95(B)'])
        lines.append(f'- {task.title()}: {int(rows[-1]["epoch"])} epochs completed; '
                     f'best epoch by logged validation fitness: {int(best["epoch"])}.')
    lines += ['', '## Matched validation results', '',
        '| Set | Detector | Version | Precision | Recall | mAP50 | mAP50-95 |',
        '|---|---|---|---:|---:|---:|---:|']
    for split in ('new_holdout','round4_holdout','round3_holdout'):
        for task in ('helmet','plate'):
            for version in ('v4','v5'):
                m=metrics[task][version][split]
                values=[m[f'metrics/{key}(B)'] for key in ('precision','recall','mAP50','mAP50-95')]
                lines.append(f'| {split} | {task} | {version} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '## Interpretation and limits', '']
    for task in ('helmet','plate'):
        changes=[]
        for split in ('new_holdout','round4_holdout','round3_holdout'):
            a=metrics[task]['v4'][split]['metrics/mAP50(B)']
            b=metrics[task]['v5'][split]['metrics/mAP50(B)']
            changes.append(f'{split}: {a*100:.2f}% to {b*100:.2f}% ({(b-a)*100:+.2f} percentage points)')
        lines.append(f'- {task.title()} mAP50: '+'; '.join(changes)+'.')
    lines += [
        '- The new holdout contains only 14 images from five clips and selects the best checkpoint. '
        'It is validation, not an independent test. Older checks contain 30 and 17 images. '
        'Known old-validation overlaps were excluded from new training.',
        '- Footage labels are model-assisted and AI-visually-reviewed, not independently human-verified. '
        'Unknown similar scenes/captures remain a limitation of still-photo provenance.',
        '- The Sp vd collection was used in earlier benchmarking and now partly in training. '
        'It must not be presented as an independent v5 test. These metrics do not measure '
        'tracking, violation-event accuracy or OCR character recognition.', '',
        '## Artifacts and preservation', '',
        f'- SHA-256 verification confirms all {complete["prior_weights_unchanged"]} earlier checkpoint files '
        'under `models/` and `training/` are unchanged.',
        '- Source image and label hashes were verified unchanged. Removed duplicates remain in the source folder.',
    ]
    for task,filename in [('helmet','helmet'),('plate','license-plate')]:
        check=json.loads((OUT/f'{task}-artifact-check.json').read_text())
        lines += [f'- `training/round5/models/{filename}-v5-best.pt`: `{check["sha256"]}`. '
                  'Matches its best checkpoint and successfully loads and predicts on a validation image.']
    lines += [
        '- `training/round5/metrics.json`, `COMPLETE.json`: final results.',
        '- `training/round5/data-audit.json`, `train-data5/manifest.json`: audit, provenance and split decisions.',
        '- `training/round5/runs/`: best/last checkpoints, arguments, CSV history and plots.',
        '- `training/round5/evaluation/`: matched evaluations and diagnostic plots.',
        '- `training/round5/training.log`: complete training log.',
        '- The PowerShell log marks redirected Ultralytics stderr output as '
        '`NativeCommandError`. Both training runs, final evaluations, export/inference '
        'checks and the completion marker finished successfully; no Python traceback occurred.',
        '- `analysis/round5-audit/`: visual-review sheets and headwear close-ups.', '',
        '## Reproduction', '',
        'Using fresh output directories (scripts refuse to overwrite existing runs):', '',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe training\\prepare_round5.py',
        '.\\.venv\\Scripts\\python.exe training\\train_round5.py',
        '.\\.venv\\Scripts\\python.exe training\\report_round5.py',
        '```', '',
    ]
    path=ROOT/'docs/model-training-round5.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)


if __name__=='__main__': main()
