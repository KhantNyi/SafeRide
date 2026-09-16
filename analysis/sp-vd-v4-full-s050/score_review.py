"""Recompute passage-level scores from explicit visual adjudications."""
import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_v4 import write_report, model_inventory

data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
labels = json.loads((OUT / 'reference-labels.json').read_text(encoding='utf-8'))
records = json.loads((OUT / 'record-adjudications.json').read_text(encoding='utf-8'))
assert {c['clip'] for c in labels['clips']} == {c['clip'] for c in data['clips']}
assert {r['record_id'] for r in records} == {v['id'] for c in data['clips'] for v in c['violations']}
events = {e['id']: dict(e, clip=c['clip']) for c in labels['clips'] for e in c['events']}
matched = {r['event_id'] for r in records if r['status'] == 'true_positive'}
assert matched <= events.keys()
assert len(matched) == sum(r['status'] == 'true_positive' for r in records)
counts = Counter(r['status'] for r in records)
tp = counts['true_positive']
precision = tp / (tp + counts['false_positive'] + counts['duplicate'])
recall = tp / len(events)
metrics = dict(confirmed_passages=len(events), correctly_recorded_passages=tp,
    missed_passages=len(events)-tp, record_statuses=dict(counts),
    precision=precision, recall=recall, f1=2*precision*recall/(precision+recall))
assert model_inventory() == data['models_before'], 'Model inventory changed'
metadata = [json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'media/metadata').glob('*_detections.json')]
assert len(metadata) == 24
assert sum(len(m['frames']) for m in metadata) == 923
assert all(all(b['frame_number']-a['frame_number'] == 30 for a,b in zip(m['frames'],m['frames'][1:])) for m in metadata)
write_report(OUT, data)
(OUT/'accuracy-metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
lines = ['# SafeRide v4: Sp vd visual evaluation', '',
    '## Result', '',
    f'V4 correctly recorded **{tp} of {len(events)} confirmed motorcycle passages ({recall:.1%} recall)**. It missed {len(events)-tp}. Of 29 saved records, 23 correctly identify unique passages, 4 are duplicates, 1 targets the wrong motorcycle, and 1 is uncertain.', '',
    f'**Precision: {precision:.1%}; recall: {recall:.1%}; F1: {metrics["f1"]:.1%}.** Precision penalizes duplicate records and excludes the one uncertain record. These measure the complete saved-alert pipeline, including detection, association, tracking and alert rules; they are not detector mAP.', '',
    '## Test configuration and runtime', '',
    '- 24 clips from `C:\\Users\\ADMIN\\Downloads\\Sp vd`; all completed.',
    '- 456.58 seconds of source video; 923 sampled frames; 103.15 seconds processing (about 4.43 times video duration per processing second). Visual-review time is separate.',
    '- Helmet-v4 and plate-v4, existing YOLO11s object detector, GPU 0; full-frame inference with helmet crop inference disabled.',
    '- Interpreted requested 0.5 sample rate as one frame every 0.5 seconds (2 fps), not 0.5 fps. Fixed sampling, adaptive sampling off; metadata confirms 30-frame spacing.',
    '- Image size 960; confidence thresholds: object 0.35, helmet 0.35, plate 0.30. OCR enabled.', '',
    '## Visual reference method', '',
    'I visually reviewed source-only contact sheets at 0.5-second spacing for all clips, with targeted full-resolution source frames to resolve heads. A positive event is one motorcycle passage with at least one visibly unhelmeted rider or passenger. Multiple unhelmeted people on one motorcycle count once. I then matched every saved alert by motorcycle identity and its highlighted target, not timestamp alone.', '',
    'These are AI visual reference labels requested by the user, not independently human-validated ground truth. Hood-covered, distant or clipped heads that cannot be resolved are excluded. Short events entirely between samples could be missed by this reference review. No per-person, bounding-box, true-negative accuracy or OCR exact-match score is claimed. Source labels, uncertainty notes and frame references are in [reference-labels.json](reference-labels.json). Every alert decision is in [record-adjudications.json](record-adjudications.json).', '',
    '## Per-clip passage results', '',
    '| Clip | Confirmed passages | Correctly recorded | Missed | Duplicate records | Wrong-target/false records | Uncertain records |',
    '|---|---:|---:|---:|---:|---:|---:|']
for c in labels['clips']:
    rr=Counter(r['status'] for r in records if r['clip']==c['clip'])
    lines.append(f"| {c['clip']} | {len(c['events'])} | {rr['true_positive']} | {len(c['events'])-rr['true_positive']} | {rr['duplicate']} | {rr['false_positive']} | {rr['uncertain']} |")
lines += ['', '## Missed passages', '', '| Event | Clip | Approx. start (s) | Visual description |', '|---|---|---:|---|']
for eid,e in events.items():
    if eid not in matched:
        lines.append(f"| {eid} | {e['clip']} | {e['start_seconds']} | {e['description']} |")
lines += ['', '## Main failure examples', '',
    '- IMG_6665, frame 540: a bare head on the motorcycle ahead is assigned to a helmeted foreground rider and plate 3872. The wrong-target record is penalized, and the actual unhelmeted passage remains missed.',
    '- Duplicate records: IMG_6705 tracks 4/3; IMG_6779 tracks 6/8; IMG_6798 tracks 5/6 and 13/14. Different track IDs recorded the same physical passage twice.',
    '- IMG_6712, IMG_6714, IMG_6774 and IMG_6776 contain confirmed events but produced no saved records. The final alert pipeline therefore needs better recall; these results alone do not isolate whether detection, association or alert gating caused each miss.', '',
    '## Plate and OCR output', '',
    f"20 of 29 records saved a plate crop; 10 produced nonempty OCR text. Pipeline OCR statuses: {data['totals']['plate_statuses']}.", '',
    'These are output-availability counts, not recognition accuracy. Full plate strings were not independently transcribed and scored. The wrong-motorcycle example also shows that detecting a plate does not ensure it belongs to the offending rider.', '',
    '## Preservation and artifacts', '',
    'All inventoried baseline/v1/v2/v3/v4 model weights match their pre-benchmark SHA-256 hashes. Previous training reports and app configuration remain preserved. Round-four training is recorded in [model-training-round4.md](../../docs/model-training-round4.md). This benchmark used its own database and media directory.', '',
    '- [Raw benchmark report](report.md)',
    '- [Evidence gallery](evidence.html)',
    '- [Machine-readable metrics](accuracy-metrics.json)',
    '- [Full settings and results](results.json)', '',
    'Conclusion: on these clips, full-frame v4 at a 0.5-second interval runs faster than video playback, but the saved-alert pipeline misses 42.5% of confirmed passages and creates duplicate records. Association and tracking errors should be addressed alongside detector recall before treating the records as reliable final results.', '']
(OUT/'accuracy-report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(metrics,indent=2))
