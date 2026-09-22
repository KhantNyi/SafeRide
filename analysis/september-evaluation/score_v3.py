"""Score v3 after visual review assigns every saved record to an event or false alert."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'v3'
data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
labels = json.loads((ROOT / 'labels.json').read_text(encoding='utf-8'))
review = json.loads((OUT / 'record-labels.json').read_text(encoding='utf-8'))
baseline = json.loads((ROOT / 'accuracy.json').read_text(encoding='utf-8'))
by_clip = {c['clip']: c['events'] for c in labels['clips']}
total = sum(map(len, by_clip.values()))
assert len(data['runs']) == 4
scored = []
for run in data['runs']:
    assert len(run['clips']) == 20
    tp = fp = duplicates = uncertain = records = 0
    clips = []
    for c in run['clips']:
        assert c['job']['status'] == 'completed'
        seen, assignments = set(), []
        valid = {e['id'] for e in by_clip[c['clip']]}
        for v in c['violations']:
            key = f"{run['interval']}:{run['ocr']}:{c['clip']}:{v['frame_number']}:{v['track_id']}"
            event = review[key]
            assert event in valid | {'false_positive', 'uncertain'}
            duplicate = event in seen if event in valid else False
            if event == 'false_positive': fp += 1
            elif event == 'uncertain': uncertain += 1
            elif duplicate: duplicates += 1
            else: tp += 1
            if event in valid: seen.add(event)
            records += 1
            assignments.append({'key': key, 'event': event, 'duplicate': duplicate})
        clips.append({'clip': c['clip'], 'ground_truth': len(valid), 'detected_events': len(seen),
                      'missed_events': sorted(valid-seen), 'assignments': assignments})
    denominator = records-uncertain
    precision = tp/denominator if denominator else 0
    recall = tp/total
    scored.append({'interval_seconds': run['interval'], 'ocr': run['ocr'],
                   'runtime_seconds': run['wall_seconds'], 'records': records,
                   'true_positive_events': tp, 'false_negative_events': total-tp,
                   'false_positive_unmatched_records': fp, 'duplicate_records': duplicates,
                   'uncertain_records_excluded': uncertain,
                   'event_precision_duplicates_penalized': precision, 'event_recall': recall,
                   'event_f1_duplicates_penalized': 2*precision*recall/(precision+recall) if precision+recall else 0,
                   'clips': clips})
(OUT / 'accuracy.json').write_text(json.dumps(scored, indent=2)+'\n', encoding='utf-8')
lines = ['# September benchmark: v3 versus v2', '',
    'All 20 clips (326.44 seconds) were rerun with helmet-v3 and plate-v3. The same 25 confirmed motorcycle-passage labels are used for both versions; one ambiguous head-cloth case is excluded. Labels are AI visual annotations from source review at approximately 0.5s spacing and targeted full-resolution inspection, not independently human-validated.', '',
    '| Interval | OCR | V2 found / 25 | V3 found / 25 | V3 recall | V3 precision* | V3 false alerts | V3 duplicates | V2 runtime | V3 runtime |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for old, new in zip(baseline, scored):
    assert (old['interval_seconds'], old['ocr']) == (new['interval_seconds'], new['ocr'])
    lines.append(f"| {new['interval_seconds']}s | {new['ocr']} | {old['true_positive_events']}/25 | {new['true_positive_events']}/25 | {new['event_recall']:.1%} | {new['event_precision_duplicates_penalized']:.1%} | {new['false_positive_unmatched_records']} | {new['duplicate_records']} | {old['runtime_seconds']:.1f}s | {new['runtime_seconds']:.1f}s |")
lines += ['', '*Precision counts only one true-positive alert per event, penalizing duplicate records. Helmet event correctness does not establish plate association or OCR correctness.', '',
    'V3 materially improves recall at every interval. On this sample, 1s offers the strongest speed/duplicate tradeoff: 19 events in 157.3s with one duplicate. At 0.5s, one more event costs 52.0s and increases duplicates to four. At 0.25s, another event costs 368.7s and increases duplicates to seven. The 0.5s OCR pass catches the same 20 events in 216.4s. These observations do not change application defaults.', '',
    'Label correction: the earlier report used 24 events. Full-resolution source frame 270 in IMG_7094 confirms gray hair on the first delivery rider previously mistaken for a helmet. Adding that event gives 25. V2 and v3 are both scored against this corrected denominator; the original v2 detections are unchanged. [Source close-up](review/7094-source270.jpg).', '',
    '## Per-clip v3 detections', '',
    '| Clip | Ground truth | 1s | 0.5s | 0.25s | 0.5s + OCR |',
    '|---|---:|---:|---:|---:|---:|']
for i, c in enumerate(labels['clips']):
    lines.append(f"| {c['clip']} | {len(c['events'])} | " + ' | '.join(str(r['clips'][i]['detected_events']) for r in scored) + ' |')
lines += ['', '## What remains at 0.25 seconds', '']
for c in scored[2]['clips']:
    for event in by_clip[c['clip']]:
        if event['id'] in c['missed_events']:
            lines.append(f"- {event['id']}, around {event['start_seconds']:g}s: {event['description']}.")
lines += ['', 'V3 is better overall but does not recover every v2 detection. At 0.5s and 0.25s, v2 caught the black/green scooter rider in IMG_7080 (E2) and the female passenger in IMG_7094 (E1); v3 misses both. V3 detects the separate gray-haired delivery rider in IMG_7094 instead.', '',
    '## Plate output coverage', '',
    '| Interval | OCR | Saved records | Plate crops | Nonempty OCR |',
    '|---|---|---:|---:|---:|']
for run in data['runs']:
    vs = [v for c in run['clips'] for v in c['violations']]
    lines.append(f"| {run['interval']}s | {run['ocr']} | {len(vs)} | {sum(bool(v['plate_image']) for v in vs)} | {sum(bool(v['plate_text']) for v in vs)} |")
lines += ['', 'Plate coverage counts include duplicate records and do not measure transcription accuracy.']
lines += ['', '## Method and limitations', '',
    '- Both specialist weights changed together; this measures the combined v3 pipeline, not the isolated contribution of either model.',
    '- Weights: training/round3/models/helmet-v3-best.pt and training/round3/models/license-plate-v3-best.pt. Both were verified to match the corresponding round3 training-run best.pt files by SHA-256 before this rerun.',
    '- All other resolved inference settings match v2: full-frame helmet inference, confidence 0.35, plate confidence 0.30, adaptive sampling enabled. Base rates are 1, 2, and 4 FPS; adaptive sampling analyzes extra frames.',
    '- Separate benchmark database and media directory; application defaults and normal database were not modified.',
    '- Single sequential timing passes. The first run in each model benchmark includes cold detector startup; timings are not repeated controlled estimates.',
    '- Every saved v3 record was visually matched by motorcycle identity. Overlapping time windows alone were not used to assign identity.',
    '- Small, single-location sample; no claim of general production accuracy or complete OCR accuracy.', '',
    'Artifacts: [raw v3 results](results.json), [scored assignments](accuracy.json), [visual record labels](record-labels.json), [v3 evidence browser](evidence.html), [shared source labels](../labels.json), [v2 accuracy report](../accuracy-report.md).', '',
    'Reproduce: `.venv/Scripts/python.exe analysis/september-evaluation/run_benchmark.py --v3`, then visually label the new records before running `python analysis/september-evaluation/score_v3.py`. Record IDs depend on inference output; the scorer rejects unreviewed records.', '']
(OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k != 'clips'} for r in scored], indent=2))
