"""Reproduce event metrics from saved predictions and visual annotations; no inference."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
# Approximate visibility windows in seconds. Identity, not temporal proximity,
# determines matching when motorcycles overlap (especially IMG_7080).
annotations = {
    7075: [(0.5, 10.5, 'Two bareheaded occupants in white tops on red scooter')],
    7076: [(4, 6, 'Two occupants wearing brimmed hats, not helmets')],
    7077: [],
    7078: [(7, 14.5, 'Bareheaded solo rider in blue/purple shirt')],
    7079: [(2.5, 3.67, 'Two bareheaded occupants in white tops with black backpacks')],
    7080: [(6, 11.5, 'Bareheaded black-shirt rider with graphic backpack on yellow/orange motorcycle'),
           (17, 21, 'Bareheaded black long-sleeve rider on black/green scooter, plate digits 5528'),
           (18, 22, 'Bareheaded J&T courier in gray/white/red jacket with loaded side bags'),
           (33, 40.5, 'Bareheaded white-shirt passenger behind dark-clothed driver')],
    7081: [(6, 9.5, 'Bareheaded solo white-shirt rider on black motorcycle')],
    7082: [(8, 14.5, 'Bareheaded dark-blue-shirt solo rider'),
           (13, 17.03, 'Bareheaded white-shirt driver and black-jacket ponytail passenger on lime scooter')],
    7083: [(4, 5.83, 'Bareheaded white-shirt passenger behind helmeted green-shirt driver')],
    7084: [(5, 8.5, 'Bareheaded white-shirt passenger behind helmeted patterned/vest driver'),
           (24, 25.63, 'Bareheaded buzzcut Grab delivery rider, plate digits 7390')],
    7085: [(1.5, 3.5, 'Bareheaded long-haired white-top passenger behind helmeted red-shirt driver')],
    7086: [(10, 19.5, 'Bareheaded blue-shirt rider in beige trousers on red/white motorcycle'),
           (20.5, 23.5, 'Bareheaded white-shirt rider with black sling bag on teal scooter')],
    7087: [(6, 8.5, 'Bareheaded blonde passenger in white T-shirt and black skirt holding drink')],
    7088: [(8.5, 11.73, 'LINE MAN rider wearing black cap, with delivery box')],
    7089: [(15.5, 19.5, 'Bareheaded black-shirt rider with gray backpack on black sporty motorcycle'),
           (23, 28, 'Bareheaded brown-haired passenger in black hoodie with hood down')],
    7090: [(8.5, 14.5, 'Bareheaded dark-navy-shirt solo rider on red/black scooter')],
    7091: [(11, 14.33, 'Bareheaded long-haired white-shirt passenger behind helmeted delivery driver')],
    7092: [],
    7093: [],
    7094: [(11, 14, 'Bareheaded female passenger in black top and white trousers, motorcycle plate digits 1380'),
           (8.5, 13, 'Bareheaded gray-haired green delivery rider, plate digits 1742; corrected after full-resolution source review during v3 evaluation')],
}
# Every saved record was visually associated with an annotated motorcycle.
# Explicit frame-to-identity assignments avoid overlapping-window false matches.
matches = {
    7075: {54: 1, 57: 1}, 7076: {138: 1, 148: 1}, 7078: {249: 1},
    7079: {96: 1, 106: 1}, 7080: {552: 2, 563: 2, 1014: 4},
    7081: {222: 1}, 7083: {156: 1, 159: 1, 173: 1},
    7084: {177: 1, 198: 1, 214: 1}, 7085: {76: 1},
    7086: {384: 1, 387: 1, 381: 1, 411: 1, 663: 2, 665: 2},
    7088: {263: 1}, 7089: {504: 1, 527: 1},
    7090: {306: 1, 280: 1, 310: 1}, 7091: {360: 1}, 7094: {366: 1},
}
labels = {
    'annotation_method': 'Single AI assistant visual review of source road-region contact sheets every 15 frames (~0.5 seconds), including final frames, plus targeted full-resolution inspection. Saved detection evidence then matched by motorcycle identity. Not independently human-validated; not exhaustive frame-by-frame review.',
    'unit': 'One motorcycle passage with at least one visibly unhelmeted occupant; multiple unhelmeted occupants count once.',
    'revision': 'During v3 review, full-resolution IMG_7094 frame 270 showed gray hair on the delivery rider previously mistaken for a helmet. Added 7094-E2, raising confirmed events from 24 to 25. V2 prediction matches are unchanged; recall corrected to 32%, 52%, 68%.',
    'source': 'C:/Users/ADMIN/Downloads/Sp data 2 september',
    'uncertain': [{'clip': 'IMG_7078.MOV', 'start_seconds': 15, 'end_seconds': 18.03,
                   'description': 'Checkered-shirt rider with white cloth covering head; concealed helmet cannot be ruled out.',
                   'treatment': 'Excluded from primary denominator; missed in all runs if counted positive.'}],
    'limitations': ['IMG_7082 source temporarily obscured/out of focus around 6.5–7.5 seconds.',
                    'Visibility windows are approximate; no bounding-box annotations or OCR ground truth.',
                    'Reviewer had previously seen predictions; this is not a blinded evaluation.'],
    'clips': [],
}
for item in data['inventory']:
    n = int(Path(item['clip']).stem.split('_')[1])
    labels['clips'].append({'clip': item['clip'], 'events': [
        {'id': f'{n}-E{i}', 'start_seconds': a, 'end_seconds': b, 'description': desc}
        for i, (a, b, desc) in enumerate(annotations[n], 1)],
        'source_review': f'review/labeling/IMG_{n}-0.jpg'})
assert len(labels['clips']) == 20
total = sum(len(c['events']) for c in labels['clips'])
assert total == 25
scored = []
for run in data['runs']:
    per_clip, records, tp, fp, duplicates = [], 0, 0, 0, 0
    for clip in run['clips']:
        n = int(Path(clip['clip']).stem.split('_')[1])
        seen, assignments = set(), []
        for v in clip['violations']:
            frame = v['frame_number']
            assert frame in matches.get(n, {}), f'Unreviewed prediction: {n}, {frame}'
            event = matches[n][frame]
            a, b, _ = annotations[n][event - 1]
            assert a - .5 <= frame / 30 <= b + .5
            duplicate = event in seen
            assignments.append({'frame': frame, 'track_id': v['track_id'],
                                'event_id': f'{n}-E{event}', 'duplicate': duplicate})
            seen.add(event)
        count = len(assignments)
        records += count
        tp += len(seen)
        duplicates += count - len(seen)
        per_clip.append({'clip': clip['clip'], 'ground_truth': len(annotations[n]),
                         'detected_events': len(seen), 'records': count,
                         'missed_events': [f'{n}-E{i}' for i in range(1, len(annotations[n])+1) if i not in seen],
                         'assignments': assignments})
    precision, recall = tp / records, tp / total
    scored.append({'interval_seconds': run['interval'], 'ocr': run['ocr'],
                   'runtime_seconds': run['wall_seconds'], 'ground_truth_events': total,
                   'records': records, 'true_positive_events': tp, 'false_negative_events': total-tp,
                   'false_positive_unmatched_records': fp, 'duplicate_records': duplicates,
                   'event_precision_duplicates_penalized': precision, 'event_recall': recall,
                   'event_f1_duplicates_penalized': 2*precision*recall/(precision+recall),
                   'helmet_correct_record_precision': (records-fp)/records,
                   'recall_if_uncertain_positive': tp/(total+1), 'clips': per_clip})
assert [r['true_positive_events'] for r in scored] == [8, 13, 17, 13]
assert [r['duplicate_records'] for r in scored] == [0, 0, 2, 0]
(OUT / 'labels.json').write_text(json.dumps(labels, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
(OUT / 'accuracy.json').write_text(json.dumps(scored, indent=2)+'\n', encoding='utf-8')
lines = ['# September video accuracy evaluation', '',
    'Visual review found **25 confirmed violation events across 20 clips (326.44 seconds)**. One additional head-cloth case is uncertain and excluded. A violation event is one motorcycle passage with at least one unhelmeted occupant; two bareheaded people on one motorcycle count once.', '',
    '| Sample interval | Base FPS | OCR | Found / 25 | Recall | Event precision* | F1* | Duplicate records | Runtime |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|']
for r in scored:
    lines.append(f"| {r['interval_seconds']}s | {1/r['interval_seconds']:g} | {r['ocr']} | {r['true_positive_events']}/25 | {r['event_recall']:.1%} | {r['event_precision_duplicates_penalized']:.1%} | {r['event_f1_duplicates_penalized']:.1%} | {r['duplicate_records']} | {r['runtime_seconds']:.1f}s |")
lines += ['', '*Event precision allows one true-positive record per ground-truth event; duplicate records count against precision. No saved helmet alert was visually judged to be an unrelated/nonviolating motorcycle. Consequently helmet-correct record precision is 100% in each run, but the 0.25s run has only 17 unique events among 19 records (89.5% event precision). This does not measure correct plate association or OCR.', '',
    'Label correction: full-resolution review during the v3 comparison confirmed that the first IMG_7094 delivery rider has gray hair, not a helmet. This adds a 25th event and supersedes the previously reported 24-event denominator. V2 recall changes from 33.3/54.2/70.8% to 32/52/68%; detected counts remain 8/13/17.', '',
    '## Interpretation', '',
    'The 1s default interval catches 32% of confirmed violations. Moving to 0.5s catches five more events (+20 percentage points recall). Moving to 0.25s catches another four (+16 points), but still misses eight and adds two duplicates. OCR at 0.5s does not change event recall on these clips. No settings were changed.', '',
    'If the uncertain IMG_7078 head-cloth case is a violation, the denominator becomes 26 and recall is 30.8%, 50%, and 65.4% at 1s, 0.5s, and 0.25s respectively.', '',
    'These are single-reviewer AI visual annotations, not independently human-validated ground truth. Review covered source contact sheets at ~0.5s spacing across every clip, plus targeted full-resolution frames and detection evidence. This was not a blinded or every-frame annotation. Report event precision/recall rather than a generic accuracy percentage: an event detector has no natural count of true-negative events.', '',
    '## Per-clip confirmed events', '',
    '| Clip | Confirmed events | Found at 1s | Found at 0.5s | Found at 0.25s |',
    '|---|---:|---:|---:|---:|']
for i, c in enumerate(labels['clips']):
    lines.append(f"| {c['clip']} | {len(c['events'])} | " + ' | '.join(str(r['clips'][i]['detected_events']) for r in scored[:3]) + ' |')
lines += ['', '## Misses remaining at 0.25s', '']
for c in scored[2]['clips']:
    n = int(Path(c['clip']).stem.split('_')[1])
    for eid in c['missed_events']:
        a, b, desc = annotations[n][int(eid.split('-E')[1])-1]
        lines.append(f'- {eid}, around {a:g}–{b:g}s: {desc}.')
lines += ['', '## Evidence and performance limitations', '',
    '- Duplicate pairs: IMG_7086 frames 381/411 and IMG_7090 frames 280/310.',
    '- IMG_7094 at 0.5s with OCR associates plate digits 1742 from the preceding bareheaded delivery rider with the violating motorcycle; the latter shows digits 1380. A correct helmet alert can still have incorrect evidence association.',
    '- OCR correctness has not been fully annotated. Nonempty OCR is not accuracy; observed errors include IMG_7079 digits 482 read as 402.',
    '- Runtime comes from single sequential passes; the 1s run includes cold detector startup. Excluding the first clip, 0.5s costs about 30% more runtime than 1s. The complete 0.25s pass costs 2.42 times the 0.5s pass.', '',
    'Artifacts: [labels and methodology](labels.json), [all record-to-event assignments and metrics](accuracy.json), [benchmark results](report.md), [detection evidence](evidence.html). Source review sheets are in review/labeling/. Run `python analysis/september-evaluation/score_saved.py` to reproduce these metrics from saved results and the visual annotations embedded in that script.', '']
(OUT / 'accuracy-report.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps([{k: r[k] for k in ('interval_seconds', 'ocr', 'true_positive_events', 'false_negative_events', 'duplicate_records', 'event_recall')} for r in scored], indent=2))
