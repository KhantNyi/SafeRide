"""Compare visually reviewed fixed-pipeline results with the original v3 run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'v3-dedup-final'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
data = read(OUT/'results.json')
labels = read(ROOT/'labels.json')
review = read(OUT/'record-labels.json')
baseline = {r['interval_seconds']: r for r in read(ROOT/'v3/accuracy.json') if not r['ocr']}
truth = {c['clip']: {e['id'] for e in c['events']} for c in labels['clips']}
total = sum(map(len, truth.values()))
scored = []
for run in data['runs']:
    assert len(run['clips']) == 20 and all(c['job']['status']=='completed' for c in run['clips'])
    clips = []
    for c in run['clips']:
        seen = set()
        assignments = []
        for v in c['violations']:
            key = f"{run['interval']}:{run['ocr']}:{c['clip']}:{v['frame_number']}:{v['track_id']}"
            event = review[key]
            assert event in truth[c['clip']] | {'false_positive', 'uncertain'}
            assignments.append({'frame':v['frame_number'], 'track_id':v['track_id'],
                                'event':event, 'duplicate':event in seen})
            if event in truth[c['clip']]: seen.add(event)
        old = next(x for x in baseline[run['interval']]['clips'] if x['clip']==c['clip'])
        old_seen = truth[c['clip']]-set(old['missed_events'])
        clips.append({'clip':c['clip'], 'found':sorted(seen), 'lost_vs_baseline':sorted(old_seen-seen),
                      'gained_vs_baseline':sorted(seen-old_seen), 'assignments':assignments})
    records = sum(len(c['assignments']) for c in clips)
    found = sum(len(c['found']) for c in clips)
    duplicates = sum(a['duplicate'] for c in clips for a in c['assignments'])
    fp = sum(a['event']=='false_positive' for c in clips for a in c['assignments'])
    scored.append({'interval':run['interval'], 'runtime_seconds':run['wall_seconds'],
                   'records':records,'found':found,'recall':found/total,
                   'duplicates':duplicates,'false_positives':fp, 'clips':clips})
assert len(scored)==3
assert not any(c['lost_vs_baseline'] for r in scored for c in r['clips']), 'A previously detected event was lost'
(OUT/'accuracy.json').write_text(json.dumps(scored,indent=2)+'\n',encoding='utf-8')
lines = ['# Duplicate violation fix: v3 validation', '',
    'All 20 September clips were processed again with helmet-v3 and plate-v3 at 1s, 0.5s and 0.25s intervals. These are full inference runs, not filtered copies of previous results. OCR was disabled. The same corrected set of 25 visual event labels is used.', '',
    '| Interval | Before: events | After: events | Before: duplicates | After: duplicates | After: recall | After: false helmet alerts |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in sorted(scored,key=lambda r:r['interval'],reverse=True):
    b=baseline[r['interval']]
    lines.append(f"| {r['interval']}s | {b['true_positive_events']} | {r['found']} | {b['duplicate_records']} | {r['duplicates']} | {r['recall']:.0%} | {r['false_positives']} |")
lines += ['', '## Change', '',
    'Overlapping whole-bike and rear-bike detections could coexist in one frame, spawning independent tracker IDs for the same rider. The pipeline now aliases a newly created overlapping track to an existing violation identity when both boxes correspond to the same nearest person. Both boxes remain available for rider and plate association. Established tracks are not merged just because they overlap. Pending events remain active while any aliased raw track remains active. Simultaneously observed separate motorcycles are protected from later position-only save suppression.', '',
    '## Per-clip changes', '']
for r in sorted(scored,key=lambda r:r['interval'],reverse=True):
    for c in r['clips']:
        if c['lost_vs_baseline'] or c['gained_vs_baseline']:
            lines.append(f"- {r['interval']}s {c['clip']}: lost {c['lost_vs_baseline']}; gained {c['gained_vs_baseline']}.")
lines += ['', '## Remaining duplicates', '',
    '- 1s: IMG_7082, second motorcycle, frames 438 and 486. The earlier IMG_7086 duplicate is removed, but this extra record is newly exposed by protecting distinct passages from broad positional suppression.',
    '- 0.5s: IMG_7090, frames 345 and 417, after the tracker loses and reacquires the rider.',
    '- 0.25s: IMG_7080, courier, frames 590 and 627. These are one event under separate identities.',
    '- Duplicate counts are now 1 at each interval. This is a reduction at higher sampling, not complete deduplication. Resolving the remaining cases needs reliable identity matching across tracking gaps; widening a positional exclusion zone risks suppressing different riders.', '',
    '## Verification and limits', '',
    '- Regression tests cover the actual duplicate boxes from IMG_7086, separate nearby motorcycles, established overlapping tracks, and pending-event lifetime after the original raw ID disappears. Existing plate-collection tests also pass.',
    '- Visual event labels and output matching were performed by the AI assistant; there is no independent human validation. One uncertain head-cloth case remains excluded.',
    '- Results are specific to these clips. Crowded occlusions, similar riders, and track losses can still create identity errors. Higher sampling is not guaranteed to be duplicate-free.',
    '- Plate/OCR correctness is not established by helmet-event accuracy. Application model paths and sampling defaults were not changed.', '',
    '- Runtime is retained in raw results, but these validation runs used a different order (0.5s, 0.25s, 1s) from the original benchmark. No controlled speedup claim is made.', '',
    'Artifacts: [raw results](results.json), [record identity labels](record-labels.json), [scored identities](accuracy.json), [evidence browser](evidence.html), [original v3 benchmark](../v3/report.md).', '']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k!='clips'} for r in scored],indent=2))
