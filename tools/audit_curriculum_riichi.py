"""Audit frozen curriculum data and cached riichi predictions without fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def category(row):
    q = row['questions']['action']
    if any(v['type'] == 'reach' for v in q['criteria'].values()):
        return 'riichi_yes' if q['criteria'][q['label']]['type'] == 'reach' else 'riichi_no'
    return 'plain'


def mean(values):
    return statistics.mean(values) if values else None


def player_values(value):
    if isinstance(value, list) and len(value) == 4:
        return value
    if isinstance(value, dict) and set(value) == {'0', '1', '2', '3'}:
        return [value[str(i)] for i in range(4)]
    return None


def summarize_data(data):
    grouped = defaultdict(list)
    duplicate = defaultdict(list)
    for row in data:
        grouped[category(row)].append(row)
        q = row['questions']['action']
        visible = dict(state=row['state'], questions={k: {a: b for a, b in v.items()
                       if a in ('type', 'instructions', 'criteria')}
                       for k, v in row['questions'].items()})
        key = hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).hexdigest()
        duplicate[key].append(q['label'])
    result = {}
    for kind, part in grouped.items():
        groups = Counter(r['_meta']['group_id'] for r in part)
        menus = Counter()
        for row in part:
            types = Counter(c['type'] for c in row['questions']['action']['criteria'].values())
            menus[json.dumps(dict(sorted(types.items())), sort_keys=True)] += 1
        result[kind] = dict(records=len(part), source_groups=len(groups),
                            largest_group_records=max(groups.values()),
                            menu_type_counts=dict(menus))
    return dict(categories=result, duplicate_visible_requests=sum(len(v)-1 for v in duplicate.values()),
                conflicting_identical_requests=sum(len(set(v)) > 1 for v in duplicate.values()))


def summarize_predictions(data, predictions):
    lookup = {r['_meta']['id']: r for r in data}
    groups = defaultdict(lambda: defaultdict(list))
    contexts = defaultdict(Counter)
    for p in predictions:
        row = lookup[p['id']]
        kind = category(row)
        if kind == 'plain':
            continue
        q = row['questions']['action']
        probabilities = dict(zip(p['keys'], p['p']))
        assert set(probabilities) == set(q['criteria'])
        reach = [probabilities[k] for k, c in q['criteria'].items() if c['type'] == 'reach']
        other = [probabilities[k] for k, c in q['criteria'].items() if c['type'] != 'reach']
        assert reach and other
        best = max(probabilities, key=probabilities.__getitem__)
        predicted = q['criteria'][best]['type'] == 'reach'
        truth = kind == 'riichi_yes'
        g = groups[kind]
        g['accepted'].append(int(predicted))
        g['exact'].append(int(best == q['label']))
        g['reach_mass'].append(sum(reach))
        g['max_reach_minus_max_other'].append(max(reach)-max(other))
        g['near_tie_within_002'].append(int(abs(max(reach)-max(other)) <= .02))
        s = row['state']
        seat = int(s['seat'])
        declared = s['riichi']
        assert isinstance(declared, list) and all(type(p) is int and p in range(4) for p in declared)
        n = sum(p != seat for p in set(declared))
        context = f'opponents_riichi_{n}'
        contexts[context]['windows'] += 1
        contexts[context]['type_errors'] += truth != predicted
        rivers = player_values(s.get('rivers'))
        if rivers is not None and isinstance(rivers[seat], (list, str)):
            turn = len(rivers[seat].split()) if isinstance(rivers[seat], str) else len(rivers[seat])
            context = 'turn_0_5' if turn < 6 else 'turn_6_11' if turn < 12 else 'turn_12_plus'
            contexts[context]['windows'] += 1
            contexts[context]['type_errors'] += truth != predicted
    return dict(categories={k: dict(windows=len(v['accepted']),
                    accepted=sum(v['accepted']), exact_matches=sum(v['exact']),
                    mean_reach_probability_mass=mean(v['reach_mass']),
                    mean_max_reach_minus_max_other=mean(v['max_reach_minus_max_other']),
                    near_tie_within_002=sum(v['near_tie_within_002'])) for k, v in groups.items()},
                context_errors={k: dict(v) for k, v in contexts.items()})


def audit(registered, out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'), 'Scheduled GPU required'
    assert not socket.gethostname().startswith(('lo-', 'login')), 'Compute node required'
    reg = read(registered)
    root = Path(__file__).resolve().parents[1]
    for p, h in reg['inputs'].items():
        with Path(p).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == h, p
    ready = read(root / reg['curriculum']['data_root'] / 'data-ready.json')
    train = rows(root / ready['stages']['stage1']['train']['path'])
    validation = rows(root / ready['stages']['stage1']['validation']['path'])
    fit = root / reg['curriculum']['out']
    updates = rows(fit / 'metrics/train_metrics.jsonl')
    counts = [r['source_counts'].get('riichi_yes', 0) + r['source_counts'].get('riichi_no', 0) for r in updates]
    both = sum(r['source_counts'].get('riichi_yes', 0) > 0 and r['source_counts'].get('riichi_no', 0) > 0 for r in updates)
    predictions = []
    for point in read(fit / 'validation-curve.json'):
        p = read(fit / 'training-run/training_eval' / f"step-{point['step']:06d}" / 'development/rows.json')
        predictions.append(dict(step=point['step'], **summarize_predictions(validation, p)))
    stage_coverage = {}
    for stage in ('stage2', 'stage3'):
        stage_coverage[stage] = {}
        for split, spec in ready['stages'][stage].items():
            pool = rows(root / spec['path'])
            selected = Counter(r['questions']['action']['criteria'][r['questions']['action']['label']]['type'] for r in pool)
            opportunities = Counter()
            for row in pool:
                opportunities.update(set(c['type'] for c in row['questions']['action']['criteria'].values()))
            stage_coverage[stage][split] = dict(records=len(pool), selected=dict(selected), opportunities=dict(opportunities))
    result = dict(status='frozen_curriculum_riichi_data_audit_complete',
                  node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'],
                  original_inputs_unchanged=True, train=summarize_data(train), validation=summarize_data(validation),
                  state_field_types={k:type(v).__name__ for k,v in train[0]['state'].items()},
                  state_schema='riichi lists declaring actor indices, not boolean seat flags; rivers use four tile-name strings',
                  optimizer_batches=dict(total=len(counts), no_riichi=sum(n == 0 for n in counts),
                      both_accept_and_decline=both, riichi_count_histogram=dict(Counter(counts)),
                      mean_riichi_records=mean(counts)), prediction_points=predictions, later_stage_coverage=stage_coverage,
                  no_model_inference=True, no_optimizer_updates=True,
                  caveat='Descriptive diagnostics; teacher choices are not proven optimal; no causal explanation is established.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.registered, a.out)
