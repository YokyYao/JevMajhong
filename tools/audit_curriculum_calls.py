"""Audit independent-source call/kan coverage before registering stage2."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ('chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'reach')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def visible(row):
    return dict(state=row['state'], questions={k: {a: b for a, b in q.items()
        if a in ('type', 'instructions', 'criteria')} for k, q in row['questions'].items()})


def selected(row):
    q = row['questions']['action']
    return q['criteria'][q['label']]['type']


def is_open_hand(state):
    # Match the public pool's phase classification: ankan does not break menzen.
    own = state['melds'][state['seat']]
    return any(m['type'] not in (3, 'ankan') for m in own)


def audit(data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    data = read(data_ready)
    pools = {}
    for stage in ('stage1', 'stage2'):
        pools[stage] = {}
        for split, spec in data['stages'][stage].items():
            path = ROOT/spec['path']
            with path.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == spec['sha256'], path
            pools[stage][split] = rows(path)
    reserved = set(read(Path(data_ready).parent/'reserved-public-groups.json'))
    result = {}
    for split, pool in pools['stage2'].items():
        labels = Counter(r['_meta']['label_type'] for r in pool)
        assert set(labels) == {'expert_replay'}
        stage1 = pools['stage1'][split]
        byid = {r['_meta']['id']: r for r in pool}
        assert len(byid) == len(pool)
        for row in stage1:
            other = byid[row['_meta']['id']]
            assert visible(row) == visible(other)
            assert row['questions']['action']['label'] == other['questions']['action']['label']
        operations = {}
        for op in OPERATIONS:
            windows = [r for r in pool if any(c['type'] == op for c in r['questions']['action']['criteria'].values())]
            positive = [r for r in windows if selected(r) == op]
            negative = [r for r in windows if selected(r) != op]
            def contexts(part):
                counts = Counter()
                for r in part:
                    state = r['state']
                    counts['closed'] += not is_open_hand(state)
                    counts['open'] += is_open_hand(state)
                    counts['concealed_kan_present'] += any(m['type'] in (3, 'ankan') for m in state['melds'][state['seat']])
                    counts['dealer'] += state['seat'] == state['dealer']
                    counts['opponent_riichi'] += any(p != state['seat'] for p in state['riichi'])
                    river = state['rivers'][state['seat']]
                    turn = len(river.split()) if isinstance(river, str) else len(river)
                    counts['early_0_5' if turn < 6 else 'middle_6_11' if turn < 12 else 'late_12_plus'] += 1
                return dict(counts)
            operations[op] = dict(opportunities=len(windows), selected=len(positive), not_selected=len(negative),
                positive_source_groups=len({r['_meta']['group_id'] for r in positive}),
                negative_source_groups=len({r['_meta']['group_id'] for r in negative}),
                positive_contexts=contexts(positive), negative_contexts=contexts(negative),
                decline_alternatives=dict(Counter(selected(r) for r in negative)))
        phases = Counter(r['_meta']['phase'] for r in pool)
        result[split] = dict(records=len(pool), source_groups=len({r['_meta']['group_id'] for r in pool}),
            labels=dict(labels), phases=dict(phases), selections=dict(Counter(selected(r) for r in pool)),
            stage1_exact_replay_records=len(stage1), stage1_public_inputs_and_labels_unchanged=True,
            operations=operations)
    train_groups = {r['_meta']['group_id'] for r in pools['stage2']['train']}
    validation_groups = {r['_meta']['group_id'] for r in pools['stage2']['validation']}
    assert not train_groups & (validation_groups | reserved)
    report = dict(status='stage2_data_coverage_audit_complete_not_fit', node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], splits=result,
        reserved_groups_excluded=True, train_validation_group_overlap=0,
        source_hashes_unchanged=True, copies_added=0, student_or_optimizer_loaded=False,
        caveat='Operation windows can overlap. Not-selected includes another call as well as pass/discard. Expert actions are imitation targets, not proof of optimal action. No stage2 fit or parent checkpoint has been admitted by this audit.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.data_ready, a.out)
