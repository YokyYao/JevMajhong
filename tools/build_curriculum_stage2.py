"""Tag complete expert stage2 data with explicit replay/call/kan loss masses."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def visible(row):
    return dict(state=row['state'], questions={k: {a: b for a, b in q.items()
        if a in ('type', 'instructions', 'criteria')} for k, q in row['questions'].items()})


def category(row):
    phase = row['_meta']['phase']
    if phase == 'closed_discard':
        q = row['questions']['action']
        if any(c['type'] == 'reach' for c in q['criteria'].values()):
            return 'riichi_yes' if q['criteria'][q['label']]['type'] == 'reach' else 'riichi_no'
        return 'plain'
    if phase in ('open_discard', 'call_window', 'kan_window'):
        return {'open_discard': 'open', 'call_window': 'call', 'kan_window': 'kan'}[phase]
    raise ValueError('Unregistered expert phase: '+phase)


def build(data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    data = read(data_ready)
    pool = {}
    for split, spec in data['stages']['stage2'].items():
        path = ROOT/spec['path']
        assert sha(path) == spec['sha256'], path
        pool[split] = rows(path)
        assert all(r['_meta']['label_type'] == 'expert_replay' for r in pool[split])
    counts = {split: Counter(category(r) for r in part) for split, part in pool.items()}
    expected = {'plain', 'riichi_yes', 'riichi_no', 'open', 'call', 'kan'}
    assert set(counts['train']) == set(counts['validation']) == expected
    n = len(pool['train'])
    riichi = counts['train']['riichi_yes']+counts['train']['riichi_no']
    mass = dict(plain=.25, open=.25, call=.25, kan=.15,
                riichi_yes=.1*counts['train']['riichi_yes']/riichi,
                riichi_no=.1*counts['train']['riichi_no']/riichi)
    weights = {k: mass[k]*n/counts['train'][k] for k in mass}
    assert abs(sum(mass.values())-1) < 1e-12 and max(weights.values()) <= 8
    groups = {split: {r['_meta']['group_id'] for r in part} for split, part in pool.items()}
    ids = {split: {r['_meta']['id'] for r in part} for split, part in pool.items()}
    reserved = set(read(Path(data_ready).parent/'reserved-public-groups.json'))
    assert not groups['train'] & (groups['validation'] | reserved)
    assert not ids['train'] & ids['validation']
    later = data['stages']['stage3']['train']
    assert sha(ROOT/later['path']) == later['sha256']
    assert not {r['_meta']['group_id'] for r in rows(ROOT/later['path'])} & (groups['validation'] | reserved)
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, part in pool.items():
        path = out/f'{split}.jsonl'
        with path.open('x') as f:
            for source in part:
                item = copy.deepcopy(source)
                item['questions']['action']['src'] = 'curriculum2:'+category(source)
                assert visible(item) == visible(source)
                assert item['questions']['action']['label'] == source['questions']['action']['label']
                f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':'))+'\n')
        files[split] = dict(path=str(path), records=len(part), groups=len(groups[split]), sha256=sha(path))
    result = dict(status='stage2_data_prepared_pending_parent_and_fit_registration',
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], files=files,
        original_source_sha256={k: v['sha256'] for k, v in data['stages']['stage2'].items()},
        categories={k: dict(v) for k, v in counts.items()}, loss_mass=mass, example_weights=weights,
        all_original_rows_once=True, copies_added=0, labels_unchanged=True,
        model_visible_inputs_unchanged=True, train_validation_group_overlap=0,
        reserved_groups_excluded_from_this_and_prepared_stage3_training=True,
        stage1_replay_records=dict(train=data['stages']['stage1']['train']['records'],
                                  validation=data['stages']['stage1']['validation']['records']),
        parent_model_admitted=False, optimizer_updates=0,
        note='Proposed stage2 dataset only. Closed plain/riichi replay receives25%/10%, open25%, call25%, kan15% loss mass. Overlapping call/kan windows retain their original primary phase. A fit still requires an accepted parent and a frozen training/evaluation recipe.')
    with (out/'data-ready.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.data_ready, a.out)
