"""Quarantine structurally uninformative masked targets without relabelling."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def support_size(row):
    if row['_meta']['label_type'] == 'expert_replay':
        return None
    q = row['questions']['action']
    assert row['_meta']['label_type'] == q['src'] == 'paired_future_masked'
    assert set(q['target']) <= set(q['criteria'])
    assert all(math.isfinite(v) and v >= 0 for v in q['target'].values())
    assert abs(sum(q['target'].values())-1) < 1e-6
    return sum(v > 0 for v in q['target'].values())


def build(data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    data = read(data_ready)
    parts = {}
    for split, spec in data['stages']['stage3'].items():
        path = ROOT/spec['path']
        assert sha(path) == spec['sha256']
        original = rows(path)
        kept, quarantined = [], []
        for row in original:
            n = support_size(row)
            assert n is None or n >= 1
            (quarantined if n == 1 else kept).append(row)
        parts[split] = dict(kept=kept, quarantined=quarantined)
    train_groups = {r['_meta']['group_id'] for r in parts['train']['kept']}
    held_groups = {r['_meta']['group_id'] for k in parts['validation'].values() for r in k}
    reserved = set(read(Path(data_ready).parent/'reserved-public-groups.json'))
    assert not train_groups & (held_groups | reserved)
    train_counts = Counter(r['_meta']['label_type'] for r in parts['train']['kept'])
    mc_phases = Counter(r['_meta']['phase'] for r in parts['train']['kept']
                        if r['_meta']['label_type'] == 'paired_future_masked')
    assert set(mc_phases) == {'closed_discard', 'open_discard', 'call_window', 'kan_window'}
    n = sum(train_counts.values())
    source_mass = dict(expert_replay=.75, paired_future_masked=.25)
    phase_mass = {p: .25 for p in mc_phases}
    weights = {k: source_mass[k]*n/v for k, v in train_counts.items()}
    phase_weights = {p: .25*phase_mass[p]*n/v for p, v in mc_phases.items()}
    assert max([*weights.values(), *phase_weights.values()]) <= 8
    out.mkdir(parents=True, exist_ok=False)
    files, counts = {}, {}
    for split, part in parts.items():
        counts[split] = {}
        for kind, pool in part.items():
            path = out/(f'{split}.jsonl' if kind == 'kept' else f'{split}-single-support-quarantine.jsonl')
            with path.open('x') as f:
                for row in pool:
                    f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
            files[f'{split}_{kind}'] = dict(path=str(path), records=len(pool), sha256=sha(path))
            counts[split][kind] = dict(labels=dict(Counter(r['_meta']['label_type'] for r in pool)),
                phases=dict(Counter(r['_meta']['phase'] for r in pool)),
                support=dict(Counter(support_size(r) for r in pool if r['_meta']['label_type'] != 'expert_replay')))
    result = dict(status='stage3_data_prepared_pending_parent_and_fit_registration',
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], files=files, counts=counts,
        original_source_sha256={k: v['sha256'] for k, v in data['stages']['stage3'].items()},
        original_validation_kept_for_legacy_diagnostics=data['stages']['stage3']['validation'],
        loss_proposal=dict(label_source_mass=source_mass, MC_phase_mass=phase_mass,
                           label_source_weights=weights, MC_phase_weights=phase_weights),
        quarantined_by_support_structure_only=True, reward_sign_selection=False,
        near_uniform_or_small_gap_targets_retained=True, no_targets_relabelled=True,
        masked_loss_identity_preserved=True, train_validation_group_overlap=0,
        all_original_heldout_and_reserved_groups_excluded=True, copies_added=0,
        independent_world_confidence_not_reconstructed=True, parent_model_admitted=False, optimizer_updates=0,
        note='Only support-size1 masked targets are quarantined because their loss and gradient are identically zero. Full original validation is retained as a separate frozen diagnostic reference. The informative validation population has changed and must be labelled explicitly. Source/phase loss masses are proposals pending final stage3 fit registration; target gaps do not provide independent-world confidence.')
    with (out/'data-ready.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.data_ready, a.out)
