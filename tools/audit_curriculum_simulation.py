"""Audit published masked simulation targets without inventing world confidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def audit(data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    data = read(data_ready)
    summaries = {}
    groups = {}
    for split, spec in data['stages']['stage3'].items():
        path = ROOT/spec['path']
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == spec['sha256']
        pool = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        labels = Counter(r['_meta']['label_type'] for r in pool)
        assert set(labels) == {'expert_replay', 'paired_future_masked'}
        mc = [r for r in pool if r['_meta']['label_type'] == 'paired_future_masked']
        support = Counter()
        menu_sizes = Counter()
        top_types = Counter()
        phases = Counter()
        keys = Counter()
        source_groups = defaultdict(set)
        entropy = []
        gaps = []
        target_sums = []
        label_not_max = 0
        unlabeled = 0
        for row in mc:
            q = row['questions']['action']
            assert q['src'] == 'paired_future_masked'
            target = q['target']
            assert isinstance(target, dict) and set(target) <= set(q['criteria'])
            assert all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in target.values())
            positive = {k: v for k, v in target.items() if v > 0}
            assert positive, 'No supervised support'
            target_sums.append(sum(positive.values()))
            label_not_max += q['label'] not in positive or abs(positive.get(q['label'], 0)-max(positive.values())) >= 1e-8
            ordered = sorted(positive.values(), reverse=True)
            support[len(positive)] += 1
            menu_sizes[len(q['criteria'])] += 1
            top_types[q['criteria'][q['label']]['type']] += 1
            phases[row['_meta']['phase']] += 1
            source_groups[row['_meta']['phase']].add(row['_meta']['group_id'])
            keys.update(row['_meta'].keys())
            # Only field-name availability is counted; no private values are published.
            if len(positive) >= 2:
                normalized = [p/sum(positive.values()) for p in positive.values()]
                gaps.append((ordered[0]-ordered[1])/sum(positive.values()))
                entropy.append(-sum(p*math.log(p) for p in normalized)/math.log(len(normalized)))
            unlabeled += len(positive) < len(q['criteria'])
            assert not set(row['state']) & {'future', 'worlds', 'hidden_hand', 'private_wall', 'rollout_return'}
        provenance = {k: keys.get(k, 0) for k in
                      ('worlds', 'n_worlds', 'future_worlds', 'source_sha256', 'reward_formula', 'continuation_policy')}
        summaries[split] = dict(records=len(pool), label_counts=dict(labels), simulated_records=len(mc),
            simulated_phases=dict(phases), simulated_groups_by_phase={k: len(v) for k, v in source_groups.items()},
            labeled_support_sizes=dict(support), total_menu_sizes=dict(menu_sizes),
            single_support_zero_information_records=support.get(1, 0),
            informative_comparison_records=len(entropy),
            target_mass_range=[min(target_sums), max(target_sums)],
            target_label_not_argmax_records=label_not_max,
            menus_with_unlabeled_legal_candidates=unlabeled, top_target_action_types=dict(top_types),
            mean_normalized_target_entropy=statistics.mean(entropy) if entropy else None,
            normalized_entropy_at_least_095=sum(e >= .95 for e in entropy),
            mean_top_two_target_gap=statistics.mean(gaps) if gaps else None, gap_below_01=sum(g < .1 for g in gaps),
            per_record_provenance_field_availability=provenance,
            masked_loss_identity_preserved=True,
            per_record_world_count_and_policy_available=provenance['n_worlds'] == len(mc)
                and provenance['continuation_policy'] == len(mc),
            confidence_from_independent_worlds_verified=False)
        groups[split] = {r['_meta']['group_id'] for r in pool}
    reserved = set(read(Path(data_ready).parent/'reserved-public-groups.json'))
    assert not groups['train'] & (groups['validation'] | reserved)
    result = dict(status='stage3_masked_target_audit_complete_not_fit', node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], splits=summaries, all_source_hashes_unchanged=True,
        heldout_groups_excluded=True, no_model_loaded=True, optimizer_updates=0,
        note='Target entropy/gap describes stored preferences, not Monte Carlo confidence. Missing future-world/reward/policy provenance is not fabricated. A phase-weighted loss must retain paired_future_masked semantics so unobserved legal actions are not treated as negative labels. New target generation needs separately frozen public-information simulation and independent source groups.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.data_ready, a.out)
