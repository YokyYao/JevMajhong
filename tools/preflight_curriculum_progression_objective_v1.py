"""TRAIN-only real materialization, loss and ordering preflight; zero updates."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_progression_objective_v1 import METHOD, MASS, COMPONENTS, annotate, parts, weights, epoch_order


def preflight(out):
    gpu_only()
    config_path = ROOT/'configs/diagnostics/curriculum-progression-objective-v1.json'; config = read(config_path)
    if config['method'] != METHOD or config['RI_binary_group_class_weight'] != 2. or config['training_and_parent_admission']:
        raise ValueError('Prospective fixed coefficient and no-fit contract required')
    audit_path = ROOT/config['data_audit']; audit = read(audit_path)
    if (audit['status'] != 'exact_progression_union_original_rows_encoder_masks_and_split_audit_passed_not_fit_admitted'
            or audit['checked_exact_source_row_objects'] != 111283):
        raise ValueError('Full independent actual union source/encoder audit required')
    paths = [Path(__file__), config_path, audit_path, audit_path.parent/'completion-binding-private.json',
        ROOT/'tools/curriculum_progression_objective_v1.py', ROOT/'tools/curriculum_losses.py',
        ROOT/'tools/curriculum_course_order.py', ROOT/'tools/curriculum_calls_post_reach_data_v1.py',
        ROOT/'tests/core/test_curriculum_progression_objective_v1.py',
        ROOT/'cluster/curriculum-progression-objective-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-progression-objective-20261123.md']
    files = dict(read(audit_path.parent/'completion-binding-private.json')['files'])
    for path in paths: files[str(path.resolve())] = sha(path)
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Audited source or active fit changed')
    reg = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevany.data import materialize
    import torch
    torch.set_num_threads(4)
    spec = audit['files']['train']; source_path = ROOT/spec['path']
    if sha(source_path) != spec['sha256']: raise ValueError('Audited TRAIN union changed')
    files[str(source_path)] = spec['sha256']; categories = []; counts = Counter(); probe = {}; RI_counts = Counter()
    for line in source_path.open():
        row = json.loads(line); q = annotate(materialize(row), row)['questions'][0]; src = q['src']
        categories.append(src); counts[src] += 1
        raw = row['questions']['action']
        if set(v['type'] for v in raw['criteria'].values()) == {'reach', 'dahai'} and raw.get('target') is None:
            RI_counts[src] += 1
        if src not in probe:
            logits = torch.linspace(-.5, .5, len(q['keys']), dtype=torch.float64, requires_grad=True)
            result = parts(logits, q, 'cpu'); result['objective'].backward()
            if any(not torch.isfinite(v) for v in result.values()) or not torch.isfinite(logits.grad).all():
                raise ValueError('Actual materialized source has nonfinite objective/gradients')
            if q.get('target') is not None:
                mask = torch.tensor(q['target']) > 0
                if (logits.grad[~mask] != 0).any(): raise ValueError('Unobserved MC alternative has nonzero gradient')
            probe[src] = {key: float(value.detach()) for key, value in result.items()}
    if len(categories) != 87595 or set(counts) != set(MASS) or len(probe) != 19:
        raise ValueError('All original19 TRAIN strata required')
    scalar_weights = weights(dict(counts)); out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(config=config, files=files, active_fit_files=active,
        TRAIN_only_no_validation_label_materialization=True))
    rng = random.Random(config['training_order_seed']); orders = []; exposure = Counter()
    with (out/'epoch-orders-private.jsonl').open('x') as stream:
        for epoch in (1, 2):
            order, receipt = epoch_order(categories, rng)
            exposure.update(categories[i] for i in order)
            actual_source = Counter({s: sum(part.get(s, 0) for part in receipt['source_counts_by_update']) for s in MASS})
            if actual_source != counts: raise ValueError('Ordering changed original source exposure')
            actual_mass = {s: actual_source[s]*scalar_weights[s]/len(order) for s in MASS}
            if any(abs(actual_mass[s]-MASS[s]) > 1e-12 for s in MASS): raise ValueError('Actual epoch scalar source mass differs')
            stream.write(json.dumps(dict(epoch=epoch, order=order, receipt=receipt))+'\n')
            orders.append(dict(epoch=epoch, **{k: v for k, v in receipt.items() if k != 'source_counts_by_update'},
                order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest()))
    if exposure != Counter({s: 2*n for s, n in counts.items()}): raise ValueError('Two-epoch total exposure differs')
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Bound source changed during preflight')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    write(out/'real-source-gradient-probes-private.json', probe)
    result = dict(status='TRAIN_progression19_groupRI_loss_and_two_epoch_orders_preflight_passed_not_fit_admitted',
        method=METHOD, annotated_actual_TRAIN_rows=87595, source_counts=dict(counts),
        proposed_scalar_loss_mass=MASS, proposed_example_weights=scalar_weights,
        maximum_scalar_example_weight=max(scalar_weights.values()), RI_group_scope_TRAIN_counts=dict(RI_counts),
        actual_real_source_loss_gradient_probes=19, expected_orders=orders,
        proposed_updates=2400, proposed_record_exposures=sum(exposure.values()),
        all19_sources_per_mixing_block_not_each_optimizer_update=True,
        class_lambda1_exact_standard_CE_not_new_objective=True, class_lambda2_additional_binary_term=True,
        validation_labels_materialized_or_used_for_arithmetic_recipe=False,
        policy_or_backbone_loaded=False, optimizer_steps=0, new_games=0,
        active_fit_dependencies_unchanged=len(active), original_parent_review_and_fit_registration_pending=True,
        raw_orders_and_per_example_probes_private=True, job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    preflight(p.parse_args().out.resolve())
