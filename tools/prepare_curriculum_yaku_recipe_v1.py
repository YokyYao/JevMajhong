"""Prepare actual TRAIN-only weights/orders and six new-source loss bindings.

No parent, backbone, student optimizer, prediction curve or fit is created.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.curriculum_yaku_objective_v1 import MASS, METHOD, NEW_SOURCES, COMPONENTS, RECORDS, FINAL_STEP, weights, annotate, parts, weighted_loss, epoch_order, materialization_unchanged
from tools.curriculum_yaku_fit_v1 import random_proxy


def prepare(config):
    gpu_only(); c = read(config)
    if (c['status'] != 'prospective_data_and_objective_only_not_parent_or_fit_registered'
            or c['method'] != METHOD or c['seed'] != 2026112305 or c['epochs'] != 2
            or c['accumulation'] != 73 or c['mixing_block_records'] != 511
            or c['proposed_updates'] != FINAL_STEP or c['proposed_record_exposures'] != 182494
            or c['proposed_validation_and_game_steps'] != [0, 625, 1250, 1875, 2500]
            or not c['parent_decision_pending_actual_stage3A_results'] or c['automatic_fit']):
        raise ValueError('Exact prospective yaku2500 recipe, not a parent registration, required')
    data_path = ROOT/c['data_ready']; data = read(data_path)
    if (data['status'] != 'exact_replay_and_normalized_targeted_yaku_corpus_prepared_not_fit_admitted'
            or data['files']['train']['records'] != RECORDS or data['files']['validation']['records'] != 24516
            or not data['original_replay_objects_labels_and_partitions_unchanged']):
        raise ValueError('Use complete actual normalized data-v2 without changing replay')
    source_registration = ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json'
    reg = read(source_registration)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(2)
    from jevany.data import materialize
    public = ('tools/curriculum_yaku_objective_v1.py', 'tools/curriculum_yaku_fit_v1.py',
        'tools/prepare_curriculum_yaku_recipe_v1.py', 'configs/training/curriculum-yaku-course-v1.json',
        'cluster/curriculum-yaku-recipe-v1.slurm', 'docs/experiments/plans/curriculum-yaku-course-20261123.md')
    train_path = ROOT/data['files']['train']['path']
    files = {str(p.resolve()): sha(p) for p in (config, data_path, source_registration, train_path,
        ROOT/'tools/curriculum_progression_objective_v1.py', ROOT/'tools/curriculum_progression_fit_v1.py',
        ROOT/'tools/curriculum_course_order.py', *[ROOT/name for name in public])}
    if files[str(train_path.resolve())] != data['files']['train']['sha256']:
        raise ValueError('Actual TRAIN corpus changed')
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    for name in public:
        if name.endswith('.py'):
            compile((ROOT/name).read_text(), str(ROOT/name), 'exec')
    categories = []; probes = {}; counts = Counter()
    for line in train_path.open():
        row = json.loads(line); src = row['questions']['action']['src']
        categories.append(src); counts[src] += 1
        if src in NEW_SOURCES and src not in probes:
            probes[src] = row
    actual_weights = weights(dict(counts))
    if dict(counts) != data['coverage']['train']['sources'] or set(probes) != set(NEW_SOURCES):
        raise ValueError('All25 original source counts and six actual new-source probes required')
    checks = []
    for src, row in probes.items():
        original = materialize(row); material = annotate(copy.deepcopy(original), row)
        materialization_unchanged(original, material)
        question = material['questions'][0]
        logits = torch.linspace(-.9, .6, len(question['keys']), dtype=torch.float64, requires_grad=True)
        terms = parts(logits, question, 'cpu')
        if abs(float((terms['objective']-terms['standard_target_ce']-terms['extra_RI_group_ce']).detach())) > 1e-10:
            raise ValueError('Reused class2 objective differs from target CE plus binary term')
        observed = []
        loss = weighted_loss(logits, question, 'cpu', actual_weights, observed); loss.backward()
        if (not torch.isfinite(loss) or not torch.isfinite(logits.grad).all()
                or float(logits.grad.norm()) <= 0 or len(observed) != 1 or observed[0]['source'] != src
                or set(observed[0]) != {'source', *COMPONENTS}):
            raise ValueError('Actual new hard source lost finite gradient/source identity')
        checks.append(dict(source=src, actual_row_materialization_unchanged=True,
            reused_RI_class2_formula=True, finite_nonzero_synthetic_logit_gradient=True))
    rng = random.Random(c['seed']); expected = []; totals = Counter()
    for epoch in (1, 2):
        order, receipt = epoch_order(categories, rng)
        expected.append(dict(epoch=epoch, order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), **receipt))
        totals.update(categories[i] for i in order)
    proxy, created = random_proxy(random, c['seed'], categories, out/'actual-wrapper-orders-private', expected)
    wrapped = proxy.Random(c['seed'])
    for epoch in (1, 2):
        order = list(range(RECORDS)); wrapped.shuffle(order)
    if len(created) != 1 or wrapped.epochs != 2 or totals != Counter({s: 2*n for s, n in counts.items()}):
        raise ValueError('Actual RNG wrapper or complete two-epoch source exposure differs')
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Prospective recipe inputs changed')
    result = dict(status='actual_TRAIN_yaku25_weights_objective_and_two_epoch_orders_ready_not_parent_or_fit_registered',
        method=METHOD, train_records=RECORDS, validation_records_metadata_only=24516,
        source_counts=dict(counts), proposed_loss_mass=MASS, proposed_example_weights=actual_weights,
        maximum_example_weight=max(actual_weights.values()), expected_orders=expected,
        six_new_source_synthetic_logit_checks=checks, proposed_steps=FINAL_STEP,
        proposed_exposures=sum(totals.values()), actual_optimizer_steps=0,
        actual_student_predictions=0, validation_payload_opened=False,
        source_metadata_aliases_not_visible_inputs=True, old_loss_source_unchanged=True,
        parent_not_selected=True, no_fit_or_automatic_release=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json', result)
    write(out/'completion-binding-private.json', dict(files=files))
    summary = {k: result[k] for k in ('status', 'method', 'train_records', 'validation_records_metadata_only',
        'source_counts', 'proposed_loss_mass', 'maximum_example_weight', 'proposed_steps',
        'proposed_exposures', 'actual_optimizer_steps', 'actual_student_predictions',
        'validation_payload_opened', 'source_metadata_aliases_not_visible_inputs', 'old_loss_source_unchanged',
        'parent_not_selected', 'no_fit_or_automatic_release', 'node', 'job_id')}
    write(out/'summary-public.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); prepare(args.config.resolve())
