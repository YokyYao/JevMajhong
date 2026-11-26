"""Numerical validation integration only; synthetic outputs never model curves."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_progression_objective_v1 import MASS
from tools.curriculum_progression_fit_v1 import profile, random_proxy


def preflight(out):
    gpu_only(); out.mkdir(parents=True, exist_ok=False)
    objective = ROOT/'runs/curriculum-progression-objective-v1/preflight-v1'
    prior = read(objective/'summary-private.json')
    if (prior['status'] != 'TRAIN_progression19_groupRI_loss_and_two_epoch_orders_preflight_passed_not_fit_admitted'
            or prior['optimizer_steps'] != 0 or prior['proposed_scalar_loss_mass'] != MASS):
        raise ValueError('Actual objective/materialization/order preflight required')
    files = dict(read(objective/'completion-binding-private.json')['files'])
    for p in (Path(__file__), objective/'summary-private.json', objective/'completion-binding-private.json',
              ROOT/'tools/curriculum_progression_fit_v1.py', ROOT/'tests/core/test_curriculum_progression_fit_v1.py',
              ROOT/'cluster/curriculum-progression-fit-preflight-v1.slurm',
              ROOT/'docs/experiments/plans/curriculum-progression-fit-preflight-20261123.md'):
        files[str(p.resolve())] = sha(p)
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Bound source changed')
    audit = read(ROOT/'runs/curriculum-progression-union-v1/audit-v1/summary-private.json')
    def rows(split):
        spec = audit['files'][split]; path = ROOT/spec['path']
        if sha(path) != spec['sha256']: raise ValueError('Actual audited union changed')
        files[str(path)] = spec['sha256']; return [json.loads(line) for line in path.open()]
    train = rows('train'); categories = [r['questions']['action']['src'] for r in train]
    proxy, created = random_proxy(random, 2026112304, categories, out/'actual-wrapper-order-receipts-private', prior['expected_orders'])
    rng = proxy.Random(2026112304)
    for _ in (1, 2):
        order = list(range(87595)); rng.shuffle(order)
    if len(created) != 1 or rng.epochs != 2: raise ValueError('Actual wrapper RNG did not consume exactly two registered permutations')
    del train, categories, order
    registration = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json')
    sys.path.insert(0, registration['source']); sys.path.insert(0, registration['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.training.base_lora import target_cross_entropy
    validation = rows('validation'); predictions = []; analytic = defaultdict(list); analytic_ce = defaultdict(list)
    for row in validation:
        q = row['questions']['action']; keys = list(q['criteria']); n = len(keys); kinds = {v['type'] for v in q['criteria'].values()}
        predictions.append(dict(id=row['_meta']['id'], keys=keys, p=[1/n]*n))
        if q.get('target') is not None:
            value = math.log(sum(v > 0 for v in q['target'].values())); ce = value
        elif kinds == {'reach', 'dahai'}:
            yes = q['criteria'][q['label']]['type'] == 'reach'
            value = 2*math.log(n) if yes else 2*math.log(n/(n-1))+math.log(n-1)
            ce = math.log(n)
        else: value = ce = math.log(n)
        analytic[q['src']].append(value); analytic_ce[q['src']].append(ce)
    actual = profile(validation, predictions, {}, target_cross_entropy)
    expected = sum(MASS[s]*sum(v)/len(v) for s, v in analytic.items())
    expected_ce = sum(MASS[s]*sum(v)/len(v) for s, v in analytic_ce.items())
    error = abs(actual['weighted_class_objective']-expected); ce_error = abs(actual['weighted_target_ce']-expected_ce)
    if (error > 1e-10 or ce_error > 1e-10 or sum(actual['stratum_counts'].values()) != 23688
            or actual['hard_target_records'] != 22504 or len(actual['MC_selected_mean_target_mass']) != 4):
        raise ValueError('Full actual validation integration differs from independent analytic uniform losses')
    write(out/'synthetic-uniform-profile-private.json', dict(profile=actual,
        synthetic_probabilities_not_model_predictions=True, forbidden_for_training_or_model_game_curves=True))
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Bound source changed during integration')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='progression_actual_order_wrapper_and_synthetic_profile_integration_passed_not_fit_admitted',
        actual_TRAIN_order_records=87595, wrapper_epochs=2, actual_optimizer_steps=0,
        synthetic_uniform_validation_records=23688, synthetic_uniform_hard_records=22504, synthetic_uniform_MC_records=1184,
        independent_analytic_weighted_loss_error=error, independent_analytic_weighted_target_CE_error=ce_error,
        synthetic_profile_not_student_quality_or_curve=True, no_validation_outcome_or_label_based_recipe_selection=True,
        no_backbone_or_model_loaded=True, no_new_games=True, parent_admission_and_real_fit_pending=True,
        active_fit_dependencies_unchanged=len(active), raw_outputs_private=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    preflight(p.parse_args().out.resolve())
