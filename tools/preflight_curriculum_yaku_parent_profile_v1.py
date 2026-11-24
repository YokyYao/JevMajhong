"""Full actual3B validation-menu contract with target-independent fake probabilities.

This measures software invariants, not a student or teacher's performance.
It never writes the active model's validation curves or admits a fit.
"""
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha


def main():
    allocation = resource(); started = time.monotonic()
    reg_path = ROOT/'runs/curriculum-progression-parent-retention-v1/fit-plan/registered.json'
    reg = read(reg_path)
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany.data import materialize
    from jevmahjong.training.base_lora import target_cross_entropy
    from tools import curriculum_yaku_parent_fit_v1 as new
    from tools import curriculum_yaku_fit_v1 as previous
    from tools.curriculum_yaku_parent_loss_v1 import COMPONENTS, MASS, METHOD
    proof = ROOT/'runs/curriculum-yaku-parent-loss-v1/preflight-v1'
    earlier = read(proof/'summary-private.json'); bound = read(proof/'completion-binding-private.json')['files']
    if (earlier['actual_TRAIN_records'] != 91247 or earlier['maximum_actual_objective_or_CE_difference'] != 0
            or earlier['maximum_representative_gradient_difference'] != 0
            or any(sha(p) != h for p, h in bound.items())):
        raise ValueError('Exact retained25-source TRAIN objective proof required')
    ready_path = ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json'
    ready = read(ready_path); path = ROOT/ready['files']['validation']['path']
    if sha(path) != ready['files']['validation']['sha256']: raise ValueError('Prepared validation changed')
    out = ROOT/'runs/curriculum-yaku-parent-loss-v1/validation-profile-preflight-v1'
    out.mkdir(exist_ok=False)
    files = {str(p.resolve()): sha(p) for p in (Path(__file__),
        ROOT/'tools/curriculum_yaku_parent_fit_v1.py', ROOT/'tools/curriculum_yaku_fit_v1.py',
        ROOT/'tools/curriculum_yaku_parent_loss_v1.py', ROOT/'tools/curriculum_progression_parent_fit_v1.py',
        ROOT/'docs/experiments/plans/curriculum-yaku-parent-profile-20261124.md',
        proof/'summary-private.json', proof/'completion-binding-private.json', reg_path, ready_path, path)}
    files.update(bound)
    for name in ('tools/curriculum_progression_fit_v1.py',
                 'tools/curriculum_complete_legal_class_policy_v1.py',
                 'tools/curriculum_riichi_class_policy_v1.py',
                 'tools/curriculum_riichi_policy_profile_v1.py'):
        files[str(ROOT/name)] = sha(ROOT/name)
    write(out/'registered-private.json', dict(files=files, allocation=allocation,
        actual_validation_menus=24516, target_independent_synthetic_probabilities=True,
        student_model_loaded=False, student_updates=0, model_curve_files_modified=False))
    try:
        records = [json.loads(line) for line in path.open() if line.strip()]
        predictions = []
        for row in records:
            keys = materialize(row)['questions'][0]['keys']
            mass = [math.exp(.7*i/max(1, len(keys)-1)) for i in range(len(keys))]
            predictions.append(dict(id=row['_meta']['id'], keys=keys, p=[v/sum(mass) for v in mass]))
        if len(records) != 24516: raise ValueError('Actual complete24516 menus required')
        actual = new.profile(records, predictions, {}, target_cross_entropy)
        original = previous.profile(records, predictions, {}, target_cross_entropy)
        changed = {'weighted_class_objective', 'extra_RI_class_loss', 'signed_objective_adjustment',
            'new_route_objective', 'source_loss_components', 'source_objective', 'replay_profile', 'loss_method'}
        for key, value in original.items():
            if key in changed: continue
            if isinstance(value, float):
                if abs(actual[key]-value) > 1e-10: raise ValueError('Canonical metric changed: '+key)
            elif actual[key] != value: raise ValueError('Serving/label/classification metric changed: '+key)
        means = actual['source_loss_components']
        if (actual['loss_method'] != METHOD or set(means) != set(MASS)
                or any(set(v) != set(COMPONENTS) for v in means.values())
                or sum(actual['stratum_counts'].values()) != 24516
                or set(actual['replay_profile']['source_loss_components']) != set(new.REPLAY_MASS)):
            raise ValueError('Complete actual25-source profile and inherited19-source profile required')
        if abs(actual['weighted_class_objective']-actual['weighted_target_ce']-actual['signed_objective_adjustment']) > 1e-10:
            raise ValueError('Signed profile accounting differs')
        if 'extra_RI_class_loss' in actual or 'extra_RI_class_loss' in actual['replay_profile']:
            raise ValueError('Retained loss must not acquire the old nonnegative-extra interpretation')
        if any(sha(p) != h for p, h in files.items()): raise ValueError('A bound validation input changed')
        write(out/'synthetic-profile-private.json', actual)
        summary = dict(status='actual24516_yaku_parent_profile_contract_passed_not_model_evaluation',
            method=METHOD, validation_menus=24516, inherited_replay_menus=23688,
            six_new_route_menus=828, source_count=25,
            canonical_CE_and_every_classification_metric_unchanged=True,
            retained_nine_component_profile_and_signed_accounting_verified=True,
            target_independent_synthetic_probabilities=True, student_model_loaded=False,
            model_forwards=0, student_updates=0, new_games=0, active_model_curves_modified=False,
            parent_qualification_registration_trainer_and_auditor_still_required=True,
            seconds=time.monotonic()-started)
        write(out/'summary-private.json', summary)
        for p in (out/'registered-private.json', out/'synthetic-profile-private.json', out/'summary-private.json'):
            files[str(p)] = sha(p)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(error=repr(error)))
        raise


if __name__ == '__main__': main()
