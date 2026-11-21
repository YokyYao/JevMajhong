"""Check real record/profile/recipe plumbing without training or model scores."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_parent import read, rows, sha
from tools.curriculum_attack_losses import canonical_question
from tools.curriculum_attack_profile import profile, target_ce
from tools.register_curriculum_attack import validate_recipe


def check(registered, data_ready, config, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    reg, data, c = read(registered), read(data_ready), read(config)
    for p, h in reg['inputs'].items():
        assert sha(p) == h, p
    for spec in data['files'].values():
        assert sha(ROOT/spec['path']) == spec['sha256']
    train, held = [rows(ROOT/data['files'][split]['path']) for split in ('train', 'validation')]
    final_step = validate_recipe(c, data, train)
    for changes in (dict(epochs=1), dict(optimizer_resume=True), dict(candidate_rule='minimum_CE')):
        try:
            validate_recipe(dict(c, **changes), data, train)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid recipe was admitted')
    sys.path.insert(0, reg['source'])
    from jevmahjong.training.base_lora import package_digest, target_cross_entropy
    assert package_digest(Path(reg['source'])/'jevmahjong') == reg['source_sha256']
    maximum = 0.
    predictions = []
    for row in held:
        q = row['questions']['action']
        keys = list(q['criteria'])
        raw = [i+1 for i in range(len(keys))]
        probabilities = [v/sum(raw) for v in raw]
        _, canonical = canonical_question(q)
        value, _ = target_ce(q, keys, probabilities)
        maximum = max(maximum, abs(value-target_cross_entropy(canonical, keys, probabilities)))
        predictions.append(dict(id=row['_meta']['id'], keys=keys, p=probabilities))
    assert maximum < 1e-12
    summary = profile(held, predictions, reg['training_config']['validation_phase_weights'], data['loss_mass'])
    assert len(summary['stratum_counts']) == 10 and sum(summary['stratum_counts'].values()) == len(held)
    # Import entry modules to catch dependency/syntax errors, without calling
    # their registrars, parent loaders or optimizers on an unfinished parent.
    import tools.train_curriculum_attack
    import tools.audit_curriculum_attack
    modules = ['tools/register_curriculum_attack.py', 'tools/train_curriculum_attack.py',
        'tools/audit_curriculum_attack.py', 'tools/curriculum_attack_profile.py',
        'configs/training/curriculum-stage3-attack-v1.json', 'tests/core/test_curriculum_attack_profile.py']
    result = dict(status='attack_entry_profile_recipe_precheck_passed_not_model_evaluation', node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], real_validation_records=len(held), profiled_strata=10,
        maximum_portable_vs_frozen_target_CE_difference=maximum, actual_exposures_if_registered=len(train)*c['epochs'],
        recipe_final_step=final_step, local_to_cumulative_final=c['stage_offset']+final_step,
        validation_and_game_steps=[0, 181, 362], data_ready_sha256=sha(data_ready),
        source_sha256={name: sha(ROOT/name) for name in modules},
        synthetic_probabilities_for_arithmetic_only=True, model_loaded=False, optimizer_updates=0,
        curve_points_written=0, parent_seal_or_registration_written=False,
        positive_complete_parent_load_and_actual_trainer_hook_path_still_pending=True,
        note='This probe uses fixed synthetic probability vectors on real menus only to check algebra and coverage. It is not a student score, win rate, reward observation or training/evaluation curve point. The actual calls final must pass its audit and seal before registration or tensor initialization.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('registered', 'data-ready', 'config', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    check(a.registered.resolve(), a.data_ready.resolve(), a.config.resolve(), a.out.resolve())
