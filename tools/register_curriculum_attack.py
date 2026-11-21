"""Register an attack/risk course only after sealing the qualified calls final."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_parent import read, rows, sha, require_child_input_contract, require_replay
from tools.curriculum_attack_losses import PREFIX, mixture, stratum, canonical_question
from tools.curriculum_calls_parent import METHOD, verify_calls_parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def validate_recipe(c, data, train):
    flags = ('preserve_calls_relative_replay_loss_mass', 'equal_mass_on_four_MC_phases', 'all10_strata_each_update',
        'same_adapter_and_pointer_head', 'new_optimizer', 'legal_wins_executed_by_existing_guard')
    if (c['stage'] != 'stage3-attack-v1' or c['batch'] != 1 or c.get('optimizer_resume')
        or any(c.get(k) is not True for k in flags)):
        raise ValueError('Explicit same-model masked attack recipe required')
    for row in train:
        key, _ = canonical_question(row['questions']['action'])
        if key != stratum(row) or row['questions']['action']['src'] != PREFIX+key:
            raise ValueError('Loss-only source tag differs from actual row source/phase')
    recipe = mixture(train, c['expert_loss_mass'])
    if (abs(c['expert_loss_mass']+c['MC_loss_mass']-1) > 1e-12
        or recipe['loss_mass'] != data['loss_mass'] or recipe['example_weights'] != data['example_weights']
        or c['accumulation'] != data['count_only_accumulation_proposal']['accumulation']
        or data['count_only_accumulation_proposal']['final_fraction'] < c['minimum_tail_fraction']):
        raise ValueError('Prepared source mixture or count-only ordering changed')
    steps = c['epochs']*math.ceil(len(train)/c['accumulation'])
    if (steps != 362 or c['candidate_rule'] != 'registered_final_full_two_epochs_362_updates'
        or c['validation_every_steps'] != 181 or c['game_evaluation_every_steps'] != 181):
        raise ValueError('Attack exposure, evaluation cadence or final-only selection changed')
    return steps


def register(config, parent_seal):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    c, sealed = read(config), read(parent_seal)
    if (c['stage'] != 'stage3-attack-v1' or c['batch'] != 1 or c.get('optimizer_resume')
        or sealed.get('method') != METHOD or not sealed['stage_quality']['passed']
        or c['stage_offset'] != sealed['cumulative_step']):
        raise ValueError('Explicit qualified calls-final continuation required')
    parent = read(sealed['parent_registration'])
    pc = parent['curriculum']
    ready = Path(sealed['child_data_ready'])
    data = read(ready)
    if data['status'] != 'stage3_attack_data_prepared_pending_parent_and_fit_registration':
        raise ValueError('Wrong prepared attack dataset')
    require_child_input_contract(parent, data)
    for spec in data['files'].values():
        if sha(ROOT/spec['path']) != spec['sha256']:
            raise ValueError('Attack data changed')
    train = rows(ROOT/data['files']['train']['path'])
    held = rows(ROOT/data['files']['validation']['path'])
    steps = validate_recipe(c, data, train)
    args_parent = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    require_replay(rows(args_parent['data']), train)
    require_replay(rows(args_parent['eval_data']), held)
    for key in ('base', 'base_revision', 'dropout', 'fresh_lora_rank', 'head_dim', 'curve_seed_start', 'curve_seed_groups',
                'stage_end_seed_groups', 'opponent_profiles', 'reference_checkpoint', 'reference_identity', 'reference_evaluation_cache', 'weight_decay'):
        c[key] = pc[key]
    c.update(data_ready=str(ready.resolve()), data_root=str(Path(data['reserved_groups']).parent), riichi_input_schema=pc.get('riichi_input_schema'),
        parent_seal=str(parent_seal.resolve()), parent_identity=sealed['model']['identity_sha256'],
        parent_evaluation_cache=str((ROOT/pc['out']/'evaluations'/f"step-{sealed['selected_step']:06d}").resolve()))
    plan = ROOT/c['plan_root']
    if plan.exists() or (ROOT/c['out']).exists():
        raise ValueError('Never overwrite or silently restart an attack fit')
    for path, expected in sealed['bound_files'].items():
        if sha(path) != expected:
            raise ValueError('Calls seal input changed before registration: '+path)
    plan.mkdir(parents=True)
    args = copy.deepcopy(args_parent)
    args.update(data=str((ROOT/data['files']['train']['path']).resolve()), eval_data=str((ROOT/data['files']['validation']['path']).resolve()),
        out=str(ROOT/c['out']/'training-run'), init_from=sealed['model']['directory'], epochs=c['epochs'], max_steps=steps,
        lr=c['learning_rate'], head_lr=c['head_learning_rate'], batch=1, accum=c['accumulation'], seed=c['seed'],
        checkpoint_every_steps=c['validation_every_steps'], eval_every_steps=c['validation_every_steps'], early_stop_patience=0)
    args.pop('resume', None)
    trainer = plan/'trainer.toml'
    trainer.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k, v in args.items()))
    receipt = plan/'data-receipt.json'
    write(receipt, dict(status='balanced_raw_data_ready', requires_masked_soft_target_loss=True, files=data['files'],
        exact_parent_expert_replay=True, confidence_filtering=False, win_choice_supervision=False))
    cfg = copy.deepcopy(parent['training_config'])
    if cfg.get('label_source_gradient_mass') is not None or cfg.get('paired_future_phase_mass') is not None:
        raise ValueError('Do not apply a second hidden source-weighting adapter')
    cfg.update(model_id=c['run_id'], out=c['out'], initialization='continued', trainer_config=str(trainer), trainer_config_sha256=sha(trainer),
        data_receipt=str(receipt), data_receipt_sha256=sha(receipt), registered_final_step=steps,
        parent_initialization=dict(method=METHOD, manifest=str(parent_seal.resolve()), sha256=sha(parent_seal),
            identity_sha256=c['parent_identity'], selected_step=sealed['selected_step'], source=parent['source'],
            source_sha256=parent['source_sha256'], run=str((ROOT/pc['out']).resolve()), child_data_ready=str(ready.resolve())))
    training_config = plan/'training-config.json'
    write(training_config, cfg)
    verify_calls_parent(cfg)
    paths = [config, parent_seal, ready, trainer, receipt, training_config, Path(__file__)]
    paths += [ROOT/'tools'/name for name in ('train_curriculum_attack.py', 'audit_curriculum_attack.py', 'curriculum_attack_profile.py',
        'curriculum_attack_losses.py', 'curriculum_calls_parent.py', 'curriculum_course_order.py', 'curriculum_losses.py',
        'build_curriculum_stage2.py', 'evaluate_curriculum.py',
        'plot_curriculum.py', 'plot_curriculum_fixed_grid.py')]
    paths += [ROOT/'tests/core'/name for name in ('test_curriculum_attack_course.py', 'test_curriculum_calls_parent.py', 'test_curriculum_attack_profile.py')]
    paths += [ROOT/'cluster/curriculum-stage3.slurm', ROOT/'cluster/curriculum-stage3-audit.slurm']
    bound = dict(sealed['bound_files'])
    bound.update({str(p.resolve()): sha(p) for p in paths})
    if data.get('source_world_audit'):
        spec = data['source_world_audit']
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Declared conditional-world audit changed')
        bound[str(Path(spec['path']).resolve())] = spec['sha256']
    for cache in (c['reference_evaluation_cache'], c['parent_evaluation_cache']):
        for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'):
            path = ROOT/cache/name
            bound[str(path.resolve())] = sha(path)
    result = dict(status='stage3_registered_not_started', curriculum=c, training_config=cfg, source=parent['source'],
        source_sha256=parent['source_sha256'], inputs=bound, registered_final_step=steps, train_records=len(train),
        course_loss_mass=data['loss_mass'], course_loss_weights=data['example_weights'], parent_model=sealed['model'],
        new_optimizer=True, optimizer_updates=0)
    write(plan/'registered.json', result)
    print(json.dumps(dict(status=result['status'], final_step=steps, actual_exposures=len(train)*c['epochs'], parent_identity=c['parent_identity'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--parent-seal', type=Path, required=True)
    a = p.parse_args()
    register(a.config.resolve(), a.parent_seal.resolve())
