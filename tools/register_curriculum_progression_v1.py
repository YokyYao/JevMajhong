"""Register exact full-budget progression learning after actual parent sealing."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_progression_parent_v1 import METHOD as PARENT_METHOD, STATUS, verify_parent
from tools.curriculum_progression_objective_v1 import MASS, METHOD, weights

PUBLIC_FILES = (
    'tools/curriculum_progression_parent_v1.py', 'tools/register_curriculum_progression_v1.py',
    'tools/train_curriculum_progression_v1.py', 'tools/evaluate_curriculum_progression_v1.py',
    'tools/audit_curriculum_progression_v1.py', 'tools/curriculum_progression_runtime_v1.py',
    'tools/prepare_curriculum_progression_v1.py', 'tools/finish_curriculum_progression_audit_v1.py',
    'tests/core/test_curriculum_progression_runtime_v1.py',
    'configs/training/curriculum-progression-v1.json',
    'docs/experiments/plans/curriculum-progression-fit-20261123.md',
    'docs/experiments/plans/curriculum-progression-launch-20261123.md',
    'cluster/curriculum-progression-v1.slurm',
)


def register(config, seal_path):
    gpu_only(); c = read(config); sealed = read(seal_path)
    if (c['status'] != 'planned_not_registered_or_fit_admitted' or c['final_optimizer_steps'] != 2400
            or c['epochs'] != 2 or c['accumulation'] != 73 or c['mixing_block_records'] != 511
            or c['loss_method'] != METHOD or c['seed'] != 2026112304
            or c['validation_and_game_steps'] != [0, 600, 1200, 1800, 2400]
            or c['learning_rate'] != 3e-6 or c['head_learning_rate'] != 1.5e-5
            or c['expected_record_exposures'] != 175190 or sealed['status'] != STATUS or sealed['method'] != PARENT_METHOD):
        raise ValueError('Exact frozen progression2400 recipe and complete sealed parent required')
    parent = read(sealed['parent_registration']); pc = parent['curriculum']
    data_path = Path(sealed['child_data_ready']); data = read(data_path)
    prospective = read(ROOT/c['objective_preflight']); adapter = read(ROOT/c['fit_adapter_preflight'])
    if (prospective['status'] != 'TRAIN_progression19_groupRI_loss_and_two_epoch_orders_preflight_passed_not_fit_admitted'
            or adapter['status'] != 'progression_actual_order_wrapper_and_synthetic_profile_integration_passed_not_fit_admitted'
            or prospective['optimizer_steps'] != 0 or adapter['actual_optimizer_steps'] != 0
            or prospective['proposed_scalar_loss_mass'] != MASS):
        raise ValueError('Actual zero-update objective and adapter checks required')
    for p, h in sealed['bound_files'].items():
        if sha(p) != h: raise ValueError('Sealed prerequisite changed: '+p)
    counts = Counter()
    for line in (ROOT/data['files']['train']['path']).open(): counts[json.loads(line)['questions']['action']['src']] += 1
    source_weights = weights(dict(counts))
    if (sum(counts.values()) != 87595 or data['files']['validation']['records'] != 23688
            or dict(counts) != prospective['source_counts'] or source_weights != prospective['proposed_example_weights']):
        raise ValueError('All original actual source counts/weights required')
    for name in ('base', 'base_revision', 'dropout', 'fresh_lora_rank', 'head_dim', 'curve_seed_start',
                 'curve_seed_groups', 'stage_end_seed_groups', 'opponent_profiles', 'reference_checkpoint',
                 'reference_identity', 'reference_evaluation_cache', 'weight_decay'):
        c[name] = pc[name]
    c.update(data_ready=str(data_path), parent_seal=str(seal_path.resolve()), riichi_input_schema=None,
        parent_identity=sealed['model']['identity_sha256'], parent_policy_identity=sealed['policy_identity'],
        parent_evaluation_cache=sealed['parent_evaluation_cache'],
        original_guarded_evaluation_cache=sealed['original_guarded_evaluation_cache'], loss_mass=MASS)
    plan = ROOT/c['plan_root']
    if plan.exists() or (ROOT/c['out']).exists(): raise ValueError('Never overwrite a registration or fit')
    plan.mkdir(parents=True)
    args = copy.deepcopy(tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())); args.pop('resume', None)
    args.update(data=str((ROOT/data['files']['train']['path']).resolve()), eval_data=str((ROOT/data['files']['validation']['path']).resolve()),
        out=str((ROOT/c['out']/'training-run').resolve()), init_from=sealed['model']['directory'],
        epochs=2, max_steps=2400, lr=3e-6, head_lr=1.5e-5, batch=1, accum=73, seed=c['seed'],
        checkpoint_every_steps=600, eval_every_steps=600, early_stop_patience=0)
    trainer = plan/'trainer.toml'
    with trainer.open('x') as stream: stream.write(''.join(f'{k} = {json.dumps(v)}\n' for k, v in args.items()))
    actual_data = {split: dict(data['files'][split], path=str((ROOT/data['files'][split]['path']).resolve()))
        for split in ('train', 'validation')}
    receipt = plan/'data-receipt.json'
    write(receipt, dict(status='balanced_raw_data_ready', requires_masked_soft_target_loss=True, files=actual_data,
        source_row_objects_unchanged=True, full_source_group_exclusions=True, win_choice_supervision=False))
    cfg = copy.deepcopy(parent['training_config'])
    for key in ('control_registration', 'label_source_gradient_mass', 'paired_future_phase_mass'): cfg.pop(key, None)
    cfg.update(model_id=c['run_id'], out=c['out'], initialization='continued', loss_method=METHOD,
        trainer_config=str(trainer), trainer_config_sha256=sha(trainer), data_receipt=str(receipt), data_receipt_sha256=sha(receipt),
        registered_final_step=2400, evaluation_candidate_rule='final_registered_budget_v1', riichi_input_schema=None,
        validation_phase_weights=MASS,
        parent_initialization=dict(method=PARENT_METHOD, manifest=str(seal_path.resolve()), sha256=sha(seal_path),
            identity_sha256=c['parent_identity'], policy_identity_sha256=c['parent_policy_identity'], selected_step=388,
            source=parent['source'], source_sha256=parent['source_sha256'], run=str((ROOT/pc['out']).resolve()),
            child_data_ready=str(data_path)))
    cfg_path = plan/'training-config.json'; write(cfg_path, cfg); verify_parent(cfg)
    files = dict(sealed['bound_files'])
    paths = [config, seal_path, trainer, receipt, cfg_path, ROOT/c['objective_preflight'], ROOT/c['fit_adapter_preflight']]
    paths.extend(ROOT/name for name in PUBLIC_FILES)
    for cache in (c['parent_evaluation_cache'], c['original_guarded_evaluation_cache'], c['reference_evaluation_cache']):
        paths.extend(ROOT/cache/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'))
    for path in paths: files[str(Path(path).resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Registered input changed')
    reg = dict(status='progression2400_registered_not_started', curriculum=c, training_config=cfg,
        source=parent['source'], source_sha256=parent['source_sha256'], inputs=files, registered_final_step=2400,
        train_records=87595, validation_records=23688, expected_record_exposures=175190,
        course_loss_weights=source_weights, course_loss_mass=MASS, expected_orders=prospective['expected_orders'],
        parent_model=sealed['model'], new_optimizer=True, optimizer_updates=0,
        actual_parent_execution_transfer_preserved=True, old_guarded_grade_not_overridden=True)
    write(plan/'registered.json', reg)
    print(json.dumps(dict(status=reg['status'], updates=2400, exposures=175190)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--parent-seal', type=Path, required=True); a = p.parse_args(); register(a.config.resolve(), a.parent_seal.resolve())
