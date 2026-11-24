"""Register the unchanged yaku2500 course after a genuine replay2472 seal."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_yaku_balanced_parent_v1 import METHOD as PARENT_METHOD, STATUS, verify_parent, gpu_only
from tools.curriculum_yaku_parent_loss_v1 import MASS, METHOD, weights
from tools.curriculum_yaku_parent_runtime_v1 import require_registration

PUBLIC_FILES = (
    'tools/curriculum_checkpoint_evaluation_fence_v1.py',
    'tests/core/test_curriculum_checkpoint_evaluation_fence_v1.py',
    'docs/experiments/plans/curriculum-save-order-recovery-20261123.md',
    'tools/curriculum_yaku_balanced_parent_v1.py', 'tools/curriculum_yaku_parent_runtime_v1.py',
    'tools/curriculum_yaku_parent_loss_v1.py', 'tools/curriculum_yaku_parent_fit_v1.py',
    'tools/register_curriculum_yaku_balanced_v1.py', 'tools/launch_curriculum_yaku_balanced_v1.py',
    'tools/train_curriculum_yaku_balanced_v1.py', 'tools/evaluate_curriculum_yaku_parent_v1.py',
    'tools/audit_curriculum_yaku_balanced_v1.py',
    'tests/core/test_curriculum_yaku_balanced_parent_v1.py',
    'tests/core/test_curriculum_yaku_balanced_course_v1.py',
    'configs/training/curriculum-yaku-balanced-fit-v1.json',
    'cluster/curriculum-yaku-balanced-v1.slurm',
    'docs/experiments/plans/curriculum-yaku-balanced-course-20261125.md',
)


def register(config, seal_path):
    gpu_only(); c = read(config); sealed = read(seal_path)
    if (c['status'] != 'planned_not_registered_or_fit_admitted' or c['final_optimizer_steps'] != 2500
            or c['epochs'] != 2 or c['batch'] != 1 or c['accumulation'] != 73
            or c['mixing_block_records'] != 511 or c['loss_method'] != METHOD
            or c['seed'] != 2026112305 or c['validation_and_game_steps'] != [0, 625, 1250, 1875, 2500]
            or c['learning_rate'] != 3e-6 or c['head_learning_rate'] != 1.5e-5
            or c['expected_record_exposures'] != 182494 or sealed['status'] != STATUS
            or sealed['method'] != PARENT_METHOD or c['optimizer_resume'] or not c['new_optimizer']):
        raise ValueError('Exact fixed yaku2500 continuation recipe required')
    parent = read(sealed['parent_registration']); pc = parent['curriculum']
    data_path = Path(sealed['child_data_ready']); data = read(data_path); recipe = read(ROOT/c['epoch_recipe'])
    objective = read(ROOT/c['objective_preflight']); profile = read(ROOT/c['validation_adapter_preflight'])
    if (objective['method'] != METHOD or objective['actual_TRAIN_records'] != 91247
            or objective['maximum_actual_objective_or_CE_difference'] != 0
            or objective['maximum_representative_gradient_difference'] != 0
            or profile['method'] != METHOD or profile['validation_menus'] != 24516
            or not profile['canonical_CE_and_every_classification_metric_unchanged']
            or not profile['retained_nine_component_profile_and_signed_accounting_verified']):
        raise ValueError('Actual retained-objective TRAIN and full-profile contracts required')
    if (recipe['status'] != 'actual_TRAIN_yaku25_weights_objective_and_two_epoch_orders_ready_not_parent_or_fit_registered'
            or recipe['actual_optimizer_steps'] != 0 or recipe['proposed_loss_mass'] != MASS):
        raise ValueError('Reuse only the prepared actual numerical source weights and epoch orders')
    for p, h in sealed['bound_files'].items():
        if sha(p) != h: raise ValueError('Sealed input changed: '+p)
    counts = Counter(json.loads(line)['questions']['action']['src'] for line in (ROOT/data['files']['train']['path']).open())
    source_weights = weights(dict(counts))
    if (sum(counts.values()) != 91247 or data['files']['validation']['records'] != 24516
            or dict(counts) != recipe['source_counts'] or source_weights != recipe['proposed_example_weights']):
        raise ValueError('Actual25 source membership or prepared weights changed')
    for name in ('base', 'base_revision', 'dropout', 'fresh_lora_rank', 'head_dim', 'curve_seed_start',
                 'curve_seed_groups', 'stage_end_seed_groups', 'opponent_profiles', 'reference_checkpoint',
                 'reference_identity', 'reference_evaluation_cache', 'weight_decay'):
        c[name] = pc[name]
    c.update(data_ready=str(data_path), parent_seal=str(seal_path.resolve()), riichi_input_schema=None,
        parent_identity=sealed['model']['identity_sha256'], parent_policy_identity=sealed['policy_identity'],
        parent_evaluation_cache=sealed['parent_evaluation_cache'],
        original_guarded_evaluation_cache=sealed['original_guarded_evaluation_cache'],
        original_guarded_identity=sealed['original_guarded_identity'], loss_mass=MASS)
    plan = ROOT/c['plan_root']
    if plan.exists() or (ROOT/c['out']).exists(): raise ValueError('Never overwrite a registration or fit')
    plan.mkdir(parents=True)
    args = copy.deepcopy(tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())); args.pop('resume', None)
    args.update(data=str((ROOT/data['files']['train']['path']).resolve()),
        eval_data=str((ROOT/data['files']['validation']['path']).resolve()),
        out=str((ROOT/c['out']/'training-run').resolve()), init_from=sealed['model']['directory'],
        epochs=2, max_steps=2500, lr=3e-6, head_lr=1.5e-5, batch=1, accum=73, seed=c['seed'],
        checkpoint_every_steps=125, eval_every_steps=625, early_stop_patience=0)
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
        trainer_config=str(trainer), trainer_config_sha256=sha(trainer),
        data_receipt=str(receipt), data_receipt_sha256=sha(receipt), registered_final_step=2500,
        evaluation_candidate_rule='final_registered_budget_v1', riichi_input_schema=None, validation_phase_weights=MASS,
        parent_initialization=dict(method=PARENT_METHOD, manifest=str(seal_path.resolve()), sha256=sha(seal_path),
            identity_sha256=c['parent_identity'], policy_identity_sha256=c['parent_policy_identity'], selected_step=2472,
            source=parent['source'], source_sha256=parent['source_sha256'], run=str((ROOT/pc['out']).resolve()),
            child_data_ready=str(data_path)))
    cfg_path = plan/'training-config.json'; write(cfg_path, cfg); verify_parent(cfg)
    files = dict(sealed['bound_files'])
    paths = [config, seal_path, trainer, receipt, cfg_path, ROOT/c['objective_preflight'],
        ROOT/c['validation_adapter_preflight'], ROOT/c['epoch_recipe']]
    paths += [ROOT/name for name in PUBLIC_FILES]
    for cache in (c['parent_evaluation_cache'], c['original_guarded_evaluation_cache'], c['reference_evaluation_cache']):
        paths += [ROOT/cache/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl')]
    paths += [ROOT/name for name in ('tools/evaluate_curriculum_yaku_cases_v1.py',
        'tools/curriculum_mortal_source.py', 'tools/curriculum_case_order.py',
        'tools/curriculum_endgame_ledger.py', 'tools/audit_curriculum_endgame_cases.py',
        'cluster/curriculum-yaku-v8-cases-v1.slurm',
        'docs/experiments/plans/curriculum-yaku-validation-cases-20261123.md')]
    paths += [ROOT/c['conditional_reference_cache']/name for name in
        ('registered-private.json', 'summary-private.json', 'case-outcomes-private.jsonl', 'completion-binding-private.json')]
    for path in paths: files[str(Path(path).resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Registered input changed')
    reg = dict(status='yaku2500_registered_not_started', curriculum=c, training_config=cfg,
        source=parent['source'], source_sha256=parent['source_sha256'], inputs=files, registered_final_step=2500,
        train_records=91247, validation_records=24516, expected_record_exposures=182494,
        course_loss_weights=source_weights, course_loss_mass=MASS, expected_orders=recipe['expected_orders'],
        parent_model=sealed['model'], new_optimizer=True, optimizer_updates=0,
        same_execution_policy_as_parent=True, original_parent_grade_not_overridden=True,
        retained_parent_replay_objective_explicit=True)
    require_registration(reg)
    write(plan/'registered.json', reg)
    print(json.dumps(dict(status=reg['status'], updates=2500, exposures=182494)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--parent-seal', type=Path, required=True); a = p.parse_args()
    register(a.config.resolve(), a.parent_seal.resolve())

