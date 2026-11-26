"""Freeze phase4 data/parent/budget; fit launch needs its own bound preflight."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_placement_parent_v1 import METHOD as PARENT_METHOD, STATUS, verify_parent
from tools.curriculum_placement_mix_v1 import masses
from tools.curriculum_placement_objective_v1 import METHOD as LOSS
from tools.curriculum_placement_input_v1 import SCHEMA
from tools.curriculum_placement_full_win_policy_v1 import METHOD as ROUTE
from tools.prepare_curriculum_placement_exposures_v1 import proposed_schedule

PUBLIC_FILES = (
    'tools/register_curriculum_placement_v1.py', 'configs/training/curriculum-placement-fit-v1.json',
    'tests/core/test_curriculum_placement_registration_v1.py',
    'docs/experiments/plans/curriculum-placement-registration-20261123.md',
    'tools/curriculum_placement_parent_v1.py', 'tools/curriculum_placement_mix_v1.py',
    'tools/curriculum_placement_exposures_v1.py', 'tools/curriculum_placement_partition_v1.py',
    'tools/curriculum_placement_objective_v1.py', 'tools/curriculum_placement_metrics_v1.py',
    'tools/curriculum_placement_input_v1.py', 'tools/curriculum_placement_full_win_policy_v1.py',
    'tools/play_curriculum_placement_v1.py', 'tools/curriculum_checkpoint_evaluation_fence_v1.py',
    'tools/curriculum_placement_runtime_v1.py', 'tests/core/test_curriculum_placement_runtime_v1.py',
    'tools/train_curriculum_placement_course_v1.py', 'tools/evaluate_curriculum_placement_course_v1.py',
    'cluster/curriculum-placement-register-v1.slurm',
)


def require_recipe(c, data, sealed):
    mass = {s: float(v) for s, v in masses().items()}
    if (c['status'] != 'planned_not_registered_or_fit_admitted' or c['seed'] != 2026112306
            or c['epochs'] != 2 or c['batch'] != 1 or c['learning_rate'] != 3e-6
            or c['head_learning_rate'] != 1.5e-5 or not c['new_optimizer'] or c['optimizer_resume']
            or not c['final_candidate_only'] or c['automatic_release']
            or c['loss_method'] != LOSS or c['selection_method'] != ROUTE
            or c['placement_input_schema'] != SCHEMA
            or sealed['status'] != STATUS or sealed['method'] != PARENT_METHOD
            or sealed['selected_step'] != 2500 or sealed['original_parent_grade']['passed'] is not True
            or sealed['same_execution_policy_as_parent'] or not sealed['parent_games_not_admitted_as_child_step0_cache']
            or data['status'] != 'placement_expanded_corpus_and_two_epoch_recipe_prepared_not_parent_or_fit_registered'
            or data['optimizer_updates'] != 0 or data['student_fit_registered']
            or data['source_loss_mass'] != mass or data['order_seed'] != c['seed']):
        raise ValueError('Qualified final3B plus exact fixed placement recipe required')
    counts = data['source_exposures']; weights = data['scalar_example_weights']
    total = data['files']['train']['records']
    if (set(counts) != set(mass) or set(weights) != set(mass)
            or any(type(n) is not int or n <= 0 for n in counts.values())
            or sum(counts.values()) != total or data['training_exposures_per_pass'] != total
            or data['files']['validation']['records'] != data['unique_validation_records']
            or any(not math.isfinite(v) or not 0 < v <= 16 for v in weights.values())
            or any(abs(counts[s]*weights[s]/total-mass[s]) > 1e-12 for s in mass)):
        raise ValueError('Actual57-source exposure counts and exact bounded loss mass required')
    schedule = proposed_schedule(total)
    if data['proposed_schedule'] != schedule:
        raise ValueError('Only the actual count-derived full two-epoch schedule is eligible')
    return schedule


def register(config, seal_path):
    gpu_only(); c = read(config); sealed = read(seal_path)
    data_path = Path(sealed['child_data_ready']); data = read(data_path)
    schedule = require_recipe(c, data, sealed)
    parent = read(sealed['parent_registration']); pc = parent['curriculum']
    sys.path.insert(0, parent['source']); sys.path.insert(0, parent['training_config']['upstream_source'])
    from jevany.resume import dataset_fingerprint
    from tools.curriculum_placement_partition_v1 import require_partition
    files = dict(sealed['bound_files'])
    for name, expected in files.items():
        if sha(name) != expected: raise ValueError('Sealed prerequisite changed: '+name)
    train_path = ROOT/data['files']['train']['path']; validation_path = ROOT/data['files']['validation']['path']
    training = [json.loads(line) for line in train_path.open()]
    validation = [json.loads(line) for line in validation_path.open()]
    counts = dict(Counter(r['questions']['action']['src'] for r in training))
    if (counts != data['source_exposures'] or len(validation) != data['files']['validation']['records']
            or dataset_fingerprint(training) != data['actual_expanded_dataset_fingerprint']):
        raise ValueError('Actual TRAIN membership/fingerprint or validation count differs from preparation')
    membership_path = Path(sealed['child_membership_file']); manifest = read(membership_path)
    require_partition(validation, manifest, 'validation')
    order_path = data_path.parent/'expected-orders-private.json'; orders = read(order_path)
    if (orders['seed'] != c['seed'] or len(orders['epochs']) != 2
            or [r['order_sha256'] for r in orders['epochs']] != data['expected_order_sha256']):
        raise ValueError('Two exact prepared epoch orders required')
    for key in ('base', 'base_revision', 'dropout', 'fresh_lora_rank', 'head_dim', 'curve_seed_start',
                'curve_seed_groups', 'stage_end_seed_groups', 'opponent_profiles', 'reference_checkpoint',
                'reference_identity', 'reference_evaluation_cache', 'weight_decay'):
        c[key] = pc[key]
    c.update(data_ready=str(data_path), parent_seal=str(seal_path.resolve()),
        parent_identity=sealed['model']['identity_sha256'], parent_policy_identity=sealed['parent_policy_identity'],
        stage_offset=pc['stage_offset']+parent['registered_final_step'],
        parent_evaluation_cache=str((ROOT/pc['out']/'evaluations/step-002500').resolve()),
        stage_zero_seed_groups=8, zero_curve_use_matching16_subset=True,
        parent_games_not_step0_cache=True, expected_record_exposures=schedule['expected_record_exposures'],
        accumulation=schedule['accumulation'], final_optimizer_steps=schedule['proposed_final_step'],
        checkpoint_every_steps=schedule['checkpoint_every_steps'], eval_every_steps=schedule['eval_every_steps'],
        validation_and_game_steps=schedule['validation_and_game_steps'],
        development_acceptance_must_be_frozen_by_prelaunch=True,
        early_input_parity_and_exact_tensor_proof_required=True,
        new_policy_zero64_actual_games_required=True)
    plan = ROOT/c['plan_root']; out = ROOT/c['out']
    if plan.exists() or out.exists(): raise ValueError('Never overwrite a registration or fit')
    plan.mkdir(parents=True)
    args = copy.deepcopy(tomllib.loads(Path(parent['training_config']['trainer_config']).read_text()))
    args.pop('resume', None)
    args.update(data=str(train_path.resolve()), eval_data=str(validation_path.resolve()),
        out=str((out/'training-run').resolve()), init_from=sealed['model']['directory'],
        epochs=2, max_steps=c['final_optimizer_steps'], lr=c['learning_rate'], head_lr=c['head_learning_rate'],
        batch=1, accum=c['accumulation'], seed=c['seed'], checkpoint_every_steps=100,
        eval_every_steps=c['eval_every_steps'], early_stop_patience=0)
    trainer = plan/'trainer.toml'
    trainer.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k, v in args.items()))
    actual_files = {split: dict(data['files'][split],path=str((ROOT/data['files'][split]['path']).resolve()))
                    for split in ('train', 'validation')}
    receipt = plan/'data-receipt.json'
    write(receipt, dict(status='balanced_raw_data_ready', requires_masked_soft_target_loss=True,
        files=actual_files, original_current_row_objects_unchanged=True,
        virtual_copies_have_explicit_provenance=True, validation_file_reused_unchanged=True,
        full_source_group_exclusions=True, win_choice_supervision=True))
    cfg = copy.deepcopy(parent['training_config'])
    for key in ('control_registration','label_source_gradient_mass','paired_future_phase_mass'):cfg.pop(key,None)
    cfg.update(model_id=c['run_id'], out=c['out'], initialization='continued',loss_method=LOSS,
        trainer_config=str(trainer),trainer_config_sha256=sha(trainer),
        data_receipt=str(receipt),data_receipt_sha256=sha(receipt),
        registered_final_step=c['final_optimizer_steps'],evaluation_candidate_rule='final_registered_budget_v1',
        riichi_input_schema=None,placement_input_schema=SCHEMA,placement_policy_method=ROUTE,
        validation_phase_weights=data['source_loss_mass'],
        parent_initialization=dict(method=PARENT_METHOD,manifest=str(seal_path.resolve()),sha256=sha(seal_path),
            identity_sha256=c['parent_identity'],policy_identity_sha256=c['parent_policy_identity'],selected_step=2500,
            source=parent['source'],source_sha256=parent['source_sha256'],
            run=str((ROOT/pc['out']).resolve()),child_data_ready=str(data_path)))
    cfg_path = plan/'training-config.json'; write(cfg_path,cfg); verify_parent(cfg)
    paths = [config,seal_path,data_path,membership_path,order_path,trainer,receipt,cfg_path]
    paths += [ROOT/name for name in PUBLIC_FILES]
    for cache,identity,policy in ((c['parent_evaluation_cache'],c['parent_identity'],c['parent_policy_identity']),
            (c['reference_evaluation_cache'],c['reference_identity'],None)):
        directory=ROOT/cache; summary=read(directory/'summary.json')
        if summary['checkpoint_sha256']!=identity or (policy is not None and summary['policy_identity']!=policy):
            raise ValueError('Reference cache has different actual parameters/policy')
        paths += [directory/name for name in ('protocol.json','summary.json','matches-private.jsonl')]
    for path in paths:
        name=str(Path(path).resolve());expected=sha(path)
        if name in files and files[name]!=expected:raise ValueError('Conflicting immutable registration input')
        files[name]=expected
    if any(sha(name)!=expected for name,expected in files.items()):raise ValueError('Registration inputs changed')
    result=dict(status='placement_data_parent_and_budget_registered_not_fit_admitted',curriculum=c,
        training_config=cfg,source=parent['source'],source_sha256=parent['source_sha256'],inputs=files,
        registered_final_step=c['final_optimizer_steps'],train_records=len(training),
        unique_train_records=data['unique_train_records'],validation_records=len(validation),
        expected_record_exposures=c['expected_record_exposures'],course_loss_weights=data['scalar_example_weights'],
        course_loss_mass=data['source_loss_mass'],expected_orders=orders['epochs'],
        expanded_dataset_fingerprint=data['actual_expanded_dataset_fingerprint'],
        parent_model=sealed['model'],child_policy_contract=sealed['child_policy'],
        same_execution_policy_as_parent=False,new_optimizer=True,optimizer_updates=0,
        fit_launch_admitted=False,prelaunch_integration_and_code_binding_required=True,
        final_reserved_grid_opened=False,automatic_release=False)
    write(plan/'registered.json',result)
    print(json.dumps(dict(status=result['status'],updates=result['registered_final_step'],
        exposures=result['expected_record_exposures'],fit_launch_admitted=False)),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--parent-seal',type=Path,required=True);a=p.parse_args()
    register(a.config.resolve(),a.parent_seal.resolve())
