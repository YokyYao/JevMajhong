"""Register the complete388-update, explicitly route-bound calls course."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write, MASS
from tools.curriculum_calls_classroute_parent_v1 import METHOD as PARENT_METHOD, STATUS, verify_parent
from tools.curriculum_calls_classroute_v1 import METHOD, require_recipe
from tools.curriculum_parent import rows, require_replay
from tools.curriculum_transfer import read, sha

PUBLIC_FILES = (
    'tools/curriculum_calls_classroute_v1.py', 'tools/curriculum_calls_classroute_parent_v1.py',
    'tools/register_curriculum_calls_classroute_v1.py', 'tools/train_curriculum_calls_classroute_v1.py',
    'tools/evaluate_curriculum_calls_classroute_v1.py', 'tools/audit_curriculum_calls_classroute_v1.py',
    'tools/preflight_curriculum_calls_classroute_v1.py',
    'tests/core/test_curriculum_calls_classroute_v1.py',
    'configs/training/curriculum-calls-classroute-v1.json',
    'docs/experiments/plans/curriculum-calls-classroute-20261123.md',
    'cluster/curriculum-calls-classroute-v1.slurm',
)


def register(config, seal_path):
    gpu_only(); sealed = read(seal_path)
    if sealed['status'] != STATUS or sealed['method'] != PARENT_METHOD:
        raise ValueError('New distinct-policy parent seal required')
    parent = read(sealed['parent_registration']); pc = parent['curriculum']
    c = read(config); c['loss_mass'] = MASS
    preflight = ROOT/'runs/curriculum-calls-classroute-v1/preflight-v1'
    if read(preflight/'summary-private.json')['status'] != 'calls13_classroute_full_inputs_CUDA_loss_and_cached_metrics_preflight_passed':
        raise ValueError('Actual thirteen-source runtime preflight required')
    preflight_files = read(preflight/'binding-private.json')['files']
    if any(sha(p) != h for p, h in preflight_files.items()):
        raise ValueError('Preflight-bound implementation changed')
    data_path = Path(sealed['child_data_ready']); data = read(data_path)
    proposal = require_recipe(c, data)
    for p, h in sealed['bound_files'].items():
        if sha(p) != h: raise ValueError('Sealed parent input changed: '+p)
    args_parent = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    pools = {s: rows(ROOT/spec['path']) for s, spec in data['files'].items()}
    for split, pool in pools.items():
        if (Counter(r['questions']['action']['src'].removeprefix('curriculum2:') for r in pool) != data['categories'][split]
                or any(r['_meta']['label_type'] != 'expert_replay' or r['questions']['action'].get('target') is not None for r in pool)):
            raise ValueError('Exact thirteen-source expert hard supervision required')
    require_replay(rows(args_parent['data']), pools['train'])
    require_replay(rows(args_parent['eval_data']), pools['validation'])
    for name in ('base', 'base_revision', 'dropout', 'fresh_lora_rank', 'head_dim', 'curve_seed_start',
                 'curve_seed_groups', 'stage_end_seed_groups', 'opponent_profiles', 'reference_checkpoint',
                 'reference_identity', 'reference_evaluation_cache', 'weight_decay'):
        c[name] = pc[name]
    c.update(data_ready=str(data_path), parent_seal=str(seal_path.resolve()), riichi_input_schema=None,
        parent_identity=sealed['model']['identity_sha256'], parent_policy_identity=sealed['policy_identity'],
        parent_evaluation_cache=sealed['parent_evaluation_cache'])
    plan = ROOT/c['plan_root']
    if plan.exists() or (ROOT/c['out']).exists(): raise ValueError('Never replace registration or existing fit')
    plan.mkdir(parents=True)
    args = copy.deepcopy(args_parent); args.pop('resume', None)
    args.update(data=str((ROOT/data['files']['train']['path']).resolve()),
        eval_data=str((ROOT/data['files']['validation']['path']).resolve()), out=str(ROOT/c['out']/'training-run'),
        init_from=sealed['model']['directory'], epochs=2, max_steps=388, lr=c['learning_rate'],
        head_lr=c['head_learning_rate'], batch=1, accum=253, seed=c['seed'],
        checkpoint_every_steps=194, eval_every_steps=194, early_stop_patience=0)
    trainer = plan/'trainer.toml'
    with trainer.open('x') as stream:
        stream.write(''.join(f'{k} = {json.dumps(v)}\n' for k, v in args.items()))
    receipt = plan/'data-receipt.json'
    write(receipt, dict(status='balanced_raw_data_ready', requires_masked_soft_target_loss=True,
        files=data['files'], fresh_source_group_split=False, win_choice_supervision=False, exact_stage1_replay=True))
    cfg = copy.deepcopy(parent['training_config'])
    cfg.update(model_id=c['run_id'], out=c['out'], initialization='continued', loss_method=METHOD,
        trainer_config=str(trainer), trainer_config_sha256=sha(trainer), data_receipt=str(receipt),
        data_receipt_sha256=sha(receipt), registered_final_step=388, riichi_input_schema=None,
        validation_phase_weights=dict(closed_discard=.657, open_discard=.343/3, call_window=.343/3, kan_window=.343/3),
        parent_initialization=dict(method=PARENT_METHOD, manifest=str(seal_path.resolve()), sha256=sha(seal_path),
            identity_sha256=c['parent_identity'], policy_identity_sha256=c['parent_policy_identity'], selected_step=608,
            source=parent['source'], source_sha256=parent['source_sha256'],
            run=str((ROOT/pc['out']).resolve()), child_data_ready=str(data_path)))
    cfg.pop('control_registration', None)
    training_config = plan/'training-config.json'; write(training_config, cfg)
    verify_parent(cfg)
    files = dict(sealed['bound_files']); files.update(preflight_files)
    paths = [config, seal_path, trainer, receipt, training_config,
             preflight/'summary-private.json', preflight/'binding-private.json']
    paths.extend(ROOT/name for name in PUBLIC_FILES)
    for cache in (c['reference_evaluation_cache'], c['parent_evaluation_cache']):
        paths.extend(ROOT/cache/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'))
    for path in paths:
        files[str(path.absolute())] = sha(path)
        files[str(path.resolve())] = sha(path)
    reg = dict(status='stage2_calls_classroute_registered_not_started', curriculum=c, training_config=cfg,
        source=parent['source'], source_sha256=parent['source_sha256'], inputs=files, registered_final_step=388,
        train_records=49022, course_loss_weights=data['example_weights'], course_loss_mass=MASS,
        optimizer_updates=0, parent_model=sealed['model'], new_optimizer=True,
        distinct_policy_source_quality_required=True, expected_record_exposures=proposal['record_exposures'])
    write(plan/'registered.json', reg)
    print(json.dumps(dict(status=reg['status'], final_step=388, exposures=proposal['record_exposures'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--parent-seal', type=Path, required=True); a = p.parse_args()
    register(a.config.resolve(), a.parent_seal.resolve())
