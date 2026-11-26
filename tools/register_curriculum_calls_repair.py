"""Register complete twelve-source calling continuation from a qualified final."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha,METHOD
from tools.curriculum_calls_fit import require_recipe,verify_parent
from tools.curriculum_parent import rows,require_child_input_contract,require_replay,ParentNotReady
from tools.seal_curriculum_basic_repair import require_source_quality


def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def register(config,parent_seal):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Allocated scheduler GPU required')
    c=read(config);seal=read(parent_seal);parent=read(seal['parent_registration']);pc=parent['curriculum']
    if (seal.get('method')!=METHOD or seal.get('source_specific_parent_quality_checked') is not True
        or seal.get('source_specific_sealer')!='tools/seal_curriculum_basic_repair_v2.py'):
        raise ParentNotReady('Completed source-qualified v2 parent required before registration')
    parent_validation=rows(ROOT/pc['out']/'metrics/validation_metrics.jsonl')[-1]
    require_source_quality(parent,read(seal['parent_evidence']),parent_validation)
    data_ready=Path(seal['child_data_ready']).resolve();data=read(data_ready);require_child_input_contract(parent,data)
    preparation=data_ready.parent/'preparation-binding-private.json';prepared=read(preparation)['files']
    if any(sha(p)!=h for p,h in prepared.items()):raise ValueError('Child data preparation source changed')
    pools={split:rows(ROOT/spec['path']) for split,spec in data['files'].items()}
    for split,spec in data['files'].items():
        if sha(ROOT/spec['path'])!=spec['sha256'] or len(pools[split])!=spec['records']:raise ValueError('Prepared input changed')
        if any(r['_meta']['label_type']!='expert_replay' or r['questions']['action'].get('target') is not None for r in pools[split]):
            raise ValueError('Expert hard targets required')
        counts=Counter(r['questions']['action']['src'].removeprefix('curriculum2:') for r in pools[split])
        if counts!=data['categories'][split]:raise ValueError('Actual source counts differ')
    proposal=require_recipe(c,data,data['categories']['train']);steps=proposal['final_step']
    args_parent=tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    require_replay(rows(args_parent['data']),pools['train']);require_replay(rows(args_parent['eval_data']),pools['validation'])
    reserved=data_ready.parent/'reserved-public-groups.json';held={r['_meta']['group_id'] for r in pools['validation']}|set(read(reserved))
    if {r['_meta']['group_id'] for r in pools['train']}&held:raise ValueError('Child held-out/reserved overlap')
    if c['stage_offset']!=pc['stage_offset']+seal['selected_step']:raise ValueError('Cumulative parent offset changed')
    for key in ('base','base_revision','dropout','fresh_lora_rank','head_dim','curve_seed_start','curve_seed_groups',
        'stage_end_seed_groups','opponent_profiles','reference_checkpoint','reference_identity','reference_evaluation_cache','weight_decay'):
        c[key]=pc[key]
    if pc.get('riichi_input_schema') is not None:raise ValueError('Plain serving schema required')
    c.update(data_ready=str(data_ready),riichi_input_schema=None,parent_seal=str(parent_seal.resolve()),
        parent_identity=seal['model']['identity_sha256'],parent_evaluation_cache=str((ROOT/pc['out']/'evaluations'/f"step-{seal['selected_step']:06d}").resolve()))
    plan=ROOT/c['plan_root']
    if plan.exists() or (ROOT/c['out']).exists():raise ValueError('Never overwrite a registration or existing fit')
    plan.mkdir(parents=True,exist_ok=False)
    args=copy.deepcopy(args_parent);args.pop('resume',None)
    args.update(data=str((ROOT/data['files']['train']['path']).resolve()),eval_data=str((ROOT/data['files']['validation']['path']).resolve()),
        out=str(ROOT/c['out']/'training-run'),init_from=seal['model']['directory'],epochs=2,max_steps=steps,
        lr=c['learning_rate'],head_lr=c['head_learning_rate'],batch=1,accum=c['accumulation'],seed=c['seed'],
        checkpoint_every_steps=c['validation_every_steps'],eval_every_steps=c['validation_every_steps'],early_stop_patience=0)
    trainer=plan/'trainer.toml';trainer.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k,v in args.items()))
    receipt=plan/'data-receipt.json';write(receipt,dict(status='balanced_raw_data_ready',requires_masked_soft_target_loss=True,
        files=data['files'],fresh_source_group_split=False,win_choice_supervision=False,exact_stage1_replay=True))
    cfg=copy.deepcopy(parent['training_config'])
    cfg.update(model_id=c['run_id'],out=c['out'],initialization='continued',trainer_config=str(trainer),trainer_config_sha256=sha(trainer),
        data_receipt=str(receipt),data_receipt_sha256=sha(receipt),registered_final_step=steps,riichi_input_schema=None,
        validation_phase_weights=dict(closed_discard=.65,open_discard=.35/3,call_window=.35/3,kan_window=.35/3),
        parent_initialization=dict(method=seal['method'],manifest=str(parent_seal.resolve()),sha256=sha(parent_seal),
            identity_sha256=seal['model']['identity_sha256'],selected_step=seal['selected_step'],source=parent['source'],
            source_sha256=parent['source_sha256'],run=str((ROOT/pc['out']).resolve()),child_data_ready=str(data_ready)))
    training_config=plan/'training-config.json';write(training_config,cfg);verify_parent(cfg)
    paths=[config,parent_seal,data_ready,preparation,reserved,trainer,receipt,training_config,Path(__file__),
        ROOT/'tools/train_curriculum_calls_repair.py',ROOT/'tools/audit_curriculum_calls_repair.py',ROOT/'tools/curriculum_calls_fit.py',
        ROOT/'tools/curriculum_calls_repair.py',ROOT/'tools/build_curriculum_calls_repair.py',ROOT/'tools/curriculum_course_order.py',
        ROOT/'tools/curriculum_transfer.py',ROOT/'tools/curriculum_parent.py',ROOT/'tools/seal_curriculum_basic_repair.py',
        ROOT/'tools/seal_curriculum_basic_repair_v2.py',ROOT/'tools/curriculum_basic_repair.py',ROOT/'tools/evaluate_curriculum.py',
        ROOT/'tools/plot_curriculum.py',ROOT/'tools/plot_curriculum_fixed_grid.py',
        ROOT/'tests/core/test_curriculum_calls_repair.py',ROOT/'tests/core/test_curriculum_calls_fit.py',
        ROOT/'docs/experiments/plans/curriculum-calls-repair-fit-20261122.md']
    files=dict(seal['bound_files']);files.update(prepared)
    for cache in (c['reference_evaluation_cache'],c['parent_evaluation_cache']):
        for name in ('protocol.json','summary.json','matches-private.jsonl'):paths.append(ROOT/cache/name)
    files.update({str(p.resolve()):sha(p) for p in paths})
    reg=dict(status='stage2_calls_repair_registered_not_started',curriculum=c,training_config=cfg,
        source=parent['source'],source_sha256=parent['source_sha256'],inputs=files,registered_final_step=steps,
        train_records=len(pools['train']),course_loss_weights=data['example_weights'],course_loss_mass=data['loss_mass'],
        optimizer_updates=0,parent_model=seal['model'],new_optimizer=True,source_specific_parent_quality_required=True)
    write(plan/'registered.json',reg)
    print(json.dumps(dict(status=reg['status'],final_step=steps,records=reg['train_records'],exposures=proposal['record_exposures'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--parent-seal',type=Path,required=True)
    a=p.parse_args();register(a.config.resolve(),a.parent_seal.resolve())
