"""Register a sealed-parent expert calling course; no model update here."""
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha,verify_curriculum_parent
from tools.curriculum_parent import require_child_input_contract,require_replay
from tools.audit_curriculum_accumulation import suggest_accumulation


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def register(config,parent_seal):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    c=read(config);seal=read(parent_seal);parent=read(seal['parent_registration']);pc=parent['curriculum']
    if c['stage']!='stage2-calls-v1' or c['batch']!=1 or c.get('optimizer_resume'):
        raise ValueError('Explicit calling course with a new optimizer required')
    data_ready=Path(seal['child_data_ready']);data=read(data_ready);require_child_input_contract(parent,data)
    if data['status']!='stage2_data_prepared_pending_parent_and_fit_registration':raise ValueError('Wrong child data scope')
    for spec in data['files'].values():
        if sha(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Prepared child input changed')
    load=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines() if s]
    pools={split:load(ROOT/spec['path']) for split,spec in data['files'].items()}
    if any(r['_meta']['label_type']!='expert_replay' or r['questions']['action'].get('target') is not None
           for pool in pools.values() for r in pool):raise ValueError('Calling course requires expert hard labels in both splits')
    args_parent=tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    require_replay(load(args_parent['data']),pools['train']);require_replay(load(args_parent['eval_data']),pools['validation'])
    counts=Counter(r['questions']['action']['src'].removeprefix('curriculum2:') for r in pools['train'])
    if dict(counts)!=data['categories']['train'] or set(counts)!={'plain','riichi_yes','riichi_no','open','call','kan'}:
        raise ValueError('Unexpected expert source strata')
    expected=suggest_accumulation(len(pools['train']),128,min(counts.values()))
    if c['accumulation']!=expected['accumulation'] or expected['final_fraction']<c['minimum_tail_fraction']:
        raise ValueError('The count-only accumulation proposal changed')
    if c['stage_offset']!=seal['selected_step']:raise ValueError('Parent cumulative offset changed')
    for key in ('base','base_revision','dropout','fresh_lora_rank','head_dim','curve_seed_start','curve_seed_groups',
                'stage_end_seed_groups','opponent_profiles','reference_checkpoint','reference_identity','reference_evaluation_cache','weight_decay'):
        c[key]=pc[key]
    c.update(data_ready=str(data_ready.resolve()),riichi_input_schema=pc.get('riichi_input_schema'),
        parent_seal=str(parent_seal.resolve()),parent_identity=seal['model']['identity_sha256'],
        parent_evaluation_cache=str((ROOT/pc['out']/'evaluations'/f"step-{seal['selected_step']:06d}").resolve()))
    for key in ('plain','open','call','kan'):
        if data['loss_mass'][key]!=c['loss_mass'][key]:raise ValueError('Proposed source loss mass changed')
    if abs(data['loss_mass']['riichi_yes']+data['loss_mass']['riichi_no']-c['loss_mass']['riichi'])>1e-12:
        raise ValueError('Riichi replay loss mass changed')
    steps=c['epochs']*math.ceil(len(pools['train'])/c['accumulation'])
    if steps!=324:raise ValueError('Registered calling-course exposure changed')
    plan=ROOT/c['plan_root'];plan.mkdir(parents=True,exist_ok=False)
    if (ROOT/c['out']).exists():raise ValueError('Never overwrite an existing fit')
    args=copy.deepcopy(args_parent)
    args.update(data=str((ROOT/data['files']['train']['path']).resolve()),eval_data=str((ROOT/data['files']['validation']['path']).resolve()),
        out=str(ROOT/c['out']/'training-run'),init_from=seal['model']['directory'],epochs=c['epochs'],max_steps=steps,
        lr=c['learning_rate'],head_lr=c['head_learning_rate'],batch=1,accum=c['accumulation'],seed=c['seed'],
        checkpoint_every_steps=c['validation_every_steps'],eval_every_steps=c['validation_every_steps'],early_stop_patience=0)
    args.pop('resume',None)
    trainer=plan/'trainer.toml';trainer.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k,v in args.items()))
    receipt=plan/'data-receipt.json';write(receipt,dict(status='balanced_raw_data_ready',requires_masked_soft_target_loss=True,
        files=data['files'],fresh_source_group_split=False,win_choice_supervision=False,exact_stage1_replay=True))
    cfg=copy.deepcopy(parent['training_config'])
    cfg.update(model_id=c['run_id'],out=c['out'],initialization='continued',trainer_config=str(trainer),trainer_config_sha256=sha(trainer),
        data_receipt=str(receipt),data_receipt_sha256=sha(receipt),registered_final_step=steps,riichi_input_schema=c['riichi_input_schema'],
        validation_phase_weights=dict(closed_discard=.35,open_discard=.25,call_window=.25,kan_window=.15),
        parent_initialization=dict(method=seal['method'],manifest=str(parent_seal.resolve()),sha256=sha(parent_seal),
            identity_sha256=seal['model']['identity_sha256'],selected_step=seal['selected_step'],source=parent['source'],
            source_sha256=parent['source_sha256'],run=str((ROOT/pc['out']).resolve()),child_data_ready=str(data_ready.resolve())))
    training_config=plan/'training-config.json';write(training_config,cfg)
    verify_curriculum_parent(cfg)
    paths=[config,parent_seal,data_ready,trainer,receipt,training_config,Path(__file__),
        ROOT/'tools/train_curriculum_course.py',ROOT/'tools/audit_curriculum_course.py',ROOT/'tools/curriculum_course_order.py',
        ROOT/'tools/curriculum_transfer.py',ROOT/'tools/curriculum_parent.py',ROOT/'tools/curriculum_research_parent.py',
        ROOT/'tools/evaluate_curriculum.py',ROOT/'tools/evaluate_curriculum_context.py',ROOT/'tools/curriculum_input_contract.py',
        ROOT/'tools/curriculum_riichi_context.py',ROOT/'tools/plot_curriculum.py',ROOT/'tools/plot_curriculum_fixed_grid.py',
        ROOT/'tests/core/test_curriculum_course_order.py',ROOT/'tests/core/test_curriculum_transfer.py',ROOT/'tests/core/test_curriculum_research_parent.py']
    files=dict(seal['bound_files']);files.update({str(p.resolve()):sha(p) for p in paths})
    for cache in (c['reference_evaluation_cache'],c['parent_evaluation_cache']):
        for name in ('protocol.json','summary.json','matches-private.jsonl'):
            p=ROOT/cache/name;files[str(p.resolve())]=sha(p)
    reg=dict(status='stage2_registered_not_started',curriculum=c,training_config=cfg,source=parent['source'],
        source_sha256=parent['source_sha256'],inputs=files,registered_final_step=steps,train_records=len(pools['train']),
        course_loss_weights=data['example_weights'],course_loss_mass=data['loss_mass'],optimizer_updates=0,
        parent_model=seal['model'],new_optimizer=True)
    write(plan/'registered.json',reg)
    print(json.dumps(dict(status=reg['status'],final_step=steps,train_records=reg['train_records'],parent_identity=c['parent_identity'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--parent-seal',type=Path,required=True)
    a=p.parse_args();register(a.config.resolve(),a.parent_seal.resolve())
