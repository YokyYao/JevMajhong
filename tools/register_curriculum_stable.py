"""Register a fresh-base balanced-order stage1 fit without altering the v1 recipe."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def register(config,base_path,reference_path):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),'Scheduled GPU required'
    c=read(config);boot=read(ROOT/'runs/curriculum-reproduction-v1/bootstrap.json')
    sys.path.insert(0,boot['training_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    for key,package in (('training','jevmahjong'),('upstream','jevany')):
        if package_digest(Path(boot[key+'_source'])/package)!=boot[key+'_source_sha256']:raise ValueError('Frozen package changed')
    spec=read(ROOT/'configs/environment/curriculum-base-files.json')
    base_path=base_path.resolve();base_files={str(base_path/name):h for name,h in spec['files'].items()}
    for name,h in base_files.items():
        if sha(name)!=h:raise ValueError('Pinned base snapshot differs: '+name)
    original_reference=c['reference_identity'];reference_path=reference_path.resolve();identity=bundle(reference_path)
    if identity['identity_sha256']!=original_reference:
        import torch
        release=read(reference_path/'play-v2-identity.json')
        if release['source_identity_sha256']!=original_reference or not release['parameter_values_unchanged']:
            raise ValueError('Public v8 export lacks source-parameter identity')
        for name,h in release['files'].items():
            if identity['files'].get(name)!=h:raise ValueError('Public v8 export file changed')
        head=torch.load(reference_path/'head.pt',map_location='cpu',weights_only=True)['head'];digest=hashlib.sha256()
        for name in sorted(head):
            t=head[name].detach().cpu().contiguous();digest.update(name.encode());digest.update(str(t.dtype).encode())
            digest.update(str(tuple(t.shape)).encode());digest.update(t.numpy().tobytes())
        if digest.hexdigest()!='2569a706de18cb1f1b920524fb94bdf50cc4562f260479d6bc6cad536d94756e':
            raise ValueError('Reference head tensors differ from v8')
    c.update(reference_checkpoint=str(reference_path),reference_identity=identity['identity_sha256'])
    data_root=ROOT/c['data_root'];data=read(data_root/'data-ready.json')
    if data['status']!='curriculum_stage1_data_ready' or data['train_validation_group_overlap']:raise ValueError('Data not admitted')
    for v in data['stages']['stage1'].values():
        if sha(ROOT/v['path'])!=v['sha256']:raise ValueError('Admitted data changed')
    if not c.get('balanced_epoch_order') or c['batch'] != 1:
        raise ValueError('This recipe requires balanced epoch ordering and microbatch1')
    plan=ROOT/c['plan_root'];plan.mkdir(parents=True,exist_ok=False)
    if (ROOT/c['out']).exists():raise ValueError('Existing fit must not be overwritten')
    n=data['stages']['stage1']['train']['records'];steps=c['epochs']*math.ceil(n/(c['batch']*c['accumulation']))
    args=dict(base=c['base'],base_revision=c['base_revision'],data=str((ROOT/data['stages']['stage1']['train']['path']).resolve()),
        eval_data=str((ROOT/data['stages']['stage1']['validation']['path']).resolve()),out=str(ROOT/c['out']/'training-run'),
        device='cuda',weights_dtype='bf16',dtype='bf16',lora=c['fresh_lora_rank'],lora_targets='all',lora_dropout=c['dropout'],
        head_dim=c['head_dim'],decision_mode='pointer',branch_mode='rows',backbone_adapter='text',
        epochs=c['epochs'],max_steps=steps,lr=c['learning_rate'],head_lr=c['head_learning_rate'],weight_decay=c['weight_decay'],
        batch=c['batch'],accum=c['accumulation'],checkpointing=1,checkpoint_every_steps=c['validation_every_steps'],
        eval_before_start=True,eval_every_steps=c['validation_every_steps'],eval_records=0,early_stop_patience=0,
        early_stop_metric='development_nll',seed=c['seed'],p_none=0.,p_none_distract=0.,p_distract=0.,p_none_pair=0.,
        max_state=4096,max_branch=8192,max_packed=8192,wandb_mode='disabled')
    trainer=plan/'trainer.toml';trainer.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k,v in args.items()))
    receipt=plan/'data-receipt.json';write(receipt,dict(status='balanced_raw_data_ready',requires_masked_soft_target_loss=True,
        files=data['stages']['stage1'],fresh_source_group_split=True,win_choice_supervision=False))
    cfg=dict(base=c['base'],base_revision=c['base_revision'],base_load_path=str(base_path),base_files=base_files,
        upstream_source=boot['upstream_source'],upstream_source_sha256=boot['upstream_source_sha256'],model_id=c['run_id'],
        out=c['out'],initialization='fresh',trainer_config=str(trainer),trainer_config_sha256=sha(trainer),
        data_receipt=str(receipt),data_receipt_sha256=sha(receipt),validation_phase_weights={'closed_discard':1.},
        evaluation_candidate_rule='final_registered_budget_v1',registered_final_step=steps)
    training_config=plan/'training-config.json';write(training_config,cfg)
    paths=[config,Path(__file__).resolve(),ROOT/'tools/train_curriculum_stable.py',ROOT/'tools/curriculum_order.py',
        ROOT/'tests/core/test_curriculum_order.py',ROOT/'tools/plot_curriculum_fixed_grid.py',
        ROOT/'tools/train_curriculum.py',ROOT/'tools/evaluate_curriculum.py',
        ROOT/'tools/build_curriculum_stage1.py',ROOT/'tools/audit_curriculum_shapes.py',ROOT/'tools/plot_curriculum.py',
        ROOT/'tests/core/test_curriculum_data.py',ROOT/'tests/core/test_metric_journal.py',data_root/'data-ready.json',
        data_root/'reserved-public-groups.json',data_root/'common-validation.jsonl',training_config,trainer,receipt]
    inputs={str(p.resolve()):sha(p) for p in paths};inputs.update(base_files)
    inputs.update({str((ROOT/v['path']).resolve()):v['sha256'] for v in data['stages']['stage1'].values()})
    if c.get('reference_evaluation_cache'):
        cached=ROOT/c['reference_evaluation_cache']
        for name in ('protocol.json','summary.json','matches-private.jsonl'):
            p=cached/name
            inputs[str(p.resolve())]=sha(p)
        if read(cached/'summary.json')['checkpoint_sha256']!=c['reference_identity']:
            raise ValueError('Cached v8 reference is a different policy')
    frozen=dict(status='stage1_registered_not_started',curriculum=c,training_config=cfg,source=boot['training_source'],
        source_sha256=boot['training_source_sha256'],inputs=inputs,registered_final_step=steps,train_records=n,
        stage1_loss_weights=data['stage1_loss_weights'],stage1_loss_mass=data['stage1_loss_mass'],
        public_reference_export_same_parameters=identity['identity_sha256']!=original_reference,optimizer_updates=0)
    write(plan/'registered.json',frozen);print(json.dumps(dict(status=frozen['status'],final_step=steps,train_records=n)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--base-path',type=Path,required=True)
    p.add_argument('--reference-path',type=Path,required=True);a=p.parse_args();register(a.config.resolve(),a.base_path,a.reference_path)
