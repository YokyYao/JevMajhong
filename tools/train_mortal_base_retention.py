"""Retain East-endgame pursuit in the newly trained Mortal/base lineage.

Train the same LoRA/head, with no extra runtime network. Teacher decisions and
terminal-outcome targets are explicitly different sources of supervision.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_mortal_teacher import require_gpu,verify,sha
from tools.build_mortal_base_retention import configuration,source_registration,isolated
from tools.train_mortal_base import read_rows,learning_rate_scale,progress


def evaluate_rows(c,checkpoint,rows,teacher_weights):
    from tools.continue_endgame_fourth_lora import normal_predictions
    from tools.train_mortal_base import metrics
    from tools.train_targeted_endgame import fresh_metrics
    predictions=normal_predictions(checkpoint,rows)
    teacher=[r for r in rows if r['questions']['action']['src']!='retention_outcome']
    endgame=[r for r in rows if r['questions']['action']['src']=='retention_outcome']
    report=metrics(teacher,predictions,teacher_weights)
    ce=sum(-sum(v*math.log(max(predictions[r['_meta']['id']][k],1e-30))
        for k,v in r['questions']['action']['target'].items()) for r in endgame)/len(endgame)
    joint=c['teacher_loss_fraction']*report['weighted_ce']+(1-c['teacher_loss_fraction'])*ce
    return dict(teacher=report,endgame=fresh_metrics(endgame,predictions),
        terminal_target_ce=ce,joint_ce=joint),predictions


def fit(c,proof):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from jevany.model import DecisionModel,load_preprocessor
    from jevany.checkpoint import Checkpoint
    from jevany.data import materialize
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.targeted_refresh import balanced_batches
    out=ROOT/c['out'];verify(proof)
    data=json.loads((out/'data-ready-private.json').read_text())
    if (data['status']!='fresh_lineage_retention_data_ready' or not data['fresh_foundation_prior']
            or not data['no_old_prior_or_soft_targets'] or data['train']!=2096 or data['validation']!=520):
        raise ValueError('Audited fresh-lineage retention data required')
    verify({'files':data['files']});verify({'files':data['source_files']})
    verify(json.loads((out/'registered-data-code-private.json').read_text()))
    if (out/'training-run').exists():raise ValueError('Inspect an existing fit rather than restarting it')
    parent=Path(data['parent_checkpoint']);parent_step=data['parent_step']
    if bundle_identity(parent)!=data['parent_identity']:raise ValueError('Foundation parent changed')
    foundation=ROOT/c['foundation_run']
    fdata=json.loads((foundation/'data-ready-private.json').read_text())
    ffit=json.loads((foundation/'fit-result-private.json').read_text())
    train=read_rows(out/'train.jsonl');val=read_rows(out/'validation.jsonl');isolated(train,val)
    schedule=balanced_batches(train,c['batch_quotas'],c['max_updates'],seed=c['seed'])
    if set(i for batch in schedule[:c['minimum_observation_updates']] for i in batch)!=set(range(len(train))):
        raise ValueError('Every admitted example must be exposed before selection/stopping')
    protocol=dict(proof,data_files=data['files'],data_source_files=data['source_files'],
        parent_identity=data['parent_identity'],parent_step=parent_step,batch_schedule=schedule,
        loss_mass=data['loss_mass'],new_optimizer=True,no_extra_networks=True,old_v8_parameters_loaded=False)
    write(out/'training-protocol-private.json',protocol)
    torch.manual_seed(c['seed']);ck=Checkpoint(str(parent));meta=ck.meta
    head_meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True)
    if not(head_meta.get('base_only_initialization') and head_meta['policy_version']=='mortal-base-v1'):
        raise ValueError('This stage accepts only the audited new base foundation')
    tok=load_preprocessor(str(parent),multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter)
    model=DecisionModel(c['base_model'],tok,'cuda',lora=c['fresh_lora_rank'],head_dim=meta.head_dim,
        head_residual_dim=meta.head_residual_dim,lora_targets='all',lora_dropout=.05,
        option_isolation=meta.option_isolation,special_embeddings=meta.special_embeddings,
        multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter,branch_mode=meta.branch_mode,
        dtype=torch.bfloat16,decision_mode=meta.decision_mode)
    for config in model.lm.peft_config.values():config.base_model_name_or_path=meta.base
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.lm.config.use_cache=False;ck.warm_start(model,meta)
    adapter0=load_file(str(parent/'adapter_model.safetensors'),device='cpu');head0=head_meta['head']
    actual=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
    if (set(actual)!=set(adapter0) or any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in actual.items())
            or any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())):
        raise ValueError('Initial tensors do not exactly match the fresh foundation')
    hp=list(model.head.parameters());head_ids={id(p) for p in hp}
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(id(p) not in head_ids and 'lora_' not in n for n,p in named):raise ValueError('Backbone must stay frozen')
    lp=[p for _,p in named if id(p) not in head_ids];parameters=lp+hp
    if not lp or not hp:raise ValueError('LoRA and head must both train')
    optimizer=torch.optim.AdamW([{'params':lp,'lr':c['lora_learning_rate']},
        {'params':hp,'lr':c['head_learning_rate']}],weight_decay=c['weight_decay'])
    assert not optimizer.state
    write(out/'initialization-private.json',dict(exact_fresh_foundation_tensors=True,parent_step=parent_step,
        parent_identity=data['parent_identity'],fresh_base_lineage=True,old_v8_parameters_loaded=False,
        new_optimizer=True,new_runtime_networks=0,trainable_lora=sum(p.numel() for p in lp),
        trainable_head=sum(p.numel() for p in hp),job_id=os.environ['SLURM_JOB_ID']))
    enc={}
    for i,row in enumerate(train):
        value=model.encode(tok,materialize(row),max_state=4096,max_branch=8192,strict=True)
        if value.get('state_truncated') or len(value['ids'])>8192:raise ValueError('Training truncation forbidden')
        enc[row['_meta']['id']]=value
        if (i+1)%256==0:write(out/'training-progress-private.json',dict(status='encoding',encoded=i+1,total=len(train)))
    root=out/'training-run';state=None;curve=[];seen=set();started=time.monotonic()

    def evaluate(update):
        nonlocal state
        total=parent_step+update;path=root/'checkpoints'/f'step-{total:06d}'
        path.mkdir(parents=True,exist_ok=False);model.eval()
        model.lm.save_pretrained(path,save_embedding_layers=False);tok.save_pretrained(path)
        saved=dict(head_meta,head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},
            optimizer_step=total,stage_optimizer_step=update,policy_version='mortal-base-retention-v1',
            training_backend='fresh_base_foundation_terminal_retention',base_only_initialization=False,
            fresh_base_lineage=True,foundation_step=parent_step,
            foundation_identity_sha256=data['parent_identity']['identity_sha256'])
        torch.save(saved,path/'head.pt')
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        try:report,predictions=evaluate_rows(c,path,val,fdata['loss_mass'])
        finally:torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        if update==0:
            anchor=next(p['metrics'] for p in ffit['curve'] if p['update']==parent_step)
            if report['teacher']!=anchor:raise ValueError('Foundation serving baseline does not reproduce exactly')
        state=progress(update,report['joint_ce'],state,c['minimum_observation_updates'],c,len(seen)==len(train))
        adapter=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        lc=any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in adapter.items())
        hc=any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())
        if update and not(lc and hc):raise ValueError('Both LoRA and head must change')
        point=dict(new_update=update,total_step=total,metrics=report,progress=dict(state),
            unique_exposure=len(seen),lora_changed=lc,head_changed=hc,
            validation_path='actual_saved_normal_serving',seconds=time.monotonic()-started)
        curve.append(point);write(root/'predictions'/f'step-{total:06d}-private.json',predictions)
        torch.save(dict(stage_step=update,total_step=total,optimizer=optimizer.state_dict(),seen_ids=sorted(seen),
            torch_rng=rng,cuda_rng=cuda_rng),path/'trainer_state.pt')
        write(out/'training-validation-private.json',point);write(out/'curve-private.json',curve)
        write(root/'selection.json',dict(selected_new_update=state['selected_update'],release_qualified=False))
        print(json.dumps(point),flush=True)

    evaluate(0);reason='registered_max384_new_updates'
    for update,batch in enumerate(schedule,1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=defaultdict(list)
        scale=learning_rate_scale(update,c)
        for group,lr in zip(optimizer.param_groups,(c['lora_learning_rate'],c['head_learning_rate'])):group['lr']=lr*scale
        for index in batch:
            row=train[index];q=row['questions']['action'];rid=row['_meta']['id'];source=q['src']
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.forward(enc[rid])[0]
            target=torch.tensor([q['target'][k] for k in q['criteria']],device=z.device,dtype=torch.float32)
            if abs(float(target.sum())-1)>1e-5:raise ValueError('Target not normalized')
            ce=-(target*z.float().log_softmax(-1)).sum();loss=ce*data['loss_mass'][source]/c['batch_quotas'][source]
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();losses[source].append(float(ce.detach()));seen.add(rid)
        norm=torch.nn.utils.clip_grad_norm_(parameters,c['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
        event=dict(status='training',new_update=update,total_step=parent_step+update,unique_exposure=len(seen),
            training_records=len(train),gradient_norm=float(norm),lr_scale=scale,
            source_counts={k:len(v) for k,v in losses.items()},seconds=time.monotonic()-started)
        write(out/'training-progress-private.json',event)
        with (out/'updates-private.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if update%8==0:print(json.dumps(event),flush=True)
        if update%c['validation_every']==0:
            evaluate(update)
            if state['stop']:reason='four_stale_checks_after128_and_full_exposure';break
    verify(proof);verify({'files':data['files']});verify({'files':data['source_files']})
    if state['selected_update'] is None or len(seen)!=len(train):raise ValueError('No eligible fully exposed checkpoint')
    result=dict(status='fresh_lineage_retention_fit_complete',last_update=parent_step+update,
        last_new_update=update,selected_update=parent_step+state['selected_update'],selected_new_update=state['selected_update'],
        parent_step=parent_step,stop_reason=reason,unique_exposure=len(seen),curve=curve,
        release_qualified=False,raw_publication=False,deployed=False)
    write(out/'fit-result-private.json',result)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def finish(c,result,proof):
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from tools.train_mortal_base import compare
    from tools.audit_mortal_teacher_run import audit_games
    out=ROOT/c['out'];foundation=ROOT/c['foundation_run']
    data=json.loads((out/'data-ready-private.json').read_text());verify({'files':data['source_files']})
    val=read_rows(out/'validation.jsonl');fdata=json.loads((foundation/'data-ready-private.json').read_text())
    points=[]
    for total in sorted({result['selected_update'],result['last_update']}):
        path=out/'training-run/checkpoints'/f'step-{total:06d}'
        report,_=evaluate_rows(c,path,val,fdata['loss_mass'])
        if report!=next(p['metrics'] for p in result['curve'] if p['total_step']==total):
            raise ValueError('Normal saved-model metrics do not reproduce')
        points.append(dict(update=total,metrics=report,identity=bundle_identity(path)))
    audit=dict(status='fresh_lineage_retention_saved_audit_passed',audits=points)
    write(out/'audit-private.json',audit)
    for name,key in (('cached-environment-audit-private.json','native_files'),('input-isolation-audit-private.json','source_files')):
        verify({'files':json.loads((foundation/name).read_text())[key]})
    compare(c,result,audit)
    foundation_rows=read_rows(foundation/'comparison-games-private.jsonl')
    new_rows=read_rows(out/'comparison-games-private.jsonl')
    paired=[dict(r,arm='parent') for r in foundation_rows if r['arm']=='candidate']
    paired += [r for r in new_rows if r['arm']=='candidate']
    seeds=list(range(c['comparison_seed_start'],c['comparison_seed_start']+c['comparison_seed_groups']))
    report=audit_games(paired,seeds)
    report.update(status='retention_vs_fresh_foundation_complete_game_audit_passed',
        no_foundation_policy_rerun=True,parent_step=result['parent_step'],selected_new_update=result['selected_new_update'],
        parent_games_sha256=sha(foundation/'comparison-games-private.jsonl'),
        new_games_sha256=sha(out/'comparison-games-private.jsonl'),raw_publication=False)
    write(out/'foundation-comparison-result-private.json',report)
    verify(proof);verify({'files':data['files']});verify({'files':data['source_files']})
    if bundle_identity(data['parent_checkpoint'])!=data['parent_identity']:raise ValueError('Foundation was modified')
    write(out/'pipeline-exit-private.json',dict(status='complete',exit_code=0,job_id=os.environ['SLURM_JOB_ID'],
        release_qualified=False,automatic_export=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_GPUS') and os.environ.get('SLURM_STEP_GPUS'):
        os.environ['SLURM_JOB_GPUS']=os.environ['SLURM_STEP_GPUS']
    require_gpu();c=configuration(a.config);os.chdir(ROOT);sys.path.insert(0,c['upstream_source'])
    proof=source_registration(c)
    from jevmahjong.endgame.collect import write
    write(ROOT/c['out']/'registered-training-code-private.json',proof)
    finish(c,fit(c,proof),proof)
