"""Continue128-step LoRA on fresh paired64 soft targets; validate normal serving."""
import argparse
from collections import defaultdict
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def fresh_metrics(rows,predictions):
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    from jevmahjong.endgame.reference import restrict_distribution
    groups=defaultdict(list)
    for r in rows:
        meta=r['_meta'];keys=fourth_chase_action_keys(r);p=predictions[meta['id']]
        chosen=max(keys,key=p.get);restricted=restrict_distribution(p,keys);q=meta['fresh_top2']
        bucket='fourth' if meta['source_initial_placement']==4 else 'third' if meta['source_initial_placement']==3 else 'hold'
        groups[bucket].append(dict(top2=q[chosen],soft_top2=sum(restricted[k]*q[k] for k in q),
            fourth=1-meta['fresh_escape'][chosen]))
    if set(groups)!={'fourth','third','hold'}:raise ValueError('Fresh monitoring must keep pursuit and holding')
    return {k:dict(positions=len(values),**{m:sum(v[m] for v in values)/len(values) for m in values[0]}) for k,values in groups.items()}


def screen(metrics,initial):
    old,new=metrics['old'],metrics['fresh'];base=initial['old'];nb=initial['fresh']
    flags=dict(primary_gain=metrics['primary']>initial['primary']+1e-12,
        old_fourth_no_material_drop=old['fourth']['top2']>=base['fourth']['top2']-.01-1e-12,
        new_fourth_no_drop=new['fourth']['top2']>=nb['fourth']['top2']-1e-12,
        third_preserved=old['legacy']['third_pursuit']['candidate']>=base['legacy']['third_pursuit']['candidate']-1e-12,
        ordinary_third_preserved=old['legacy']['utility_by_stratum']['third']>=base['legacy']['utility_by_stratum']['third']-1e-12,
        holding_preserved=old['legacy']['utility_by_stratum']['hold']>=base['legacy']['utility_by_stratum']['hold']-.02-1e-12,
        fresh_third_preserved=new['third']['top2']>=nb['third']['top2']-1/32-1e-12,
        fresh_hold_preserved=new['hold']['top2']>=nb['hold']['top2']-1/32-1e-12,
        no_severe_replay_regression=old['legacy']['original_expert_ce']<=base['legacy']['original_expert_ce']+.5)
    return dict(flags,research_eligible=all(flags.values()),fourth_escape_and_kl_report_only=True,release_qualified=False)


def progress_state(update,metrics,flags,previous,c):
    if not all(math.isfinite(metrics[k]) for k in ('primary','soft_primary')):raise ValueError('Nonfinite selection metric')
    if update==0:return dict(selected_update=None,selected_primary=None,best_hard=metrics['primary'],
                            best_soft=metrics['soft_primary'],stale=0,stop=False)
    p=dict(previous);selected=flags['research_eligible'] and (p['selected_primary'] is None or metrics['primary']>p['selected_primary']+1e-12)
    if selected:p.update(selected_update=update,selected_primary=metrics['primary'])
    advancing=selected or metrics['primary']>p['best_hard']+1e-12 or metrics['soft_primary']>p['best_soft']+1e-5
    p['best_hard']=max(p['best_hard'],metrics['primary']);p['best_soft']=max(p['best_soft'],metrics['soft_primary'])
    p['stale']=0 if update<=c['observation_new_updates'] or advancing else p['stale']+1
    p['stop']=p['stale']>=c['early_stop_patience'];return p


def validate_training_config(c):
    from tools.build_targeted_endgame import validate_config
    validate_config(c)
    expected_mass={'old_replay':.30,'old_leading':.15,'old_fourth':.10,'old_third':.025,'old_hold':.025,
        'new_discard':.13,'new_call':.12,'new_win':.05,'new_third':.05,'new_hold':.05}
    if (c['loss_mass']!=expected_mass or c['lora_learning_rate']!=1e-6 or c['head_learning_rate']!=5e-6
            or c['new_optimizer'] is not True or c['warmup_steps']!=8 or c['max_new_updates']!=192
            or c['validation_every_steps']!=16 or c['observation_new_updates']!=64 or c['early_stop_patience']!=3):
        raise ValueError('Unregistered fresh-data fitting schedule')


def fit(c,registration):
    validate_training_config(c)
    from tools.build_targeted_endgame import old_assets,verify_registration
    from tools.train_endgame_fourth_chase import verify
    from tools.continue_endgame_fourth_lora import normal_predictions
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import checksum,read_jsonl
    from jevmahjong.endgame.targeted_refresh import balanced_batches
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_guards,chase_training_metrics
    out=Path(c['out']);verify_registration(c,registration)
    if (out/'training-run').exists() or (out/'training-protocol-private.json').exists():raise ValueError('Existing fit; inspect instead of restarting')
    receipt=json.loads((out/'data-ready-private.json').read_text())
    if (receipt['status']!='fresh64_verified_training_data_ready' or receipt['train']!=874
            or not receipt['validation_unchanged'] or receipt['fresh_train']!=48 or receipt['fresh_monitor']!=8):
        raise ValueError('Verified new data are not ready')
    for name,sha in receipt['data_sha256'].items():
        if checksum(out/(name+'.jsonl'))!=sha:raise ValueError('Prepared data changed')
    train=read_jsonl(out/'train.jsonl');validation=read_jsonl(out/'validation.jsonl');monitor=read_jsonl(out/'monitor.jsonl')
    if {r['_meta']['id'] for r in train}&{r['_meta']['id'] for r in validation+monitor}:raise ValueError('Training/monitoring identity leakage')
    ready,formal,oldtrain,oldval,oldfourth,ck=old_assets(c);parent=ck[128]
    batches=balanced_batches(train,c['batch_quotas'],192)
    if len({i for b in batches[:64] for i in b})!=874:raise ValueError('Observation window misses a positive training record')
    proof=dict(registration,data_receipt_sha256=checksum(out/'data-ready-private.json'),
        data_sha256=receipt['data_sha256'],batch_schedule=batches,parent_step=128,
        parent_checkpoint=str(parent),new_optimizer=True,new_networks=0)
    write(out/'training-protocol-private.json',proof)
    sys.path.insert(0,ready['source_config']['upstream_source'])
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from jevany.model import DecisionModel,load_preprocessor
    from jevany.checkpoint import Checkpoint
    from jevany.data import materialize
    torch.manual_seed(2026111920)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=Checkpoint(str(parent));meta=checkpoint.meta
    tok=load_preprocessor(str(parent),multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter)
    model=DecisionModel(ready['source_config']['base_load_path'],tok,'cuda',lora=16,head_dim=meta.head_dim,
        head_residual_dim=meta.head_residual_dim,lora_targets='all',lora_dropout=.05,option_isolation=meta.option_isolation,
        special_embeddings=meta.special_embeddings,multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter,
        branch_mode=meta.branch_mode,dtype=torch.bfloat16,decision_mode=meta.decision_mode)
    for cfg in model.lm.peft_config.values():cfg.base_model_name_or_path=meta.base
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.lm.config.use_cache=False
    checkpoint.warm_start(model,meta)
    adapter0=load_file(str(parent/'adapter_model.safetensors'),device='cpu')
    head_meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True);head0=head_meta['head']
    actual=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
    if (set(actual)!=set(adapter0) or any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in actual.items())
            or any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())):
        raise ValueError('Initialization differs from128-step weights')
    hp=list(model.head.parameters());hid={id(p) for p in hp};named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(id(p) not in hid and 'lora_' not in n for n,p in named):raise ValueError('Base weights must remain frozen')
    lp=[p for _,p in named if id(p) not in hid];parameters=[p for _,p in named]
    optimizer=torch.optim.AdamW([{'params':lp,'lr':1e-6},{'params':hp,'lr':5e-6}],weight_decay=.001)
    write(out/'training-initialization-private.json',dict(exact_parent_step128_tensors=True,new_optimizer=True,
        parent384_not_used=True,trainable_lora=sum(p.numel() for p in lp),trainable_head=sum(p.numel() for p in hp),
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    enc={}
    for row in train:
        e=model.encode(tok,materialize(row),max_state=4096,max_branch=8192,strict=True)
        if e.get('state_truncated') or len(e['ids'])>8192:raise ValueError('No token truncation allowed')
        enc[row['_meta']['id']]=e
    source=Path(c['prepared_data_run']);reference=json.loads((source/'reference-targets-private.json').read_text())['rows']
    fresh_train=[r for r in train if r['_meta'].get('source')=='targeted_fresh64']
    old_fourth=[r for r in train if r['questions']['action']['src']=='old_fourth']
    if len(fresh_train)!=48 or len(old_fourth)!=149:raise ValueError('Diagnostic cohort changed')
    root=out/'training-run';curve=[];state=None;initial=None;seen=set();started=time.monotonic()

    def evaluate(update):
        nonlocal state,initial
        total=128+update;path=root/'checkpoints'/f'step-{total:06d}';path.mkdir(parents=True,exist_ok=False)
        model.eval();model.lm.save_pretrained(path,save_embedding_layers=False);tok.save_pretrained(path)
        saved=dict(head_meta,head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},
            optimizer_step=total,stage_optimizer_step=update,training_backend='targeted64_new_mix_from128',
            policy_version='endgame-fourth-chase-v1')
        torch.save(saved,path/'head.pt')
        # Loading temporary normal inference models must not perturb training RNG.
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        try:p=normal_predictions(path,validation+monitor+old_fourth+fresh_train)
        finally:torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        old=chase_validation(validation,{r['_meta']['id']:p[r['_meta']['id']] for r in validation},reference)
        fresh=fresh_metrics(monitor,p)
        metrics=dict(old=old,fresh=fresh,primary=.5*(old['fourth']['top2']+fresh['fourth']['top2']),
                     soft_primary=.5*(old['fourth']['soft_top2']+fresh['fourth']['soft_top2']))
        if update==0:
            if abs(old['fourth']['top2']-.22549019607843138)>1e-12:raise ValueError('Initial saved128 normal path changed')
            initial=copy.deepcopy(metrics);write(out/'training-anchor-private.json',initial)
        flags=screen(metrics,initial);state=progress_state(update,metrics,flags,state,c)
        adapter=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        changed_adapter=any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in adapter.items())
        changed_head=any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())
        if update and not (changed_adapter and changed_head):raise ValueError('Actual LoRA and head updates required')
        diagnostics=dict(old_fourth=chase_training_metrics(old_fourth,p,reference),fresh=fresh_metrics(fresh_train,p))
        point=dict(new_update=update,total_step=total,metrics=metrics,screen=flags,progress=state,
            unique_exposure=len(seen),training_diagnostics=diagnostics,lora_changed=changed_adapter,head_changed=changed_head,
            validation_path='actual_saved_normal_serving',seconds=time.monotonic()-started)
        curve.append(point);write(root/'predictions'/f'step-{total:06d}-private.json',p)
        torch.save(dict(step=update,total_step=total,optimizer=optimizer.state_dict(),seen_ids=sorted(seen),
            torch_rng=cpu_rng,cuda_rng=cuda_rng),path/'trainer_state.pt')
        write(root/'selection.json',dict(selected_new_update=state['selected_update'],release_qualified=False))
        write(out/'training-validation-private.json',point);write(out/'training-curve-private.json',curve)
        print(json.dumps(dict(new_update=update,total_step=total,primary=metrics['primary'],old_fourth=old['fourth'],
                             fresh=fresh,screen=flags,progress=state,unique_exposure=len(seen))),flush=True)
        return point

    evaluate(0);stop_reason='registered_max192_new_updates'
    for update,batch in enumerate(batches,1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=defaultdict(list)
        scale=min(1.,update/8);optimizer.param_groups[0]['lr']=1e-6*scale;optimizer.param_groups[1]['lr']=5e-6*scale
        for index in batch:
            row=train[index];q=row['questions']['action'];src=q['src'];rid=row['_meta']['id']
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.forward(enc[rid])[0]
            target=torch.tensor([q['target'][k] for k in q['criteria']],device=z.device,dtype=torch.float32);target/=target.sum()
            ce=-(target*z.float().log_softmax(-1)).sum();value=ce*c['loss_mass'][src]/c['batch_quotas'][src]
            if not torch.isfinite(value):raise ValueError('Nonfinite training loss')
            value.backward();losses[src].append(float(ce.detach()));seen.add(rid)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        event=dict(new_update=update,total_step=128+update,unique_exposure=len(seen),gradient_norm=float(norm),
            source_counts={k:len(v) for k,v in losses.items()},source_ce={k:sum(v)/len(v) for k,v in losses.items()},
            lora_lr=optimizer.param_groups[0]['lr'],head_lr=optimizer.param_groups[1]['lr'],seconds=time.monotonic()-started)
        write(out/'training-progress-private.json',event)
        with (out/'training-updates-private.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        if update%16==0:
            point=evaluate(update)
            if not point['screen']['no_severe_replay_regression']:stop_reason='severe_old_replay_ce_regression';break
            if state['stop']:stop_reason='three_stale_validations_after64_new_updates';break
    verify(registration)
    for name,sha in receipt['data_sha256'].items():
        if checksum(out/(name+'.jsonl'))!=sha:raise ValueError('Training/monitoring data changed')
    result=dict(status='targeted64_from128_fit_complete',last_new_update=update,last_total_step=128+update,
        selected_new_update=state['selected_update'],stop_reason=stop_reason,curve=curve,old_weights_start128=True,
        new_optimizer=True,actual_normal_inference_validation=True,release_qualified=False,
        independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'training-result-private.json',result);write(out/'training-status-private.json',{k:v for k,v in result.items() if k!='curve'})


def audit(c,reg):
    from tools.build_targeted_endgame import verify_registration,old_assets
    from tools.continue_endgame_fourth_lora import normal_predictions
    from jevmahjong.endgame.prepare import read_jsonl,checksum
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.fourth_chase_training import chase_validation
    validate_training_config(c);verify_registration(c,reg);out=Path(c['out'])
    result=json.loads((out/'training-result-private.json').read_text())
    ready,formal,train,val,fourth,ck=old_assets(c);sys.path.insert(0,ready['source_config']['upstream_source'])
    reference=json.loads((Path(c['prepared_data_run'])/'reference-targets-private.json').read_text())['rows']
    monitor=read_jsonl(out/'monitor.jsonl');anchor=json.loads((out/'training-anchor-private.json').read_text());reports=[]
    selected=result['selected_new_update']
    for update in sorted({result['last_new_update']}|({selected} if selected is not None else set())):
        path=out/'training-run/checkpoints'/f'step-{128+update:06d}';p=normal_predictions(path,val+monitor)
        old=chase_validation(val,{r['_meta']['id']:p[r['_meta']['id']] for r in val},reference);new=fresh_metrics(monitor,p)
        metrics=dict(old=old,fresh=new,primary=.5*(old['fourth']['top2']+new['fourth']['top2']),
                     soft_primary=.5*(old['fourth']['soft_top2']+new['fourth']['soft_top2']))
        logged=next(x for x in result['curve'] if x['new_update']==update)
        if metrics!=logged['metrics']:raise ValueError('Reloaded saved validation decisions differ')
        reports.append(dict(new_update=update,total_step=128+update,metrics=metrics,screen=screen(metrics,anchor),
            eligible_selected_for_consumed_development=update==selected and screen(metrics,anchor)['research_eligible']))
    verify_registration(c,reg)
    summary=dict(status='targeted64_saved_model_audit_complete',audits=reports,
        eligible_for_consumed_development=any(x['eligible_selected_for_consumed_development'] for x in reports),
        release_qualified=False,independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'training-audit-private.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fit','audit'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):raise RuntimeError('Scheduled GPU allocation required')
    from tools.prepare_native_context_replay import allocation_metadata
    d=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps(dict(node=socket.gethostname(),allocation=allocation_metadata(d,socket.gethostname()))),flush=True)
    c=json.loads(a.config.read_text());reg=json.loads(a.registration.read_text());{'fit':fit,'audit':audit}[a.stage](c,reg)
