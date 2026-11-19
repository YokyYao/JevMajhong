"""Resume the audited128-step LoRA optimizer, sampler and RNG through384.

Research screening uses terminal objectives. Historical KL/CE guards remain
reported unchanged; no screening result authorizes independent tests or release.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_config(c):
    expected={'schema':'endgame-fourth-lora-exact-extension-v1','out':'runs/endgame-fourth-lora-extension-v1',
        'prepared_data_run':'runs/endgame-fourth-top2-chase-v1','parent_run':'runs/endgame-fourth-lora-balanced-v1',
        'resume_step':128,'max_total_steps':384,'validation_every_steps':32,'observation_end_step':192,
        'early_stop_patience':3,'soft_progress_min_delta':1e-5,'lora_learning_rate':2e-6,'head_learning_rate':1e-5,
        'weight_decay':.001,'gradient_clip_norm':1.,'severe_replay_ce_increase_stop':.5,
        'resume_optimizer_and_rng':True,'repeat_warmup':False,'new_training_records':0,'changed_loss_masses':False,
        'research_selection':'hard_top2_above_best_audited_parent_and_terminal_third_holding',
        'reference_kl_role':'report_only_for_research_screening','formal_guard_protocol':'unchanged_report',
        'fourth_escape_constraint':'report_only','training_registered':False,
        'independent_confirmation':False,'export_registered':False,'raw_publication':False}
    if c!=expected:raise ValueError('Only the registered chronological128→384 continuation is enabled')


def research_screen(step,metrics,anchor):
    """A research candidate can change preferences, but must improve outcomes."""
    fourth=metrics['fourth'];old=metrics['legacy'];formal=anchor['formal']
    values=[fourth['top2'],fourth['fourth'],fourth['soft_top2'],old['third_pursuit']['candidate'],
            old['utility_by_stratum']['third'],old['utility_by_stratum']['hold'],old['original_expert_ce']]
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite research metric')
    flags=dict(new_extension_step=step>128,
        hard_top2_gain=fourth['top2']>anchor['best_audited_parent_top2']+1e-12,
        ordinary_third_preserved=old['utility_by_stratum']['third']>=formal['legacy']['utility_by_stratum']['third']-1e-12,
        third_pursuit_preserved=old['third_pursuit']['candidate']>=max(formal['legacy']['third_pursuit']['candidate'],
            formal['legacy']['third_pursuit']['original_policy'])-1e-12,
        holding_preserved=old['utility_by_stratum']['hold']>=formal['legacy']['utility_by_stratum']['hold']-.02-1e-12,
        no_severe_replay_regression=old['original_expert_ce']<=formal['legacy']['original_expert_ce']+.5)
    return dict(flags,eligible_for_consumed_research=all(flags.values()),
                preference_kl_and_fourth_escape_are_report_only=True,release_qualified=False)


def progress_state(step,metrics,screen,previous,anchor,c):
    hard=metrics['fourth']['top2'];soft=metrics['fourth']['soft_top2'];tail=metrics['fourth']['fourth']
    if step==128:
        return dict(selected_research_step=None,selected_hard=None,selected_tail=None,
            best_hard=max(anchor['best_audited_parent_top2'],hard),best_soft=max(anchor['parent_best_live_soft'],soft),
            stale=0,stop=False)
    state=dict(previous)
    selected=screen['eligible_for_consumed_research'] and (state['selected_hard'] is None
        or hard>state['selected_hard']+1e-12 or abs(hard-state['selected_hard'])<=1e-12 and tail<state['selected_tail']-1e-12)
    if selected:state.update(selected_research_step=step,selected_hard=hard,selected_tail=tail)
    advancing=selected or hard>state['best_hard']+1e-12 or soft>state['best_soft']+c['soft_progress_min_delta']
    state['best_hard']=max(state['best_hard'],hard);state['best_soft']=max(state['best_soft'],soft)
    state['stale']=0 if step<=c['observation_end_step'] or advancing else state['stale']+1
    state['stop']=state['stale']>=c['early_stop_patience'];return state


def validate_resume_state(state,seen_ids):
    if state['step']!=128 or set(state['seen_ids'])!=set(seen_ids) or len(seen_ids)!=826:
        raise ValueError('Resume must retain all826 examples and the actual128-step state')
    opt=state['optimizer']
    if ([g['lr'] for g in opt['param_groups']]!=[2e-6,1e-5]
            or any(g['weight_decay']!=.001 for g in opt['param_groups'])
            or not opt['state'] or any(float(v['step'])!=128 for v in opt['state'].values())
            or 'torch_rng' not in state or 'cuda_rng' not in state):
        raise ValueError('Optimizer moments, learning rates or RNG are not a true continuation')


def continuation_coverage(schedule,positive_count=826,resume_step=128,observation_end=192):
    historical={i for batch in schedule[:observation_end] for i in batch}
    extension={i for batch in schedule[resume_step:observation_end] for i in batch}
    if historical!=set(range(positive_count)):
        raise ValueError('Cumulative training history must cover every positive row')
    # A sliding window can span two differently shuffled source cycles and
    # omit a row even when each complete cycle is without replacement.
    # Preserve the true sampler position and report new-window exposure.
    return dict(cumulative_unique=len(historical),extension_unique=len(extension))


def assets(c):
    from tools.train_endgame_fourth_adamw import validate_assets
    from tools.train_endgame_fourth_chase import verify
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.prepare import checksum
    ready,formal,train,val,fourth=validate_assets(c);parent=Path(c['parent_run'])
    verify(json.loads((parent/'registered-code-private.json').read_text()))
    jobs=json.loads((parent/'jobs-private.json').read_text())
    for stage in ('fit','postfit'):
        if json.loads((parent/f'{stage}-job-{jobs[stage]}-exit.json').read_text())['exit_code']!=0:
            raise ValueError('Parent fit and normal-serving audit must have succeeded')
    status=json.loads((parent/'status-private.json').read_text());audit=json.loads((parent/'postfit-summary-private.json').read_text())
    if (status['last_step']!=128 or audit['last_step']!=128
            or audit['status']!='fourth_lora_normal_service_audit_complete'
            or not audit['actual_existing_lora_and_head_updates_verified']):raise ValueError('Parent final128 is not audited')
    checkpoints={step:parent/'training-run/checkpoints'/f'step-{step:06d}' for step in (96,128)}
    return ready,formal,train,val,fourth,checkpoints


def normal_predictions(checkpoint,rows):
    from jevmahjong.policy.routing import load_model
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.futures import bundle_identity
    from jevany.api import validate_distribution
    import torch
    before=bundle_identity(checkpoint);client=load_model(checkpoint);client.runtime.model.eval().requires_grad_(False)
    if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Inference prefix cache must be disabled')
    p={}
    for row in rows:
        keys=list(row['questions']['action']['criteria']);answer=client.system_one(**inference_request(row))
        values,_=validate_distribution(answer['answers']['action']['probabilities'],keys)
        p[row['_meta']['id']]=dict(zip(keys,values))
    if bundle_identity(checkpoint)!=before:raise ValueError('Serving changed model files')
    del client;gc.collect();torch.cuda.empty_cache();return p


def prepare(c,registration):
    validate_config(c)
    from tools.train_endgame_fourth_chase import verify
    from tools.train_endgame_fourth_lora_balanced import balanced_schedule
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_training_metrics
    out=Path(c['out'])
    if any((out/n).exists() for n in ('protocol-private.json','ready-private.json','training-run')):
        raise ValueError('Inspect an existing extension rather than silently restarting')
    if registration['configuration']!=c or registration['ci']['conclusion']!='success':raise ValueError('Public source CI incomplete')
    verify(registration);ready,formal,train,val,fourth,checkpoints=assets(c)
    parent=Path(c['parent_run']);source=Path(c['prepared_data_run'])
    parent_protocol=json.loads((parent/'protocol-private.json').read_text())
    schedule=balanced_schedule(train,384)
    if schedule[:128]!=parent_protocol['batch_schedule']:raise ValueError('The first128 sampler batches changed')
    coverage=continuation_coverage(schedule)
    paths=[parent/n for n in ('protocol-private.json','registered-code-private.json','status-private.json',
        'postfit-summary-private.json','curve-private.json','selection-anchor-private.json')]
    paths.extend(source/n for n in ('ready-private.json','train.jsonl','validation.jsonl','reference-targets-private.json'))
    paths.extend(checkpoints[s]/'trainer_state.pt' for s in (96,128))
    proof=dict(registration,source_sha256=dict(registration['source_sha256'],**{str(p.resolve()):checksum(p) for p in paths}),
        formal_identity=ready['parent_identity'],parent_identities={str(s):bundle_identity(p) for s,p in checkpoints.items()},
        chronological_resume_step=128,batch_schedule=schedule)
    write(out/'protocol-private.json',proof)
    sys.path.insert(0,ready['source_config']['upstream_source'])
    import torch
    state=torch.load(checkpoints[128]/'trainer_state.pt',map_location='cpu',weights_only=True)
    validate_resume_state(state,[r['_meta']['id'] for r in train]);del state
    reference=json.loads((source/'reference-targets-private.json').read_text())['rows'];reports=[]
    for step,path in checkpoints.items():
        p=normal_predictions(path,val+fourth)
        metric=chase_validation(val,{r['_meta']['id']:p[r['_meta']['id']] for r in val},reference)
        reports.append(dict(step=step,metrics=metric,training149=chase_training_metrics(fourth,
            {r['_meta']['id']:p[r['_meta']['id']] for r in fourth},reference),normal_service_queries=335))
        write(out/f'parent-step-{step:06d}-service-private.json',p)
    anchor=dict(formal=ready['initial_validation'],
        best_audited_parent_top2=max([ready['initial_validation']['fourth']['top2']]+[p['metrics']['fourth']['top2'] for p in reports]),
        parent_best_live_soft=max(p['metrics']['fourth']['soft_top2'] for p in json.loads((parent/'curve-private.json').read_text())))
    verify(proof)
    result=dict(status='fourth_lora_extension_ready',anchor=anchor,parent_service_reports=reports,
        resume_optimizer_step=128,first128_sampler_batches_identical=True,observation_coverage=coverage,
        configuration=c,proof_sha256=checksum(out/'protocol-private.json'))
    write(out/'ready-private.json',result);print(json.dumps(result),flush=True)


def fit(c,registration):
    validate_config(c)
    from tools.train_endgame_fourth_chase import verify
    from tools.train_endgame_fourth_lora_balanced import record_loss_coefficient
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_guards,chase_training_metrics
    from jevmahjong.endgame.prepare import checksum
    out=Path(c['out']);prepared=json.loads((out/'ready-private.json').read_text())
    proof=json.loads((out/'protocol-private.json').read_text());verify(proof)
    if (prepared['configuration']!=c or prepared['proof_sha256']!=checksum(out/'protocol-private.json')
            or (out/'training-run').exists()):raise ValueError('Unverified preparation or existing training')
    ready,formal,train,val,fourth,checkpoints=assets(c);resume=checkpoints[128]
    sys.path.insert(0,ready['source_config']['upstream_source'])
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from jevany.model import DecisionModel,load_preprocessor
    from jevany.checkpoint import Checkpoint
    from jevany.data import materialize
    ck=Checkpoint(str(resume));meta=ck.meta;base=ready['source_config']['base_load_path']
    tok=load_preprocessor(str(resume),multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter)
    model=DecisionModel(base,tok,'cuda',lora=16,head_dim=meta.head_dim,head_residual_dim=meta.head_residual_dim,
        lora_targets='all',lora_dropout=.05,option_isolation=meta.option_isolation,special_embeddings=meta.special_embeddings,
        multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter,branch_mode=meta.branch_mode,
        dtype=torch.bfloat16,decision_mode=meta.decision_mode)
    for x in model.lm.peft_config.values():x.base_model_name_or_path=meta.base
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.lm.config.use_cache=False
    ck.warm_start(model,meta)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter0=load_file(str(resume/'adapter_model.safetensors'),device='cpu')
    head_meta=torch.load(resume/'head.pt',map_location='cpu',weights_only=True);head0=head_meta['head']
    actual=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
    if (set(actual)!=set(adapter0) or any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in actual.items())
            or any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())):
        raise ValueError('Actual initialization is not the exact chronological128 checkpoint')
    hp=list(model.head.parameters());hid={id(p) for p in hp};named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any('lora_' not in n and id(p) not in hid for n,p in named):raise ValueError('Base model must stay frozen')
    lp=[p for _,p in named if id(p) not in hid];parameters=[p for _,p in named]
    optimizer=torch.optim.AdamW([{'params':lp,'lr':2e-6},{'params':hp,'lr':1e-5}],weight_decay=.001)
    state=torch.load(resume/'trainer_state.pt',map_location='cpu',weights_only=True)
    validate_resume_state(state,[r['_meta']['id'] for r in train]);optimizer.load_state_dict(state['optimizer'])
    for p,st in optimizer.state.items():
        if st['exp_avg'].shape!=p.shape or st['exp_avg_sq'].shape!=p.shape:raise ValueError('Optimizer moments do not match parameters')
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    seen=set(state['seen_ids']);del state
    write(out/'initialization-private.json',dict(exact_parent_step128_tensors=True,optimizer_moments_restored=True,
        optimizer_step=128,torch_and_cuda_rng_restored=True,first128_sampler_batches_identical=True,
        repeat_warmup=False,lora_lr=2e-6,head_lr=1e-5,trainable_lora=sum(p.numel() for p in lp),
        trainable_head=sum(p.numel() for p in hp),node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    enc={}
    for row in train+val:
        value=model.encode(tok,materialize(row),max_state=4096,max_branch=8192,strict=True)
        if value.get('state_truncated') or len(value['ids'])>8192:raise ValueError('No truncation or rejection is allowed')
        enc[row['_meta']['id']]=value
    reference=json.loads((Path(c['prepared_data_run'])/'reference-targets-private.json').read_text())['rows']
    masses={k:v for k,v in ready['loss_recipe']['mass'].items() if v>0}
    anchor=prepared['anchor'];root=out/'training-run';curve=[];progress=None;extension_seen=set();started=time.monotonic()
    def predictions(rows):
        model.eval();result={}
        with torch.no_grad():
            for row in rows:
                rid=row['_meta']['id'];z=model.forward(enc[rid])[0].float();p=z.softmax(-1).cpu().tolist()
                if not all(math.isfinite(x) for x in p):raise ValueError('Nonfinite prediction')
                result[rid]=dict(zip(row['questions']['action']['criteria'],p))
        return result
    def evaluate(step):
        nonlocal progress
        p=predictions(val);metrics=chase_validation(val,p,reference);screen=research_screen(step,metrics,anchor)
        old_flags=chase_guards(step,metrics,anchor['formal'],ready['source_config'],exposure_met=True)
        progress=progress_state(step,metrics,screen,progress,anchor,c)
        learning=chase_training_metrics(fourth,predictions(fourth),reference)
        path=root/'checkpoints'/f'step-{step:06d}'
        if step==128:shutil.copytree(resume,path)
        else:
            path.mkdir(parents=True,exist_ok=False);model.lm.save_pretrained(path,save_embedding_layers=False);tok.save_pretrained(path)
            saved=dict(head_meta,head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},
                optimizer_step=step,training_backend='exact128_optimizer_lora_extension',policy_version='endgame-fourth-chase-v1')
            torch.save(saved,path/'head.pt')
            torch.save(dict(step=step,optimizer=optimizer.state_dict(),seen_ids=sorted(seen),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path/'trainer_state.pt')
        current=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        changed_adapter=any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in current.items())
        changed_head=any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())
        if step>128 and not (changed_adapter and changed_head):raise ValueError('Both existing LoRA and head must actually continue updating')
        point=dict(step=step,metrics=metrics,research_screen=screen,historical_formal_guards=old_flags,
            progress=progress,training149=learning,extension_unique_exposure=len(extension_seen),
            lora_changed_since128=changed_adapter,head_changed_since128=changed_head,seconds=time.monotonic()-started)
        curve.append(point);write(root/'predictions'/f'step-{step:06d}-private.json',p)
        write(root/'selection.json',dict(research_step=progress['selected_research_step'],release_qualified=False))
        write(out/'progress-private.json',point);write(out/'curve-private.json',curve)
        print(json.dumps({'validation_step':step,'fourth':metrics['fourth'],'research_screen':screen,'progress':progress,
                          'third_kl_report':metrics['native_third_kl'],'training149':learning}),flush=True)
        return point
    evaluate(128);stop_reason='registered_total384_steps'
    for step in range(129,385):
        model.train();optimizer.zero_grad(set_to_none=True);losses=defaultdict(list)
        for index in proof['batch_schedule'][step-1]:
            row=train[index];rid=row['_meta']['id'];q=row['questions']['action'];source=q['src']
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.forward(enc[rid])[0]
            target=torch.tensor([q['target'][k] for k in q['criteria']],device=z.device,dtype=torch.float32);target/=target.sum()
            ce=-(target*z.float().log_softmax(-1)).sum();loss=ce*record_loss_coefficient(source,masses)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();losses[source].append(float(ce.detach()));extension_seen.add(rid)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        event=dict(step=step,new_updates=step-128,extension_unique_exposure=len(extension_seen),
            source_counts={k:len(v) for k,v in losses.items()},source_ce={k:sum(v)/len(v) for k,v in losses.items()},
            gradient_norm=float(norm),lora_lr=optimizer.param_groups[0]['lr'],head_lr=optimizer.param_groups[1]['lr'],seconds=time.monotonic()-started)
        write(out/'training-progress-private.json',event)
        with (out/'updates-private.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        if step%32==0:
            point=evaluate(step)
            if not point['research_screen']['no_severe_replay_regression']:stop_reason='severe_replay_ce_regression';break
            if progress['stop']:stop_reason='three_stale_checks_after_total192';break
    verify(proof);assets(c)
    result=dict(status='fourth_lora_extension_fit_complete',last_step=step,new_updates=step-128,
        selected_research_step=progress['selected_research_step'],stop_reason=stop_reason,curve=curve,
        release_qualified=False,new_training_records=0,new_simulations=0,independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'fit-result-private.json',result);write(out/'status-private.json',{k:v for k,v in result.items() if k!='curve'})


def audit(c,registration):
    from tools.train_endgame_fourth_chase import verify
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_guards,chase_training_metrics
    out=Path(c['out']);proof=json.loads((out/'protocol-private.json').read_text());verify(proof)
    if (out/'postfit-summary-private.json').exists():raise ValueError('Existing audit')
    ready,formal,train,val,fourth,checkpoints=assets(c)
    sys.path.insert(0,ready['source_config']['upstream_source'])
    result=json.loads((out/'fit-result-private.json').read_text());anchor=json.loads((out/'ready-private.json').read_text())['anchor']
    reference=json.loads((Path(c['prepared_data_run'])/'reference-targets-private.json').read_text())['rows']
    reports=[];selected=result['selected_research_step']
    for step in sorted({result['last_step']}|({selected} if selected is not None else set())):
        p=normal_predictions(out/'training-run/checkpoints'/f'step-{step:06d}',val+fourth)
        metrics=chase_validation(val,{r['_meta']['id']:p[r['_meta']['id']] for r in val},reference)
        screen=research_screen(step,metrics,anchor)
        reports.append(dict(step=step,normal_service_queries=335,metrics=metrics,research_screen=screen,
            historical_formal_guards=chase_guards(step,metrics,anchor['formal'],ready['source_config'],exposure_met=True),
            training149=chase_training_metrics(fourth,{r['_meta']['id']:p[r['_meta']['id']] for r in fourth},reference),
            eligible_selected_for_consumed_research=step==selected and screen['eligible_for_consumed_research']))
        write(out/f'normal-service-step-{step:06d}-private.json',p)
    verify(proof);assets(c)
    summary=dict(status='fourth_lora_extension_normal_service_audit_complete',last_step=result['last_step'],
        selected_research_step=selected,audits=reports,
        eligible_for_consumed_research=any(r['eligible_selected_for_consumed_research'] for r in reports),
        release_qualified=False,new_simulations=0,independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'postfit-summary-private.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','fit','audit'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Continuation requires a scheduler-allocated GPU node')
    from tools.prepare_native_context_replay import allocation_metadata
    d=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(),'allocation':allocation_metadata(d,socket.gethostname())}),flush=True)
    c=json.loads(a.config.read_text());validate_config(c)
    {'prepare':prepare,'fit':fit,'audit':audit}[a.command](c,json.loads(a.registration.read_text()))
