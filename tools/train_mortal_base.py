"""Distill Mortal into a new Jev LoRA and pointer head initialized from base.

No Mahjong-trained weights or replay targets are loaded by fit(). Run only in
a bounded scheduled GPU allocation. Raw data and research checkpoints stay local.
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_mortal_teacher import sha, verify, registration, require_gpu


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def validate_reference_protocol(protocol,seeds,parent_id):
    if (protocol['arms']['parent']['identity_sha256']!=parent_id or protocol['seeds']!=seeds
            or protocol['profiles']!=['calling','mortal'] or protocol['seats']!=4
            or not protocol['same_formatter_and_guard']):
        raise ValueError('Cached parent protocol/identity differs')


def cached_parent(c):
    donor=ROOT/c['teacher_source'];reference=ROOT/c['reference_games']
    if sha(reference)!=c['reference_games_sha256']:raise ValueError('Cached comparison changed')
    previous=json.loads((donor/'independent-audit-private.json').read_text())
    protocol=json.loads((donor/'comparison-protocol-private.json').read_text())
    parent_id='db153c5a445a9350a41e20b73f459956f5484df96731e915dd042b5e24d9846b'
    if (previous['status']!='independent_accounting_and_statistics_passed'
            or previous['comparison_games_sha256']!=c['reference_games_sha256']):
        raise ValueError('Cached parent not independently audited')
    seeds=list(range(c['comparison_seed_start'],c['comparison_seed_start']+c['comparison_seed_groups']))
    validate_reference_protocol(protocol,seeds,parent_id)
    verify(json.loads((donor/'registered-training-code-private.json').read_text()))
    rows=[r for r in read_rows(reference) if r['arm']=='parent']
    if len(rows)!=c['cached_parent_games']:raise ValueError('Incomplete cached parent')
    return rows,protocol['arms']['parent'],seeds


def learning_rate_scale(update,c):
    warm=c['warmup_updates']
    if update<=warm:return update/warm
    fraction=(update-warm)/(c['max_updates']-warm)
    floor=c['cosine_final_fraction']
    return floor+(1-floor)*.5*(1+math.cos(math.pi*fraction))


def progress(update,ce,previous,observation,c,fully_exposed):
    if not math.isfinite(ce):raise ValueError('Nonfinite validation CE')
    p=dict(previous or dict(best_ce=ce,selected_update=None,selected_ce=None,stale=0,stop=False))
    p['best_ce']=min(p['best_ce'],ce)
    if update<observation or not fully_exposed:return p
    old=p['selected_ce']
    significant=old is None or ce<old*(1-c['progress_relative_delta'])
    if old is None or ce<old:p.update(selected_update=update,selected_ce=ce)
    p['stale']=0 if significant else p['stale']+1
    p['stop']=p['stale']>=c['early_stop_patience']
    return p


def metrics(rows,predictions,weights):
    from tools.train_mortal_teacher import teacher_metrics
    result=teacher_metrics(rows,predictions)
    if set(result['classes'])!=set(weights):raise ValueError('Validation strata differ from training')
    result['weighted_ce']=sum(weights[k]*result['classes'][k]['ce'] for k in weights)
    result['weighted_accuracy']=sum(weights[k]*result['classes'][k]['accuracy'] for k in weights)
    from jevmahjong.endgame.public_policy import original_policy_keys
    opportunities={}
    for kind in ('reach','chi','pon','ankan','daiminkan','kakan'):
        counts=Counter()
        for row in rows:
            q=row['questions']['action']
            if kind not in {v['type'] for v in q['criteria'].values()}:continue
            p=predictions[row['_meta']['id']];chosen=max(original_policy_keys(row),key=p.get)
            teacher=q['criteria'][row['_meta']['teacher_action']]['type']==kind
            student=q['criteria'][chosen]['type']==kind
            counts['positions']+=1;counts['teacher_selects']+=int(teacher);counts['student_selects']+=int(student)
            counts['true_positive' if teacher and student else 'false_negative' if teacher else
                   'false_positive' if student else 'true_negative']+=1
        opportunities[kind]=dict(counts)
    result['opportunities']=opportunities
    return result


def fresh_model(c):
    import torch
    from peft import get_peft_model_state_dict
    from jevany.model import DecisionModel,load_preprocessor
    torch.manual_seed(c['seed'])
    tok=load_preprocessor(c['base_model'],multimodal=False,backbone_adapter='auto')
    model=DecisionModel(c['base_model'],tok,'cuda',lora=c['fresh_lora_rank'],head_dim=c['fresh_head_dim'],
        head_residual_dim=0,option_isolation=False,special_embeddings=False,lora_targets='all',lora_dropout=.05,
        dtype=torch.bfloat16,multimodal=False,backbone_adapter='auto',branch_mode='auto',decision_mode='pointer')
    for cfg in model.lm.peft_config.values():cfg.base_model_name_or_path=c['base_id']
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.lm.config.use_cache=False
    adapter0={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model.lm,save_embedding_layers=False).items()}
    b=[v for k,v in adapter0.items() if 'lora_B' in k]
    if not b or any(torch.count_nonzero(v).item() for v in b):raise ValueError('Fresh LoRA B matrices must be zero')
    head0={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()}
    hp=list(model.head.parameters());ids={id(p) for p in hp}
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(id(p) not in ids and 'lora_' not in n for n,p in named):raise ValueError('Unexpected trainable base parameters')
    lp=[p for _,p in named if id(p) not in ids]
    if not hp or not lp:raise ValueError('Both fresh LoRA and head must be trainable')
    return tok,model,adapter0,head0,lp,hp


def fit(c,proof):
    import torch
    from peft import get_peft_model_state_dict
    from jevany.data import materialize
    from jevany.checkpoint import Meta
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.targeted_refresh import balanced_batches
    from tools.continue_endgame_fourth_lora import normal_predictions
    out=ROOT/c['out'];verify(proof)
    data=json.loads((out/'data-ready-private.json').read_text())
    if (data['status']!='fresh_base_teacher_data_ready' or not data['no_v8_replay_targets']
            or not c['no_old_checkpoint_initialization'] or not c['no_v8_replay_targets']):
        raise ValueError('Fresh-base teacher-only protocol required')
    verify({'files':data['files']});verify(json.loads((out/'registered-data-code-private.json').read_text()))
    if (out/'training-run').exists():raise ValueError('Existing fit must be inspected rather than overwritten')
    train=read_rows(out/'train.jsonl');val=read_rows(out/'validation.jsonl')
    if len(train)!=data['train'] or len(val)!=data['validation']:raise ValueError('Dataset lengths differ')
    for r in train+val:
        q=r['questions']['action']
        if q['label']!=r['_meta']['teacher_action'] or r['_meta']['source']!='mortal_teacher_only_no_v8_targets':
            raise ValueError('Targets must come directly from the fixed teacher')
    schedule=balanced_batches(train,c['batch_quotas'],c['max_updates'],seed=c['seed'])
    floor=data['observation_updates'];weights=data['loss_mass']
    if set(i for b in schedule[:floor] for i in b)!=set(range(len(train))):raise ValueError('Full exposure floor failed')
    proof=dict(proof,data_files=data['files'],observation_updates=floor,loss_mass=weights,
               no_old_checkpoint_initialization=True,no_v8_replay_targets=True,batch_schedule=schedule)
    write(out/'training-protocol-private.json',proof)
    tok,model,adapter0,head0,lp,hp=fresh_model(c);parameters=lp+hp
    optimizer=torch.optim.AdamW([{'params':lp,'lr':c['lora_learning_rate']},
        {'params':hp,'lr':c['head_learning_rate']}],weight_decay=c['weight_decay'])
    write(out/'initialization-private.json',dict(base=c['base_id'],base_revision=c['base_revision'],
        fresh_lora=True,fresh_random_head=True,initial_lora_B_all_zero=True,
        no_old_checkpoint_or_adapter_or_head_loaded=True,no_v8_replay_targets=True,
        trainable_lora=sum(p.numel() for p in lp),trainable_head=sum(p.numel() for p in hp),
        job_id=os.environ['SLURM_JOB_ID']))
    enc={}
    for i,row in enumerate(train):
        rid=row['_meta']['id'];value=model.encode(tok,materialize(row),max_state=4096,max_branch=8192,strict=True)
        if value.get('state_truncated') or len(value['ids'])>8192:raise ValueError('Training truncation forbidden')
        enc[rid]=value
        if (i+1)%256==0:
            event=dict(status='encoding_training_tokens',encoded=i+1,total=len(train))
            write(out/'training-progress-private.json',event);print(json.dumps(event),flush=True)
    root=out/'training-run';curve=[];state=None;seen=set();started=time.monotonic()

    def evaluate(update):
        nonlocal state
        path=root/'checkpoints'/f'step-{update:06d}';path.mkdir(parents=True,exist_ok=False);model.eval()
        model.lm.save_pretrained(path,save_embedding_layers=False);tok.save_pretrained(path)
        meta=Meta(base=c['base_id'],base_revision=c['base_revision'],lora=c['fresh_lora_rank'],
            head_dim=c['fresh_head_dim'],head_residual_dim=0,option_isolation=model.option_isolation,
            special_embeddings=model.special_embeddings,multimodal=False,backbone_adapter=model.backbone_adapter,
            branch_mode=model.branch_mode,tokenizer_saved=True,weights_dtype='bf16',decision_mode='pointer',temperature=1.,
            head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},
            extra=dict(optimizer_step=update,stage_optimizer_step=update,policy_version='mortal-base-v1',
                       training_backend='fresh_base_mortal_teacher',base_only_initialization=True))
        torch.save(meta.to_dict(),path/'head.pt')
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        try:predictions=normal_predictions(path,val);report=metrics(val,predictions,weights)
        finally:torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        state=progress(update,report['weighted_ce'],state,floor,c,len(seen)==len(train))
        adapter=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        lc=any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in adapter.items())
        hc=any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())
        if update and not(lc and hc):raise ValueError('Fresh LoRA and head must both change')
        point=dict(update=update,metrics=report,progress=dict(state),unique_exposure=len(seen),
            lora_changed=lc,head_changed=hc,validation_path='actual_saved_normal_serving',seconds=time.monotonic()-started)
        curve.append(point);write(root/'predictions'/f'step-{update:06d}-private.json',predictions)
        torch.save(dict(step=update,optimizer=optimizer.state_dict(),seen_ids=sorted(seen),
                        torch_rng=rng,cuda_rng=cuda_rng),path/'trainer_state.pt')
        write(out/'training-validation-private.json',point);write(out/'curve-private.json',curve)
        write(root/'selection.json',dict(selected_update=state['selected_update'],release_qualified=False))
        print(json.dumps(point),flush=True)

    evaluate(0);reason='registered_max1536_updates'
    for update,batch in enumerate(schedule,1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=defaultdict(list)
        scale=learning_rate_scale(update,c)
        for group,lr in zip(optimizer.param_groups,(c['lora_learning_rate'],c['head_learning_rate'])):group['lr']=lr*scale
        for index in batch:
            row=train[index];q=row['questions']['action'];src=q['src'];rid=row['_meta']['id']
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.forward(enc[rid])[0]
            target=torch.tensor([q['target'][k] for k in q['criteria']],device=z.device,dtype=torch.float32)
            if abs(float(target.sum())-1.)>1e-5:raise ValueError('Teacher target not normalized')
            ce=-(target*z.float().log_softmax(-1)).sum();loss=ce*weights[src]/c['batch_quotas'][src]
            if not torch.isfinite(loss):raise ValueError('Nonfinite teacher loss')
            loss.backward();losses[src].append(float(ce.detach()));seen.add(rid)
        norm=torch.nn.utils.clip_grad_norm_(parameters,c['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
        event=dict(status='training',update=update,unique_exposure=len(seen),training_records=len(train),
            gradient_norm=float(norm),lr_scale=scale,source_counts={k:len(v) for k,v in losses.items()},
            source_ce={k:sum(v)/len(v) for k,v in losses.items()},seconds=time.monotonic()-started)
        write(out/'training-progress-private.json',event)
        with (out/'updates-private.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if update%8==0:print(json.dumps(event),flush=True)
        if update%c['validation_every']==0:
            evaluate(update)
            if state['stop']:reason='four_stale_checks_after_full_exposure_floor';break
    verify(proof);verify({'files':data['files']})
    if state['selected_update'] is None or len(seen)!=len(train):raise ValueError('No fully exposed selected candidate')
    result=dict(status='fresh_base_teacher_fit_complete',last_update=update,selected_update=state['selected_update'],
        stop_reason=reason,observation_updates=floor,unique_exposure=len(seen),curve=curve,
        no_old_checkpoint_initialization=True,release_qualified=False,deployed=False,raw_publication=False)
    write(out/'fit-result-private.json',result);write(out/'fit-status-private.json',{k:v for k,v in result.items() if k!='curve'})
    del optimizer,model;gc.collect();torch.cuda.empty_cache()
    return result


def audit(c,result):
    from tools.continue_endgame_fourth_lora import normal_predictions
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    out=ROOT/c['out'];data=json.loads((out/'data-ready-private.json').read_text())
    val=read_rows(out/'validation.jsonl');points=[]
    for update in sorted({result['last_update'],result['selected_update']}):
        path=out/'training-run/checkpoints'/f'step-{update:06d}'
        report=metrics(val,normal_predictions(path,val),data['loss_mass'])
        logged=next(x for x in result['curve'] if x['update']==update)
        if report!=logged['metrics']:raise ValueError('Normal loaded model failed validation reproduction')
        points.append(dict(update=update,metrics=report,identity=bundle_identity(path)))
    report=dict(status='fresh_base_saved_model_audit_passed',audits=points,release_qualified=False)
    write(out/'audit-private.json',report);return report


def compare(c,result,model_audit):
    import torch
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.policy.routing import load_model
    from jevmahjong.endgame.fourth_policy import FourthChaseAgent
    from jevmahjong.teachers.mortal import MortalTeacher
    from tools.train_mortal_teacher import play_match
    from tools.audit_mortal_teacher_run import audit_games
    out=ROOT/c['out'];rows,parent_id,seeds=cached_parent(c)
    selected=result['selected_update'];path=out/'training-run/checkpoints'/f'step-{selected:06d}'
    identity=next(x['identity'] for x in model_audit['audits'] if x['update']==selected)
    raw=out/'comparison-games-private.jsonl'
    if raw.exists():raise ValueError('Existing comparison requires inspection')
    write(out/'comparison-protocol-private.json',dict(seeds=seeds,selected_update=selected,
        arms=dict(parent=parent_id,candidate=identity),cached_parent_sha256=c['reference_games_sha256'],
        same_formatter_and_guard=True,stage='reused_development_grid_not_confirmation',raw_publication=False))
    teacher=MortalTeacher(c['teacher_root']);client=load_model(path);agent=FourthChaseAgent(client);started=time.monotonic()
    with raw.open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
        f.flush()
        for seed in seeds:
            for profile in ('calling','mortal'):
                for seat in range(4):
                    row=dict(arm='candidate',**play_match(c,agent,teacher,seed,seat,profile))
                    rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
                    state=dict(status='complete_match_development_running',candidate_games_complete=len(rows)-64,
                        candidate_games_total=c['new_candidate_games'],parent_games_reused=64,
                        seconds=time.monotonic()-started,job_id=os.environ['SLURM_JOB_ID'])
                    write(out/'comparison-status-private.json',state)
                    if len(rows)%8==0:print(json.dumps(state),flush=True)
    del agent,client;gc.collect();torch.cuda.empty_cache()
    if bundle_identity(path)!=identity:raise ValueError('Comparison modified candidate')
    report=audit_games(rows,seeds)
    report.update(status='fresh_base_complete_match_accounting_and_statistics_passed',selected_update=selected,
        raw_publication=False,comparison_games_sha256=sha(raw),cached_parent_games=64,new_candidate_games=64)
    write(out/'comparison-result-private.json',report)
    write(out/'comparison-status-private.json',dict(status='complete',candidate_games_complete=64,parent_games_reused=64,
        release_qualified=False));print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    require_gpu();os.chdir(ROOT);c=json.loads(a.config.read_text());sys.path.insert(0,c['upstream_source'])
    out=ROOT/c['out'];proof=registration(c,('configs/training/mortal-base-v1.json','tools/build_mortal_base.py',
        'tools/train_mortal_base.py','tests/core/test_mortal_base.py','tests/core/test_mortal_base_training.py',
        'tools/train_mortal_teacher.py','tools/continue_endgame_fourth_lora.py','tools/audit_mortal_teacher_run.py',
        'docs/experiments/plans/mortal-base-training-20261120.md'))
    from jevmahjong.endgame.collect import write
    # Fail before any expensive fit if the reusable opponent grid/provenance is invalid.
    cached_parent(c)
    write(out/'registered-training-code-private.json',proof);started=time.monotonic()
    result=fit(c,proof);model_audit=audit(c,result);compare(c,result,model_audit)
    verify(proof)
    write(out/'pipeline-exit-private.json',dict(status='complete',exit_code=0,job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.monotonic()-started,release_qualified=False))


if __name__=='__main__':main()
