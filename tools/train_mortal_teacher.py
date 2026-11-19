"""Continue existing Jev LoRA/head with a local Mortal teacher and retention."""
import argparse
from collections import Counter, defaultdict
import copy
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_mortal_teacher import sha, verify, registration, require_gpu


def teacher_metrics(rows, predictions):
    from jevmahjong.endgame.public_policy import original_policy_keys
    classes = defaultdict(list)
    for r in rows:
        q=r['questions']['action']; p=predictions[r['_meta']['id']]
        label=r['_meta']['teacher_action']; eligible=original_policy_keys(r)
        if label not in eligible or set(p)!=set(q['criteria']) or not all(math.isfinite(v) and v>=0 for v in p.values()):
            raise ValueError('Teacher benchmark legal support or probabilities differ')
        if abs(sum(p.values())-1.)>1e-5:
            raise ValueError('Teacher benchmark prediction is not a distribution')
        classes[q['src']].append(dict(ce=-math.log(max(p[label],1e-30)),
            hit=float(max(eligible,key=p.get)==label)))
    if len(classes)!=7:
        raise ValueError('Teacher validation requires all seven decision classes')
    per={k:dict(positions=len(v),ce=sum(x['ce'] for x in v)/len(v),
                accuracy=sum(x['hit'] for x in v)/len(v)) for k,v in classes.items()}
    return dict(classes=per,macro_ce=sum(v['ce'] for v in per.values())/7,
        macro_accuracy=sum(v['accuracy'] for v in per.values())/7,
        overall_accuracy=sum(x['hit'] for v in classes.values() for x in v)/len(rows),positions=len(rows))


def research_screen(metrics, anchor):
    ce=metrics['teacher']['macro_ce']; initial=anchor['teacher']['macro_ce']
    flags=dict(teacher_macro_ce_gain=ce<=initial*.98,
        endgame_primary_within_tolerance=metrics['endgame_primary']>=anchor['endgame_primary']-.02-1e-12,
        holding_within_tolerance=metrics['old']['legacy']['utility_by_stratum']['hold']
            >=anchor['old']['legacy']['utility_by_stratum']['hold']-.05-1e-12)
    return dict(flags,research_eligible=all(flags.values()),release_qualified=False)


def progress_state(update, metrics, flags, previous, c, exposure_met):
    ce=metrics['teacher']['macro_ce']
    if not math.isfinite(ce):
        raise ValueError('Nonfinite teacher validation')
    if update==0:
        return dict(best_ce=ce,selected_update=None,selected_ce=None,stale=0,stop=False)
    p=dict(previous)
    improved=ce<p['best_ce']*(1-1e-4)
    p['best_ce']=min(p['best_ce'],ce)
    if flags['research_eligible'] and (p['selected_ce'] is None or ce<p['selected_ce']-1e-12):
        p.update(selected_update=update,selected_ce=ce)
    p['stale']=0 if update<=c['observation_new_updates'] or not exposure_met or improved else p['stale']+1
    p['stop']=p['stale']>=c['early_stop_patience']
    return p


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate_rows(c, checkpoint, teacher_rows, old_rows, fresh_rows, reference):
    from tools.continue_endgame_fourth_lora import normal_predictions
    from tools.train_targeted_endgame import fresh_metrics
    from jevmahjong.endgame.fourth_chase_training import chase_validation
    predictions=normal_predictions(checkpoint,teacher_rows+old_rows+fresh_rows)
    old=chase_validation(old_rows,{r['_meta']['id']:predictions[r['_meta']['id']] for r in old_rows},reference)
    fresh=fresh_metrics(fresh_rows,predictions)
    result=dict(teacher=teacher_metrics(teacher_rows,predictions),old=old,fresh=fresh,
        endgame_primary=.5*(old['fourth']['top2']+fresh['fourth']['top2']))
    return result,predictions


def fit(c, proof):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from jevany.model import DecisionModel, load_preprocessor
    from jevany.checkpoint import Checkpoint
    from jevany.data import materialize
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.targeted_refresh import balanced_batches
    out=ROOT/c['out']; verify(proof)
    data=json.loads((out/'data-ready-private.json').read_text())
    if data['status']!='mortal_teacher_data_ready' or data['train']!=1898 or data['teacher_validation']!=256:
        raise ValueError('Teacher data are not complete')
    verify({'files':data['files']});verify(json.loads((out/'registered-data-code-private.json').read_text()))
    if (out/'training-run').exists():
        raise ValueError('Existing fit must be inspected, not restarted')
    parent=ROOT/c['parent_checkpoint']
    if bundle_identity(parent)!=data['parent_identity']:
        raise ValueError('Parent176 weights changed after data admission')
    train=read_rows(out/'train.jsonl'); val=read_rows(out/'validation.jsonl')
    oldval=read_rows(ROOT/c['retention_root']/'validation.jsonl')
    fresh=read_rows(ROOT/c['retention_root']/'monitor.jsonl')
    reference=json.loads((ROOT/c['reference_root']/'reference-targets-private.json').read_text())['rows']
    schedule=balanced_batches(train,c['batch_quotas'],c['max_new_updates'],seed=2026111921)
    if set(i for batch in schedule[:c['observation_new_updates']] for i in batch)!=set(range(len(train))):
        raise ValueError('Observation phase must expose every retained and teacher training record')
    proof=dict(proof,data_files=data['files'],parent_identity=data['parent_identity'],batch_schedule=schedule,
               old_validation_files={str(ROOT/c['retention_root']/p):sha(ROOT/c['retention_root']/p)
                   for p in ('validation.jsonl','monitor.jsonl')})
    write(out/'training-protocol-private.json',proof)
    verify({'files':proof['old_validation_files']})
    torch.manual_seed(2026111921)
    ck=Checkpoint(str(parent));meta=ck.meta
    tok=load_preprocessor(str(parent),multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter)
    model=DecisionModel(c['base_model'],tok,'cuda',lora=16,head_dim=meta.head_dim,
        head_residual_dim=meta.head_residual_dim,lora_targets='all',lora_dropout=.05,
        option_isolation=meta.option_isolation,special_embeddings=meta.special_embeddings,
        multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter,branch_mode=meta.branch_mode,
        dtype=torch.bfloat16,decision_mode=meta.decision_mode)
    for cfg in model.lm.peft_config.values(): cfg.base_model_name_or_path=meta.base
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.lm.config.use_cache=False;ck.warm_start(model,meta)
    adapter0=load_file(str(parent/'adapter_model.safetensors'),device='cpu')
    head_meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True);head0=head_meta['head']
    adapter=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
    if (set(adapter)!=set(adapter0) or any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in adapter.items())
            or any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())):
        raise ValueError('Initialization is not the exact176 adapter and action head')
    hp=list(model.head.parameters());head_ids={id(p) for p in hp}
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(id(p) not in head_ids and 'lora_' not in n for n,p in named):
        raise ValueError('Frozen base model has unexpected trainable parameters')
    lp=[p for _,p in named if id(p) not in head_ids];parameters=[p for _,p in named]
    optimizer=torch.optim.AdamW([{'params':lp,'lr':c['lora_learning_rate']},
        {'params':hp,'lr':c['head_learning_rate']}],weight_decay=c['weight_decay'])
    write(out/'initialization-private.json',dict(exact_parent176=True,new_optimizer=True,new_runtime_networks=0,
        trainable_lora=sum(p.numel() for p in lp),trainable_head=sum(p.numel() for p in hp),job_id=os.environ['SLURM_JOB_ID']))
    enc={}
    for i,row in enumerate(train):
        value=model.encode(tok,materialize(row),max_state=4096,max_branch=8192,strict=True)
        if value.get('state_truncated') or len(value['ids'])>8192:
            raise ValueError('No training truncation is permitted')
        enc[row['_meta']['id']]=value
        if (i+1)%128==0:
            write(out/'training-progress-private.json',dict(status='encoding_training_tokens',encoded=i+1,total=len(train)))
    root=out/'training-run';curve=[];state=None;anchor=None;seen=set();started=time.monotonic()

    def evaluate(update):
        nonlocal state,anchor
        total=c['parent_total_step']+update;path=root/'checkpoints'/f'step-{total:06d}'
        path.mkdir(parents=True,exist_ok=False);model.eval()
        model.lm.save_pretrained(path,save_embedding_layers=False);tok.save_pretrained(path)
        saved=dict(head_meta,head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},
            optimizer_step=total,stage_optimizer_step=update,training_backend='mortal_teacher_v1_from176')
        torch.save(saved,path/'head.pt')
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        try: metrics,predictions=evaluate_rows(c,path,val,oldval,fresh,reference)
        finally: torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        if update==0:
            if abs(metrics['endgame_primary']-.23514093137254902)>1e-12:
                raise ValueError('Parent176 normal-serving endgame baseline differs')
            anchor=copy.deepcopy(metrics);write(out/'anchor-private.json',anchor)
        flags=research_screen(metrics,anchor)
        state=progress_state(update,metrics,flags,state,c,len(seen)==len(train))
        actual=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        lora_changed=any(not torch.equal(v.detach().cpu(),adapter0[k]) for k,v in actual.items())
        head_changed=any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())
        if update and not (lora_changed and head_changed):
            raise ValueError('Both existing LoRA and action head must actually update')
        point=dict(new_update=update,total_step=total,metrics=metrics,screen=flags,progress=state,
            unique_exposure=len(seen),lora_changed=lora_changed,head_changed=head_changed,
            validation_path='actual_saved_normal_serving',seconds=time.monotonic()-started)
        curve.append(point);write(root/'predictions'/f'step-{total:06d}-private.json',predictions)
        torch.save(dict(step=update,total_step=total,optimizer=optimizer.state_dict(),seen_ids=sorted(seen),
                        torch_rng=rng,cuda_rng=cuda_rng),path/'trainer_state.pt')
        write(out/'training-validation-private.json',point);write(out/'curve-private.json',curve)
        write(root/'selection.json',dict(selected_new_update=state['selected_update'],release_qualified=False))
        print(json.dumps(dict(validation_step=total,new_update=update,teacher=metrics['teacher'],
            endgame_primary=metrics['endgame_primary'],screen=flags,progress=state,exposure=len(seen))),flush=True)

    evaluate(0);reason='registered_max512_new_updates'
    for update,batch in enumerate(schedule,1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=defaultdict(list)
        scale=min(1.,update/c['warmup_steps'])
        optimizer.param_groups[0]['lr']=c['lora_learning_rate']*scale
        optimizer.param_groups[1]['lr']=c['head_learning_rate']*scale
        for index in batch:
            row=train[index];q=row['questions']['action'];src=q['src'];rid=row['_meta']['id']
            with torch.autocast('cuda',dtype=torch.bfloat16): z=model.forward(enc[rid])[0]
            target=torch.tensor([q['target'][k] for k in q['criteria']],device=z.device,dtype=torch.float32)
            if target.sum()<=0: raise ValueError('Empty training target')
            target/=target.sum();ce=-(target*z.float().log_softmax(-1)).sum()
            loss=ce*c['loss_mass'][src]/c['batch_quotas'][src]
            if not torch.isfinite(loss): raise ValueError('Nonfinite teacher training loss')
            loss.backward();losses[src].append(float(ce.detach()));seen.add(rid)
        norm=torch.nn.utils.clip_grad_norm_(parameters,c['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        event=dict(status='training',new_update=update,total_step=c['parent_total_step']+update,
            unique_exposure=len(seen),training_records=len(train),gradient_norm=float(norm),
            source_counts={k:len(v) for k,v in losses.items()},source_ce={k:sum(v)/len(v) for k,v in losses.items()},
            seconds=time.monotonic()-started)
        write(out/'training-progress-private.json',event)
        with (out/'updates-private.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        if update%8==0: print(json.dumps(event),flush=True)
        if update%c['validation_every_steps']==0:
            evaluate(update)
            if state['stop']: reason='four_stale_checks_after192_and_full_exposure';break
    verify(proof);verify({'files':data['files']});verify({'files':proof['old_validation_files']})
    result=dict(status='mortal_teacher_fit_complete',last_new_update=update,last_total_step=c['parent_total_step']+update,
        selected_new_update=state['selected_update'],stop_reason=reason,curve=curve,
        release_qualified=False,independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'fit-result-private.json',result);write(out/'fit-status-private.json',{k:v for k,v in result.items() if k!='curve'})
    del model,optimizer;gc.collect();torch.cuda.empty_cache()


def audit(c):
    from jevmahjong.endgame.collect import write
    out=ROOT/c['out'];proof=json.loads((out/'training-protocol-private.json').read_text())
    verify(proof);verify({'files':proof['data_files']});verify({'files':proof['old_validation_files']})
    result=json.loads((out/'fit-result-private.json').read_text());anchor=json.loads((out/'anchor-private.json').read_text())
    reference=json.loads((ROOT/c['reference_root']/'reference-targets-private.json').read_text())['rows']
    val=read_rows(out/'validation.jsonl');old=read_rows(ROOT/c['retention_root']/'validation.jsonl')
    fresh=read_rows(ROOT/c['retention_root']/'monitor.jsonl');selected=result['selected_new_update'];audits=[]
    for update in sorted({result['last_new_update']}|({selected} if selected is not None else set())):
        path=out/'training-run/checkpoints'/f"step-{c['parent_total_step']+update:06d}"
        metrics,_=evaluate_rows(c,path,val,old,fresh,reference)
        logged=next(x for x in result['curve'] if x['new_update']==update)
        if metrics!=logged['metrics']:
            raise ValueError('Audited normal-loaded checkpoint metrics differ')
        flags=research_screen(metrics,anchor)
        audits.append(dict(new_update=update,total_step=c['parent_total_step']+update,metrics=metrics,screen=flags,
                           selected_eligible=update==selected and flags['research_eligible']))
    verify(proof)
    report=dict(status='mortal_teacher_saved_model_audit_complete',audits=audits,
        eligible_for_development=any(x['selected_eligible'] for x in audits),release_qualified=False)
    write(out/'audit-private.json',report);print(json.dumps(report),flush=True)
    return report


def paired_summary(rows, expected_groups):
    from scipy.stats import t
    def mean(v): return sum(v)/len(v)
    arms={arm:[r for r in rows if r['arm']==arm] for arm in ('parent','candidate')}
    expected={(n,p,s) for n in expected_groups for p in ('calling','mortal') for s in range(4)}
    for values in arms.values():
        if len(values)!=len(expected) or {(r['seed'],r['profile'],r['seat']) for r in values}!=expected:
            raise ValueError('Incomplete or duplicate paired complete-game grid')
        if any(not r['completed'] or not r['ledger_passed'] for r in values):
            raise ValueError('Truncated or unreconciled evaluation game')
    stats={arm:dict(games=len(v),top2_rate=mean([int(r['rank']<=2) for r in v]),
        first_rate=mean([int(r['rank']==1) for r in v]),fourth_rate=mean([int(r['rank']==4) for r in v]),
        mean_rank=mean([r['rank'] for r in v]),mean_net_points=mean([r['net_points'] for r in v]),
        hand_win_rate=sum(r['wins'] for r in v)/sum(r['hands'] for r in v),
        dealin_rate=sum(r['dealins'] for r in v)/sum(r['hands'] for r in v),
        choices=dict(sum((Counter(r['choices']) for r in v),Counter()))) for arm,v in arms.items()}
    diffs=[]
    for seed in expected_groups:
        diffs.append(mean([int(r['rank']<=2) for r in arms['candidate'] if r['seed']==seed])
                    -mean([int(r['rank']<=2) for r in arms['parent'] if r['seed']==seed]))
    gain=mean(diffs);n=len(diffs)
    se=math.sqrt(sum((v-gain)**2 for v in diffs)/(n-1)/n)
    width=float(t.ppf(.975,n-1))*se
    return dict(arms=stats,top2_gain=gain,whole_seed_ci95=[gain-width,gain+width],seed_groups=n,
        positive_development_point_estimate=gain>0,independent_confirmation=False,release_qualified=False)


def play_match(c,agent,teacher,seed,seat,profile):
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.evaluation.agents import EfficiencyAgent,CallingOpponent
    from jevmahjong.core.scoring import settlement_ledger
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
    obs=env.reset();ordinary=EfficiencyAgent();caller=CallingOpponent();mortal=teacher.player((seat+1)%4)
    for step in range(30000):
        if env.done(): break
        actions={}
        for p,o in obs.items():
            opponent=mortal if profile=='mortal' else caller
            actions[p]=(agent if p==seat else opponent if p==(seat+1)%4 else ordinary).act(o)
        obs=env.step(actions)
    if not env.done(): raise RuntimeError('Complete-match development was truncated')
    log=env.mjai_log;ledger=settlement_ledger(log,env.scores())
    counts=Counter(e['type'] for e in log if e.get('actor')==seat)
    return dict(seed=seed,seat=seat,profile=profile,rank=env.ranks()[seat],net_points=env.scores()[seat]-25000,
        final_scores=list(env.scores()),hands=ledger['hands'],wins=counts['hora'],
        dealins=sum(e['type']=='hora' and e['actor']!=seat and e['target']==seat for e in log),
        choices={k:counts[k] for k in ('chi','pon','ankan','daiminkan','kakan','reach')},
        completed=True,ledger_passed=True,mjai_log=log)


def compare(c, audit_report):
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.policy.routing import load_model
    from jevmahjong.endgame.fourth_policy import FourthChaseAgent
    from jevmahjong.teachers.mortal import MortalTeacher
    import torch
    out=ROOT/c['out'];proof=json.loads((out/'training-protocol-private.json').read_text());verify(proof)
    if not audit_report['eligible_for_development']:
        write(out/'comparison-status-private.json',dict(status='not_started_no_eligible_checkpoint',release_qualified=False))
        return
    point=next(x for x in audit_report['audits'] if x['selected_eligible'])
    candidate=out/'training-run/checkpoints'/f"step-{point['total_step']:06d}"
    parent=ROOT/c['parent_checkpoint']
    checkpoints={'parent':parent,'candidate':candidate};ids={k:bundle_identity(v) for k,v in checkpoints.items()}
    seeds=list(range(c['comparison_seed_start'],c['comparison_seed_start']+c['comparison_seed_groups']))
    protocol=dict(seeds=seeds,profiles=['calling','mortal'],seats=4,arms=ids,
        selected_total_step=point['total_step'],same_formatter_and_guard=True,
        whole_seed_interval=True,stage='development_not_confirmation',raw_publication=False)
    write(out/'comparison-protocol-private.json',protocol)
    raw=out/'comparison-games-private.jsonl'
    if raw.exists(): raise ValueError('Inspect existing comparison rather than silently repeating games')
    teacher=MortalTeacher(c['teacher_root']);rows=[];started=time.monotonic()
    for arm,path in checkpoints.items():
        client=load_model(path);agent=FourthChaseAgent(client)
        with raw.open('a') as f:
            for seed in seeds:
                for profile in ('calling','mortal'):
                    for seat in range(4):
                        row=dict(arm=arm,**play_match(c,agent,teacher,seed,seat,profile))
                        rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
                        state=dict(status='complete_match_development_running',games_complete=len(rows),games_total=128,
                            arm=arm,seconds=time.monotonic()-started,job_id=os.environ['SLURM_JOB_ID'])
                        write(out/'comparison-status-private.json',state)
                        if len(rows)%8==0: print(json.dumps(state),flush=True)
        del agent,client;gc.collect();torch.cuda.empty_cache()
        if bundle_identity(path)!=ids[arm]: raise ValueError('Evaluation modified checkpoint')
    result=paired_summary(rows,seeds);verify(proof)
    write(out/'comparison-result-private.json',result)
    write(out/'comparison-status-private.json',dict(status='complete',games_complete=len(rows),games_total=128,
        selected_total_step=point['total_step'],release_qualified=False))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    require_gpu();os.chdir(ROOT);c=json.loads(a.config.read_text());sys.path.insert(0,c['upstream_source'])
    out=ROOT/c['out'];proof=registration(c,('tools/train_mortal_teacher.py','tools/continue_endgame_fourth_lora.py',
        'tools/train_targeted_endgame.py','tests/core/test_mortal_teacher.py','tests/core/test_mortal_training.py'))
    from jevmahjong.endgame.collect import write
    write(out/'registered-training-code-private.json',proof)
    started=time.monotonic()
    fit(c,proof);report=audit(c);compare(c,report)
    write(out/'pipeline-exit-private.json',dict(status='complete',exit_code=0,job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.monotonic()-started,release_qualified=False))


if __name__=='__main__': main()
