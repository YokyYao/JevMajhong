"""Continue the existing v8 LoRA and head with source-balanced microbatches."""
import argparse
from collections import defaultdict
import copy
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


QUOTAS={'replay_reference':16,'replay_reference_leading':4,
        'endgame_terminal_win_fourth_chase':4,'endgame_terminal_win_third':2,
        'endgame_terminal_third':2,'endgame_terminal_hold':2,
        'endgame_terminal_fourth_chase_improve':1,'endgame_terminal_fourth_chase_keep':1}


def validate_config(config):
    expected={'schema':'endgame-fourth-balanced-joint-lora-v1','out':'runs/endgame-fourth-lora-balanced-v1',
        'prepared_data_run':'runs/endgame-fourth-top2-chase-v1',
        'optimizer_diagnostic_run':'runs/endgame-fourth-adamw-v1','parent':'released_v8_step2088',
        'trainable_scope':'existing_lora_and_head','lora_rank':16,
        'lora_learning_rate':2e-6,'head_learning_rate':1e-5,'weight_decay':.001,'gradient_clip_norm':1.,
        'learning_rate_warmup_steps':8,'max_optimizer_steps':128,'validation_every_steps':16,
        'early_stop_observation_start':64,'early_stop_patience':3,'soft_progress_min_delta':1e-5,
        'severe_replay_ce_increase_stop':.5,'batch_source_counts':QUOTAS,
        'fourth_escape_constraint':'report_only','fourth_native_kl_constraint':'report_only',
        'per_source_ce_monotonicity':False,'fit_registered':False,'independent_confirmation':False,
        'export_registered':False,'raw_publication':False}
    if config!=expected:raise ValueError('Only the preregistered balanced existing-LoRA continuation is enabled')


def balanced_schedule(records,steps,quotas=QUOTAS,seed=2026111918):
    """Cycle each source without replacement; repeat only after its full pass."""
    pools=defaultdict(list)
    for i,row in enumerate(records):pools[row['questions']['action']['src']].append(i)
    if set(pools)!=set(quotas) or any(type(n) is not int or n<1 for n in quotas.values()):
        raise ValueError('Every positive source needs a registered batch quota')
    rng=random.Random(seed);cursor={k:0 for k in pools}
    for values in pools.values():rng.shuffle(values)
    batches=[]
    for _ in range(steps):
        batch=[]
        for source,n in quotas.items():
            for _ in range(n):
                if cursor[source]==len(pools[source]):rng.shuffle(pools[source]);cursor[source]=0
                batch.append(pools[source][cursor[source]]);cursor[source]+=1
        rng.shuffle(batch);batches.append(batch)
    return batches


def record_loss_coefficient(source,masses,quotas=QUOTAS):
    if set(masses)!=set(quotas) or abs(sum(masses.values())-1.)>1e-9:
        raise ValueError('Fixed normalized source masses must match the balanced quotas')
    return masses[source]/quotas[source]


def fit(config,registration):
    validate_config(config)
    from tools.train_endgame_fourth_adamw import validate_assets,progress_state
    from tools.train_endgame_fourth_chase import verify
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_guards,chase_training_metrics
    out=Path(config['out'])
    if (out/'protocol-private.json').exists() or (out/'training-run').exists():raise ValueError('Existing run; inspect rather than restart')
    if registration['configuration']!=config or registration['ci']['conclusion']!='success':raise ValueError('Public source not verified')
    verify(registration)
    ready,parent,train,validation,fourth=validate_assets(config)
    prior=Path(config['optimizer_diagnostic_run']);prior_audit=json.loads((prior/'postfit-summary-private.json').read_text())
    if (prior_audit['status']!='fourth_adamw_saved_weight_and_service_audit_complete'
            or prior_audit['last_step']!=128 or prior_audit['selected_step']!=0
            or not all(r['control']['passed'] for r in prior_audit['service_audits'])):
        raise ValueError('Complete the longer-head and actual-service diagnostic first')
    batches=balanced_schedule(train,config['max_optimizer_steps'],config['batch_source_counts'])
    if len({i for batch in batches[:64] for i in batch})!=826:raise ValueError('Observation window must include all826 positive rows')
    protocol=dict(registration,parent_identity=ready['parent_identity'],parent=str(parent),
        source_ready_sha256=checksum(Path(config['prepared_data_run'])/'ready-private.json'),
        optimizer_diagnostic_sha256=checksum(prior/'postfit-summary-private.json'),
        data_sha256=ready['data_sha256'],reference_sha256=ready['reference_sha256'],
        loss_mass=ready['loss_recipe']['mass'],training_ids=[r['_meta']['id'] for r in train],
        validation_ids=[r['_meta']['id'] for r in validation],batch_schedule=batches)
    write(out/'protocol-private.json',protocol)
    sys.path.insert(0,ready['source_config']['upstream_source'])
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from jevany.model import DecisionModel,load_preprocessor
    from jevany.checkpoint import Checkpoint
    from jevany.data import materialize
    torch.manual_seed(2026111918);random.seed(2026111918)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=Checkpoint(str(parent));meta=ck.meta;base=ready['source_config']['base_load_path']
    tok=load_preprocessor(str(parent) if meta.tokenizer_saved else base,
                          multimodal=meta.multimodal,backbone_adapter=meta.backbone_adapter)
    model=DecisionModel(base,tok,'cuda',lora=16,head_dim=meta.head_dim,head_residual_dim=meta.head_residual_dim,
        lora_targets='all',lora_dropout=.05,option_isolation=meta.option_isolation,
        special_embeddings=meta.special_embeddings,multimodal=meta.multimodal,
        backbone_adapter=meta.backbone_adapter,branch_mode=meta.branch_mode,
        dtype=torch.bfloat16,decision_mode=meta.decision_mode)
    for value in model.lm.peft_config.values():value.base_model_name_or_path=meta.base
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.lm.config.use_cache=False;ck.warm_start(model,meta)
    original_adapter=load_file(str(parent/'adapter_model.safetensors'),device='cpu')
    original_head=torch.load(parent/'head.pt',map_location='cpu',weights_only=True)['head']
    actual=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
    if set(actual)!=set(original_adapter) or any(not torch.equal(v.detach().cpu(),original_adapter[k]) for k,v in actual.items()):
        raise ValueError('Initialization does not match formal v8 LoRA')
    if any(not torch.equal(v.detach().cpu(),original_head[k]) for k,v in model.head.state_dict().items()):
        raise ValueError('Initialization does not match formal v8 head')
    head_params=list(model.head.parameters());head_ids={id(p) for p in head_params}
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(id(p) not in head_ids and 'lora_' not in n for n,p in named):raise ValueError('Base weights must remain frozen')
    lora_params=[p for _,p in named if id(p) not in head_ids]
    if not lora_params or sum(p.numel() for p in head_params)!=524800:raise ValueError('Existing adapter/head scope missing')
    parameters=[p for _,p in named]
    write(out/'initialization-private.json',dict(parent=str(parent),exact_v8_lora_and_head=True,
        trainable_head_parameters=sum(p.numel() for p in head_params),trainable_lora_parameters=sum(p.numel() for p in lora_params),
        new_residual_networks=0,new_feature_cache_used=False,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    # Tokens can be reused; encoder activations cannot be cached when LoRA trains.
    encodings={}
    for row in train+validation:
        rid=row['_meta']['id'];enc=model.encode(tok,materialize(row),max_state=4096,max_branch=8192,strict=True)
        if enc.get('state_truncated') or len(enc['ids'])>8192:raise ValueError('No training or validation token truncation is permitted')
        encodings[rid]=enc
    write(out/'token-admission-private.json',dict(train=826,validation=186,rejected=0,truncated=0))
    optimizer=torch.optim.AdamW([{'params':lora_params,'lr':config['lora_learning_rate']},
        {'params':head_params,'lr':config['head_learning_rate']}],weight_decay=config['weight_decay'])
    source=Path(config['prepared_data_run']);reference=json.loads((source/'reference-targets-private.json').read_text())['rows']
    masses={k:v for k,v in ready['loss_recipe']['mass'].items() if v>0}
    root=out/'training-run';curve=[];progress=None;seen=set();initial=copy.deepcopy(ready['initial_validation'])
    started=time.monotonic();stop_reason='registered_max128_updates'
    def predictions(rows):
        result={};model.eval()
        with torch.no_grad():
            for row in rows:
                rid=row['_meta']['id'];logits=model.forward(encodings[rid])
                if len(logits)!=1:raise ValueError('Expected one action decision per record')
                values=logits[0].float().softmax(-1).cpu().tolist();keys=list(row['questions']['action']['criteria'])
                if len(values)!=len(keys) or not all(math.isfinite(x) for x in values):raise ValueError('Invalid model distribution')
                result[rid]=dict(zip(keys,values))
        return result
    def evaluate(step):
        nonlocal progress
        pred=predictions(validation);current=chase_validation(validation,pred,reference)
        if step==0:
            # A changed numerical inference path must not lower the original
            # formal-v8 zero anchor and create a spurious training gain.
            initial['fourth']['top2']=max(initial['fourth']['top2'],current['fourth']['top2'])
            write(out/'zero-path-comparison-private.json',dict(live=current,formal=ready['initial_validation'],selection_anchor=initial))
        flags=chase_guards(step,current,initial,ready['source_config'],exposure_met=len(seen)==826)
        progress=progress_state(step,current,flags,progress,config)
        learning=chase_training_metrics(fourth,predictions(fourth),reference)
        path=root/'checkpoints'/f'step-{step:06d}';path.mkdir(parents=True,exist_ok=False)
        model.lm.save_pretrained(path,save_embedding_layers=False);tok.save_pretrained(path)
        saved=torch.load(parent/'head.pt',map_location='cpu',weights_only=True)
        saved.update(head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},
            optimizer_step=step,formal_parent_step=2088,training_backend='source_balanced_existing_lora_and_head',
            policy_version='endgame-fourth-chase-v1',weights_dtype='bf16',tokenizer_saved=True)
        torch.save(saved,path/'head.pt')
        torch.save({'step':step,'optimizer':optimizer.state_dict(),'seen_ids':sorted(seen),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path/'trainer_state.pt')
        adapter=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        changed_adapter=any(not torch.equal(v.detach().cpu(),original_adapter[k]) for k,v in adapter.items())
        changed_head=any(not torch.equal(v.detach().cpu(),original_head[k]) for k,v in model.head.state_dict().items())
        if step and not (changed_adapter and changed_head):raise ValueError('Actual LoRA and head updates are both required')
        point=dict(step=step,metrics=current,guards=flags,progress=progress,training149=learning,
            unique_positive_exposure=len(seen),positive_presentations=32*step,
            lora_changed=changed_adapter,head_changed=changed_head,seconds=time.monotonic()-started)
        curve.append(point);write(root/'predictions'/f'step-{step:06d}-private.json',pred)
        write(root/'selection.json',dict(step=progress['selected_step'],selected_checkpoint=f"checkpoints/step-{progress['selected_step']:06d}"))
        write(out/'progress-private.json',point);write(out/'curve-private.json',curve)
        write(out/'selection-anchor-private.json',initial)
        print(json.dumps({'validation_step':step,'fourth':current['fourth'],'guards':flags,'progress':progress,
                          'training149':learning,'unique_exposure':len(seen),'seconds':point['seconds']}),flush=True)
        return point
    evaluate(0)
    for step,batch in enumerate(batches,1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=defaultdict(list)
        scale=min(1.,step/config['learning_rate_warmup_steps'])
        optimizer.param_groups[0]['lr']=config['lora_learning_rate']*scale
        optimizer.param_groups[1]['lr']=config['head_learning_rate']*scale
        for index in batch:
            row=train[index];rid=row['_meta']['id'];q=row['questions']['action'];source_name=q['src']
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.forward(encodings[rid])[0]
            target=torch.tensor([q['target'][k] for k in q['criteria']],device=logits.device,dtype=torch.float32)
            target=target/target.sum();ce=-(target*logits.float().log_softmax(-1)).sum()
            value=ce*record_loss_coefficient(source_name,masses,config['batch_source_counts'])
            if not torch.isfinite(value):raise ValueError('Nonfinite supervised loss')
            value.backward();losses[source_name].append(float(ce.detach()));seen.add(rid)
        norm=torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        event=dict(step=step,source_ce={k:sum(v)/len(v) for k,v in losses.items()},
            source_counts={k:len(v) for k,v in losses.items()},unique_exposure=len(seen),
            lora_lr=optimizer.param_groups[0]['lr'],head_lr=optimizer.param_groups[1]['lr'],
            gradient_norm=float(norm),seconds=time.monotonic()-started)
        write(out/'training-progress-private.json',event)
        with (out/'updates-private.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        if step%config['validation_every_steps']==0 or step==len(batches):
            point=evaluate(step)
            if point['metrics']['legacy']['original_expert_ce']>initial['legacy']['original_expert_ce']+config['severe_replay_ce_increase_stop']:
                stop_reason='severe_original_replay_ce_regression';break
            if progress['stop']:stop_reason='three_stale_validations_after64_observation_updates';break
    verify(registration);validate_assets(config)
    result=dict(status='fourth_balanced_joint_lora_fit_complete',last_step=step,selected_step=progress['selected_step'],
        stop_reason=stop_reason,curve=curve,unique_positive_exposure=len(seen),same_original_v8_parent=True,
        fourth_escape_is_not_a_veto=True,independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'fit-result-private.json',result);write(out/'status-private.json',{k:v for k,v in result.items() if k!='curve'})
    print(json.dumps({k:v for k,v in result.items() if k!='curve'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Only scheduler-allocated GPU computation is permitted')
    fit(json.loads(args.config.read_text()),json.loads(args.registration.read_text()))
