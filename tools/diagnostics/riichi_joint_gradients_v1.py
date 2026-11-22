"""Dropout-free local LoRA/head gradient geometry on fixed TRAIN-only panels."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_basic_repair import MASS

SEED=2026112306


def select_panels(rows):
    groups=defaultdict(list)
    for row in rows:
        key=row['questions']['action']['src'].removeprefix('curriculum1:')
        if key not in MASS:raise ValueError('Only registered six-source TRAIN decisions supported')
        groups[key].append(row)
    if set(groups)!=set(MASS):raise ValueError('Every declared source required')
    selected={}
    for key,values in groups.items():
        ordered=sorted(values,key=lambda row:hashlib.sha256((str(SEED)+':'+row['_meta']['id']).encode()).hexdigest())
        taken=[];seen=set()
        for row in ordered:
            group=row['_meta']['group_id']
            if group in seen:continue
            seen.add(group);taken.append(row)
            if len(taken)==8:break
        if len(taken)!=8:raise ValueError('Eight distinct TRAIN groups/source required')
        selected[key]=taken
    return [{key:values[4*i:4*i+4] for key,values in selected.items()} for i in range(2)]


def geometry(tasks,mask=None):
    import torch
    values={key:(value if mask is None else value[mask]).double() for key,value in tasks.items()}
    if not values or any(not torch.isfinite(value).all() for value in values.values()):raise ValueError('Finite task vectors required')
    lengths={value.numel() for value in values.values()}
    if len(lengths)!=1:raise ValueError('Aligned parameter vectors required')
    norms={key:float(value.norm()) for key,value in values.items()}
    cosines={}
    names=list(values)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            denominator=norms[a]*norms[b]
            cosines[a+'|'+b]=float(torch.dot(values[a],values[b]))/denominator if denominator else None
    if any(not math.isfinite(n) for n in norms.values()):raise ValueError('Nonfinite gradient norm')
    return dict(norms=norms,cosines=cosines,parameter_elements=next(iter(lengths)))


def measure(checkpoint,panels,reg,out):
    import torch
    from jevany.data import materialize
    from jevany.model import load_preprocessor
    from jevmahjong.policy.routing import load_model
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.curriculum_riichi_class_loss import annotate,parts
    identity=bundle(checkpoint)
    client=load_model(checkpoint);model=client.runtime.model;model.eval()
    if model.decision_mode!='pointer' or model.branch_mode!='rows' or model.temperature!=1.:
        raise ValueError('Unchanged plain row-pointer model required')
    named=[]
    for name,parameter in model.named_parameters():
        enabled='lora_' in name or name.startswith('head.')
        parameter.requires_grad_(enabled)
        if enabled:named.append((name,parameter))
    if len(named)!=376 or sum(name.startswith('head.') for name,_ in named)!=4:
        raise ValueError('Actual372 LoRA/four pointer tensors required')
    params=[p for _,p in named];before={n:p.detach().cpu().clone() for n,p in named}
    if any(p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n and not n.startswith('head.')):
        raise ValueError('Base parameters must remain frozen')
    if any(p.dtype!=torch.float32 for _,p in named):raise ValueError('Actual fp32 adapter/head parameters required')
    tokenizer=load_preprocessor(checkpoint,backbone_adapter='text')
    is_head=torch.cat([torch.full((p.numel(),),name.startswith('head.'),dtype=torch.bool) for name,p in named])
    result=[];details=[];max_forward_error=0.
    for panel_index,panel in enumerate(panels):
        means={}
        for source,rows in panel.items():
            components=defaultdict(list)
            for i,row in enumerate(rows):
                rec=annotate(materialize(row),row)
                enc=model.encode(tokenizer,rec,max_state=4096,max_branch=8192,strict=True)
                if enc['state_truncated'] or len(enc['ids'])>8192:raise ValueError('Frozen TRAIN menu truncated')
                with torch.autocast('cuda',dtype=torch.bfloat16):z=model.forward(enc)[0].float()
                if i==0:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):repeat=model.forward(enc)[0].float()
                    error=float((z.detach().softmax(0)-repeat.softmax(0)).abs().max());max_forward_error=max(max_forward_error,error)
                    if error>5e-5:raise ValueError('Gradient-enabled/no-grad forward canary differs')
                values=parts(z,rec['questions'][0],'cuda')
                names=['plain_action_ce'] if source.endswith(':plain') else ['riichi_max_class_ce']
                if source.endswith(':riichi_no'):names.append('conditional_discard_ce')
                for j,name in enumerate(names):
                    grads=torch.autograd.grad(values[name],params,retain_graph=j<len(names)-1,allow_unused=True)
                    vector=torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().flatten() for p,g in zip(params,grads)])
                    if not torch.isfinite(vector).all():raise ValueError('Nonfinite actual parameter gradient')
                    components[name].append(vector)
                details.append(dict(panel=panel_index,source=source,id=row['_meta']['id'],group=row['_meta']['group_id'],
                    loss_parts={name:float(value.detach()) for name,value in values.items()},packed_tokens=len(enc['ids'])))
            means[source]={name:torch.stack(vectors).mean(0) for name,vectors in components.items()}
        zeros=torch.zeros_like(next(iter(next(iter(means.values())).values())))
        combine=lambda component:sum((MASS[source]*values.get(component,zeros) for source,values in means.items()),zeros.clone())
        plain=combine('plain_action_ce');binary=combine('riichi_max_class_ce');conditional=combine('conditional_discard_ce')
        tasks=dict(plain=plain,RI_binary=binary,RI_conditional=conditional,RI_total=binary+conditional,combined=plain+binary+conditional)
        result.append(dict(panel=panel_index,all=geometry(tasks),LoRA=geometry(tasks,~is_head),pointer=geometry(tasks,is_head)))
        print(json.dumps(dict(status='gradient_panel_complete',checkpoint=checkpoint.name,panel=panel_index)),flush=True)
    assert all(torch.equal(p.detach().cpu(),before[name]) for name,p in named)
    assert all(p.grad is None for p in params)
    assert any(row['LoRA']['norms']['combined']>0 for row in result) and any(row['pointer']['norms']['combined']>0 for row in result)
    with (out/(checkpoint.name+'-sample-details-private.json')).open('x') as stream:json.dump(details,stream,indent=2);stream.write('\n')
    return dict(identity=identity['identity_sha256'],panels=result,
        adapter_and_head_tensors_unchanged=True,base_parameters_frozen=True,parameter_grad_buffers_unmodified=True,
        dropout_disabled=True,max_gradient_mode_forward_probability_error=max_forward_error,train_examples=len(details))


def run(registered,out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Scheduled GPU allocation required')
    import torch
    torch.set_num_threads(4)
    reg=read(registered);sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from tools.curriculum_riichi_control import require_registered
    require_registered(reg);out.mkdir(parents=True,exist_ok=False)
    files=dict(reg['inputs']);files[str(registered.resolve())]=sha(registered)
    names=[Path(__file__),ROOT/'tests/core/test_curriculum_riichi_joint_gradients_v1.py',
        ROOT/'docs/experiments/plans/curriculum-riichi-joint-gradients-20261123.md']
    files.update({str(p.resolve()):sha(p) for p in names})
    assert all(sha(p)==h for p,h in files.items())
    receipt=read(reg['training_config']['data_receipt']);train_path=Path(receipt['files']['train']['path'])
    validation_path=Path(receipt['files']['validation']['path'])
    rows=[json.loads(line) for line in train_path.open()];panels=select_panels(rows)
    held={json.loads(line)['_meta']['group_id'] for line in validation_path.open()}
    assert not held&{r['_meta']['group_id'] for panel in panels for values in panel.values() for r in values}
    fit=ROOT/reg['curriculum']['out'];checkpoints={step:fit/'training-run/checkpoints'/f'step-{step:06d}' for step in (0,152)}
    for checkpoint in checkpoints.values():
        for path in checkpoint.iterdir():
            if path.is_file() and path.name!='trainer_state.pt':files[str(path)]=sha(path)
    write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
    write(out/'registered-private.json',dict(method='fixed_TRAIN_dropout_free_joint_gradient_geometry_v1',files=files,
        seed=SEED,checkpoints=[0,152],panels=2,cases_per_source_per_panel=4,
        selected_ids={str(i):{s:[r['_meta']['id'] for r in values] for s,values in panel.items()} for i,panel in enumerate(panels)},
        source_mass=MASS,optimizer_steps=0,raw_gradients_not_public=True))
    os.environ['JEV_BASE_MODEL_DIR']=reg['training_config']['base_load_path'];results={}
    for step,checkpoint in checkpoints.items():
        results[str(step)]=measure(checkpoint,panels,reg,out);gc.collect();torch.cuda.empty_cache()
    if not all(sha(p)==h for p,h in files.items()):raise ValueError('Bound data/checkpoint/source changed')
    summary=dict(status='fixed_TRAIN_real_LoRA_pointer_gradient_geometry_complete',results=results,
        total_example_checkpoint_pairs=96,new_optimizer_steps=0,no_validation_or_final_sample_selection=True,
        dropout_free_bf16_forward_fp32_trainable_parameters=True,
        source_conditional_mean_gradients_weighted_by_registered_mass=True,
        not_actual_stochastic_AdamW_update_or_causal_interference_proof=True,no_new_policy_games_or_parent_admission=True)
    write(out/'summary-private.json',summary);write(out/'binding-private.json',dict(files=files,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registered',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.registered.resolve(),args.out.resolve())
