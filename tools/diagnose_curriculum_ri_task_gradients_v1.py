"""Finite TRAIN-only gradient diagnosis; never updates or qualifies a model.

Eight deterministic distinct families per source (or all if fewer) are fixed
before loading either model. All 19 strata use their current registered loss.
Gradients are measured in eval mode with dropout disabled, without an optimizer.
The geometry is an exploratory local diagnostic, not an Adam update or strength
estimate. Private selections and gradients must not be published.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_progression_parent_loss_v1 import MASS,annotate,parts

RUN=ROOT/'runs/curriculum-ri-task-gradients-v1'
PARENT=ROOT/'runs/curriculum-progression-parent-retention-v1'
REG=PARENT/'fit-plan/registered.json'
PLAN=ROOT/'docs/experiments/plans/curriculum-ri-task-gradients-20261125.md'
ENTRY=ROOT/'cluster/curriculum-ri-task-gradients-v1.slurm'


def select(rows):
    """No outcomes, validation predictions or checkpoint loss rank the sample."""
    grouped=defaultdict(dict)
    for row in rows:
        source=row['questions']['action']['src'];meta=row['_meta']
        if source not in MASS:raise ValueError('Unknown source')
        family=meta['group_id']
        rank=hashlib.sha256(('ri-grad-v1:'+meta['id']).encode()).hexdigest()
        old=grouped[source].get(family)
        if old is None or rank<old[0]:grouped[source][family]=(rank,row)
    if set(grouped)!=set(MASS):raise ValueError('All19 TRAIN sources required')
    return {s:[r for _,r in sorted(families.values(),key=lambda x:x[0])[:8]]
            for s,families in sorted(grouped.items())}


def prepare():
    allocation=resource()
    if (RUN/'registered-private.json').exists():raise ValueError('Never replace a registration')
    reg=read(REG);sys.path[:0]=[reg['source'],reg['training_config']['upstream_source']]
    from jevmahjong.evaluation.unified_contract import bundle
    receipt_path=Path(reg['training_config']['data_receipt']);receipt=read(receipt_path)
    files={str(p):sha(p) for p in (Path(__file__).resolve(),PLAN,ENTRY,REG,receipt_path)}
    pools={}
    for split in ('train','validation'):
        spec=receipt['files'][split];path=Path(spec['path'])
        if sha(path)!=spec['sha256']:raise ValueError('Frozen corpus changed')
        files[str(path)]=spec['sha256']
        with path.open() as stream:pools[split]=[json.loads(line) for line in stream]
        if len(pools[split])!=spec['records']:raise ValueError('Full corpus required')
    if len(pools['train'])!=87595 or len(pools['validation'])!=23688:raise ValueError('Fixed19-source union required')
    tr_ids={r['_meta']['id'] for r in pools['train']};va_ids={r['_meta']['id'] for r in pools['validation']}
    tr_groups={r['_meta']['group_id'] for r in pools['train']};va_groups={r['_meta']['group_id'] for r in pools['validation']}
    if len(tr_ids)!=87595 or len(va_ids)!=23688 or tr_ids&va_ids or tr_groups&va_groups:
        raise ValueError('Distinct TRAIN and validation families required')
    chosen=select(pools['train']);models={}
    for step in (0,1200):
        path=PARENT/'stage3-fit/training-run/checkpoints'/f'step-{step:06d}'
        models[str(step)]=bundle(path)
    if models['1200']['identity_sha256']!='d86892fc74f59b4c31f9028d87eca889347cd3f9406feefbf0f02b91c6fc6733':
        raise ValueError('Actual examined1200 checkpoint required')
    for path in (ROOT/'tools/curriculum_progression_parent_loss_v1.py',
                 ROOT/'tools/curriculum_calls_classroute_v1.py',
                 ROOT/'tools/curriculum_riichi_class_loss.py',
                 ROOT/'tools/curriculum_progression_objective_v1.py'):
        files[str(path)]=sha(path)
    RUN.mkdir(exist_ok=True)
    sample=RUN/'selected-train-private.json'
    write(sample,chosen);files[str(sample)]=sha(sample)
    status={'status':'TRAIN_only_local_gradient_probe_registered_no_model_loaded',
        'files':files,'source':reg['source'],'upstream_source':reg['training_config']['upstream_source'],
        'models':models,'mass':MASS,'samples_per_source':{s:len(v) for s,v in chosen.items()},
        'total_samples_per_model':sum(map(len,chosen.values())),
        'sample_rule':'sha256_ri_grad_v1_identifier_min_per_family_then_first8_per_source',
        'no_cross_split_groups':True,'allocation':allocation,
        'model_loaded':False,'student_updates':0,'new_games':0,'automatic_fit':False}
    write(RUN/'registered-private.json',status)
    print(json.dumps({k:v for k,v in status.items() if k not in ('files','models','allocation')}),flush=True)


def geometry(vectors,mask):
    import torch
    def take(x):return x if mask is None else x[mask]
    decline=[s for s in MASS if s.endswith(':riichi_no')]
    accept=[s for s in MASS if s.endswith(':riichi_yes')]
    new=[s for s in MASS if s.startswith('curriculum3:')]
    def combine(sources):
        return sum((take(vectors[s]).double()*MASS[s] for s in sources),
                   torch.zeros_like(take(next(iter(vectors.values()))),dtype=torch.float64))
    d=combine(decline);a=combine(accept);n=combine(new);total=combine(list(MASS))
    def compare(x,y):
        dot=float(torch.dot(x,y));nx=float(x.norm());ny=float(y.norm())
        return {'dot':dot,'cosine':dot/(nx*ny) if nx and ny else None,
                'left_norm':nx,'right_norm':ny,
                'negative_dot_predicts_first_order_loss_increase_under_negative_right_gradient':dot<0}
    return {'decline_vs_new':compare(d,n),'decline_vs_accept':compare(d,a),
            'decline_vs_total':compare(d,total),
            'decline_vs_each_new':{s:compare(d,combine([s])) for s in new}}


def run():
    allocation=resource();started=time.monotonic();reg=read(RUN/'registered-private.json')
    if reg['status']!='TRAIN_only_local_gradient_probe_registered_no_model_loaded' or reg['mass']!=MASS:
        raise ValueError('Frozen probe registration required')
    for path,digest in reg['files'].items():
        if sha(path)!=digest:raise ValueError('Frozen probe input changed: '+path)
    out=RUN/'probe-v1'
    if out.exists():raise ValueError('Never overwrite an attempted probe')
    out.mkdir()
    sys.path[:0]=[reg['source'],reg['upstream_source']]
    import torch
    torch.set_num_threads(4)
    from jevmahjong.policy.routing import load_model
    from jevmahjong.evaluation.unified_contract import bundle
    from jevany.model import encode
    from jevany.data import materialize
    selected=read(RUN/'selected-train-private.json');result={}
    for step,spec in reg['models'].items():
        path=Path(spec['directory'])
        if bundle(path)!=spec:raise ValueError('Checkpoint changed before probe')
        client=load_model(path);model=client.runtime.model;model.eval()
        named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        if (len(named)!=376 or sum(name.startswith('head.') for name,p in named)!=4
                or any(not (name.startswith('head.') or 'lora_' in name) for name,p in named)):
            raise ValueError('Expected exact372 LoRA and4 head trainable tensors')
        parameters=[p for name,p in named]
        before=[p.detach().cpu().clone() for p in parameters]
        head_mask=torch.cat([torch.full((p.numel(),),name.startswith('head.'),dtype=torch.bool) for name,p in named])
        vectors={};losses={};none_counts={};blocks=[{},{}]
        for source,rows in selected.items():
            vectors[source]=torch.zeros(head_mask.numel(),dtype=torch.float32)
            block_sums=[torch.zeros_like(vectors[source]),torch.zeros_like(vectors[source])]
            block_counts=[0,0];source_losses=[];unused=Counter()
            for index,row in enumerate(rows):
                if row['questions']['action']['src']!=source:raise ValueError('Source changed')
                record=annotate(materialize(row),row)
                encoded=encode(client.runtime.tok,record,max_state=4096,max_branch=8192,strict=True,
                               option_isolation=model.option_isolation)
                if len(encoded['ids'])>8192 or encoded['state_truncated']:raise ValueError('No truncation admitted')
                logits=model(encoded)
                if len(logits)!=1:raise ValueError('One action question required')
                loss=parts(logits[0],record['questions'][0],'cuda')['objective']
                gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
                flat=torch.cat([torch.zeros_like(p,device='cpu',dtype=torch.float32).flatten()
                    if g is None else g.detach().float().cpu().flatten()
                    for p,g in zip(parameters,gradients)])
                if not torch.isfinite(flat).all() or not torch.isfinite(loss):raise ValueError('Nonfinite actual gradient')
                for (name,p),g in zip(named,gradients):
                    if g is None:unused[name]+=1
                vectors[source]+=flat;block_sums[index%2]+=flat;block_counts[index%2]+=1
                source_losses.append(float(loss.detach()))
                del logits,loss,gradients,flat
            vectors[source]/=len(rows)
            for block in (0,1):
                if not block_counts[block]:raise ValueError('Two nonempty fixed family blocks required')
                blocks[block][source]=block_sums[block]/block_counts[block]
            losses[source]=sum(source_losses)/len(source_losses);none_counts[source]=dict(unused)
            print(json.dumps({'checkpoint_step':step,'source':source,'samples':len(rows),
                              'seconds':time.monotonic()-started}),flush=True)
        unchanged=all(torch.equal(p.detach().cpu(),old) for p,old in zip(parameters,before))
        if not unchanged or any(p.grad is not None for p in parameters) or bundle(path)!=spec:
            raise ValueError('Read-only probe modified parameters or checkpoint')
        result[step]={'parameter_tensors':len(named),'parameter_elements':head_mask.numel(),
            'head_elements':int(head_mask.sum()),'dropout_disabled':True,'actual_parameters_unchanged':True,
            'sample_mean_objectives':losses,'unused_gradient_counts':none_counts,
            'all_parameters':geometry(vectors,None),'head_only':geometry(vectors,head_mask),
            'LoRA_only':geometry(vectors,~head_mask),
            'fixed_two_block_all_parameter_geometry':[geometry(v,None) for v in blocks]}
        write(out/f'model-{step}-summary-private.json',result[step])
        del vectors,blocks,model,client,parameters,named,before,head_mask
        gc.collect();torch.cuda.empty_cache()
    report={'status':'actual_TRAIN_local_task_gradients_complete_not_optimizer_or_strength_proof',
        'models':result,'samples_per_source':reg['samples_per_source'],'allocation':allocation,
        'seconds':time.monotonic()-started,'student_updates':0,'new_games':0,
        'active_training_or_selection_changed':False,'uses_validation_gradients':False,
        'Adam_preconditioning_or_training_dropout_not_reproduced':True}
    write(out/'summary-private.json',report)
    files=dict(reg['files']);files[str(RUN/'registered-private.json')]=sha(RUN/'registered-private.json')
    for p in out.glob('*-private.json'):files[str(p)]=sha(p)
    write(out/'completion-binding-private.json',{'files':files})
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],
        'samples_per_model':reg['total_samples_per_model'],'student_updates':0,'new_games':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run'),required=True)
    args=parser.parse_args()
    prepare() if args.phase=='prepare' else run()
