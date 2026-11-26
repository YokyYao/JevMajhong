"""Prospective RI replay exposure schedule; no live recipe or weight changes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_progression_parent_loss_v1 import MASS

RI=tuple(sorted(s for s in MASS if s.endswith((':riichi_yes',':riichi_no'))))
METHOD='four_RI_sources_each_update_inverse_repeat_weighted_replay_v1'


def build(categories,baseline,seed,batch=73,max_repeat=8,max_weight=16.):
    n=len(categories);counts=Counter(categories)
    if (len(RI)!=4 or set(counts)!=set(MASS) or sorted(baseline)!=list(range(n))
            or type(batch) is not int or batch<=4 or type(seed) is not int):
        raise ValueError('Complete original TRAIN permutation and19 sources required')
    nonri=[i for i in baseline if categories[i] not in RI]
    width=batch-4;updates=math.ceil(len(nonri)/width);extra=updates*width-len(nonri)
    rng=random.Random(seed);duplicated=[]
    while len(duplicated)<extra:
        cycle=list(nonri);rng.shuffle(cycle);duplicated.extend(cycle[:extra-len(duplicated)])
    expanded=[];original_subsequence=[];cursor=0
    for position,index in enumerate(nonri):
        expanded.append(index);original_subsequence.append(index)
        target=(position+1)*extra//len(nonri)
        while cursor<target:
            expanded.append(duplicated[cursor]);cursor+=1
    streams={}
    for source in RI:
        pool=[i for i in baseline if categories[i]==source]
        local=random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{source}'.encode()).digest()[:8],'big'))
        stream=[]
        while len(stream)<updates:
            cycle=list(pool);local.shuffle(cycle);stream.extend(cycle[:updates-len(stream)])
        streams[source]=stream
    order=[]
    for step in range(updates):
        group=[*expanded[step*width:(step+1)*width],*(streams[s][step] for s in RI)]
        rng.shuffle(group);order.extend(group)
    repeats=Counter(order);exposures=len(order)
    if (exposures!=updates*batch or set(repeats)!=set(range(n))
            or original_subsequence!=nonri or max(repeats.values())>max_repeat):
        raise ValueError('Every original row, fixed full batches and bounded repeats required')
    weights={i:exposures*MASS[categories[i]]/(counts[categories[i]]*repeats[i]) for i in repeats}
    if max(weights.values())>max_weight or any(not math.isfinite(w) or w<=0 for w in weights.values()):
        raise ValueError('Positive inverse-repeat weights must satisfy the original cap')
    error=max(abs(weights[i]*repeats[i]/exposures-MASS[categories[i]]/counts[categories[i]]) for i in range(n))
    if error>1e-12:raise ValueError('Per-original-row empirical objective mass changed')
    old_weights={s:n*MASS[s]/counts[s] for s in counts}
    comparison={}
    for source in RI:
        old=[];new=[];old_counts=[]
        for start in range(0,n,batch):
            group=baseline[start:start+batch];k=sum(categories[i]==source for i in group)
            old_counts.append(k);old.append(k*old_weights[source]/len(group))
        for start in range(0,exposures,batch):
            group=order[start:start+batch]
            if sum(categories[i]==source for i in group)!=1:raise ValueError('One row of each RI source per update required')
            new.append(sum(weights[i] for i in group if categories[i]==source)/batch)
        def describe(values):
            return {'mean':statistics.mean(values),'std':statistics.pstdev(values),'min':min(values),'max':max(values)}
        comparison[source]={'original_updates_missing_source':sum(k==0 for k in old_counts),
            'candidate_updates_missing_source':0,'original_effective_mass':describe(old),
            'candidate_effective_mass':describe(new),
            'mass_std_reduced':statistics.pstdev(new)<statistics.pstdev(old)}
    return {'indices':order,'weights_by_original_index':{str(i):weights[i] for i in range(n)},
        'nonRI_original_subsequence':original_subsequence,
        'summary':{'method':METHOD,'unique_TRAIN_rows':n,'physical_exposures':exposures,
            'optimizer_updates':updates,'batch':batch,'additional_exposures':exposures-n,
            'nonRI_inserted_repeats':extra,'max_repeat':max(repeats.values()),
            'max_example_weight':max(weights.values()),'maximum_original_row_mass_error':error,
            'unique_source_counts':dict(counts),'physical_source_counts':dict(Counter(categories[i] for i in order)),
            'nonRI_original_order_preserved':True,'RI_update_mass_comparison':comparison,
            'per_original_empirical_objective_preserved':True,'optimization_trajectory_equivalence_claimed':False}}


def prepare():
    from tools.capture_curriculum_paired_terminal_v2 import resource
    from tools.curriculum_transfer import read,sha
    from tools.curriculum_calls_post_reach_data_v1 import write
    from tools.curriculum_progression_paired_v1 import paired_epoch
    allocation=resource();parent=ROOT/'runs/curriculum-progression-parent-retention-v1'
    registration=parent/'fit-plan/registered.json';reg=read(registration)
    out=ROOT/'runs/curriculum-ri-every-update-replay-v1'
    if out.exists():raise ValueError('Never overwrite a prospective data arrangement')
    receipt_path=Path(reg['training_config']['data_receipt']);receipt=read(receipt_path)
    files={str(p):sha(p) for p in (Path(__file__).resolve(),registration,receipt_path,
        ROOT/'tests/core/test_curriculum_ri_every_update_replay_v1.py',
        ROOT/'docs/experiments/plans/curriculum-ri-every-update-replay-20261125.md')}
    pools={}
    for split in ('train','validation'):
        spec=receipt['files'][split];path=Path(spec['path'])
        if sha(path)!=spec['sha256']:raise ValueError('Original frozen corpus changed')
        files[str(path)]=spec['sha256']
        with path.open() as stream:pools[split]=[json.loads(line) for line in stream]
        if len(pools[split])!=spec['records']:raise ValueError('Original complete split required')
    train=pools['train'];validation=pools['validation']
    if len(train)!=87595 or len(validation)!=23688:raise ValueError('Fixed original union required')
    if {r['_meta']['group_id'] for r in train}&{r['_meta']['group_id'] for r in validation}:
        raise ValueError('TRAIN/validation family overlap')
    categories=[r['questions']['action']['src'] for r in train]
    baseline_rng=random.Random(reg['curriculum']['seed']);plans=[]
    for epoch in (1,2):
        baseline,base_receipt=paired_epoch(categories,baseline_rng,epoch)
        expected={k:v for k,v in base_receipt.items() if k!='source_counts_by_update'}
        if expected!=reg['expected_orders'][epoch-1]:raise ValueError('Actual baseline order reconstruction differs')
        plan=build(categories,baseline,2026112501+epoch)
        plan['summary'].update(epoch=epoch,baseline_order_sha256=base_receipt['order_sha256'])
        plans.append(plan)
    out.mkdir(exist_ok=False)
    for epoch,plan in enumerate(plans,1):
        path=out/f'epoch-{epoch:02d}-private.json';write(path,plan);files[str(path)]=sha(path)
    report={'status':'prospective_TRAIN_replay_arrangement_prepared_not_registered_or_trained',
        'epochs':[p['summary'] for p in plans],'total_physical_exposures':sum(p['summary']['physical_exposures'] for p in plans),
        'total_optimizer_updates':sum(p['summary']['optimizer_updates'] for p in plans),
        'validation_unchanged':True,'no_validation_examples_oversampled':True,
        'no_label_or_target_changed':True,'active_trial_not_changed':True,
        'model_loaded':False,'student_updates':0,'new_games':0,'allocation':allocation,
        'parent_selection_and_training_integration_pending':True}
    write(out/'summary-private.json',report);files[str(out/'summary-private.json')]=sha(out/'summary-private.json')
    write(out/'completion-binding-private.json',{'files':files})
    print(json.dumps({k:v for k,v in report.items() if k not in ('allocation','epochs')}),flush=True)
    print(json.dumps({'epoch_summaries':[p['summary'] for p in plans]}),flush=True)


if __name__=='__main__':prepare()
