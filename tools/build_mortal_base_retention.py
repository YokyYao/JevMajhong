"""Prepare terminal-outcome retention for the audited fresh-base lineage.

No old v8/128 prior, soft target, checkpoint or optimizer is inherited. The
reused terminal outcomes still depend on those old continuation policies.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_mortal_teacher import registration,require_gpu,sha,verify


def configuration(path):
    overlay=json.loads(Path(path).read_text())
    c=json.loads((ROOT/overlay['foundation_config']).read_text())
    c['foundation_run']=c['out'];c.update(overlay)
    c['no_old_checkpoint_initialization']=False
    c['initialization']='audited_fresh_base_foundation_only'
    if (sum(c['replay_quotas'].values())!=2048 or sum(c['batch_quotas'].values())!=48
            or c['teacher_loss_fraction']!=.85 or c['minimum_observation_updates']!=128
            or c['max_updates']!=384 or c['validation_every']!=32
            or c['raw_publication'] or c['automatic_export']):
        raise ValueError('Unregistered fresh-lineage retention recipe')
    return c


def select_replay(rows,quotas,cap):
    selected=[];groups=Counter();used=set()
    for source in ('teacher_kan','teacher_riichi_yes','teacher_riichi_no','teacher_call_yes',
                   'teacher_call_no','teacher_defense','teacher_plain'):
        pool=sorted((r for r in rows if r['questions']['action']['src']==source),
            key=lambda r:hashlib.sha256((r['_meta']['id']+':retention-replay-v1').encode()).hexdigest())
        take=[]
        for r in pool:
            rid=r['_meta']['id'];group=r['_meta']['group_id']
            if rid not in used and groups[group]<cap:
                take.append(copy.deepcopy(r));used.add(rid);groups[group]+=1
                if len(take)==quotas[source]:break
        if len(take)!=quotas[source]:raise ValueError('Replay quota shortfall: '+source)
        selected+=take
    return selected


def terminal_target(label,prior,eligible):
    """Derive targets only from raw outcomes and the fresh foundation's prior."""
    from tools.train_targeted_endgame import validate_world_trace
    from jevmahjong.endgame.targeted_refresh import reliable_target
    row=label['source'];keys=list(row['request']['questions']['action']['criteria'])
    if set(keys)!=set(prior) or set(label['outcomes'])!=set(keys):
        raise ValueError('Terminal action support or fresh prior differs')
    if label['worlds_per_action']!=64 or not label['all_actions_completed']:
        raise ValueError('Every action requires complete paired64 outcomes')
    utility={};escape={}
    for key in keys:
        values=label['outcomes'][key];validate_world_trace(values)
        utility[key]=[int(v['ranks'][row['seat']]<=2) for v in values]
        escape[key]=sum(v['ranks'][row['seat']]<4 for v in values)/64
    q,details=reliable_target(prior,utility,eligible)
    return q,details,{k:sum(v)/64 for k,v in utility.items()},escape


def isolated(train,validation):
    if ({r['_meta']['id'] for r in train}&{r['_meta']['id'] for r in validation}
            or {r['_meta']['group_id'] for r in train}&{r['_meta']['group_id'] for r in validation}):
        raise ValueError('Whole source game crosses training/validation')


def source_registration(c):
    return registration(c,('configs/training/mortal-base-retention-v1.json',c['foundation_config'],
        'tools/build_mortal_base_retention.py','tools/train_mortal_base_retention.py',
        'tools/train_mortal_base.py','tools/train_mortal_teacher.py','tools/train_targeted_endgame.py',
        'tools/continue_endgame_fourth_lora.py','tools/audit_mortal_teacher_run.py',
        'tests/core/test_mortal_base_retention.py','docs/experiments/plans/mortal-base-retention-20261120.md'))


def prepare(c):
    import torch
    from jevmahjong.endgame.collect import write,digest
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    from jevmahjong.endgame.targeted_refresh import balanced_batches
    from tools.train_mortal_base import read_rows
    from tools.continue_endgame_fourth_lora import normal_predictions
    from jevany.data import materialize
    foundation=ROOT/c['foundation_run'];out=ROOT/c['out'];out.mkdir(parents=True,exist_ok=True)
    if any((out/p).exists() for p in ('data-ready-private.json','train.jsonl','validation.jsonl','training-run')):
        raise ValueError('Inspect existing admitted or partial retention data before retrying')
    public_path=out/'public-code-identity-private.json'
    public=json.loads(public_path.read_text());ci=public['ci']
    required={str(ROOT/p) for p in ('configs/training/mortal-base-retention-v1.json',
        'tools/build_mortal_base_retention.py','tools/train_mortal_base_retention.py',
        'tests/core/test_mortal_base_retention.py','docs/experiments/plans/mortal-base-retention-20261120.md')}
    if (set(public['files'])!=required or ci['head_sha']!=public['commit']
            or any(ci[k]!='success' for k in ('conclusion','core','desktop'))):
        raise ValueError('Require published matching source and successful core/desktop CI')
    verify(public)
    final=json.loads((foundation/'final-evidence-private.json').read_text())
    fit=json.loads((foundation/'fit-result-private.json').read_text())
    pipeline=json.loads((foundation/'pipeline-exit-private.json').read_text())
    if (final['status']!='fresh_base_final_evidence_audit_passed' or pipeline['exit_code']!=0
            or final['selected_update']!=fit['selected_update']):raise ValueError('Foundation not fully audited')
    verify(json.loads((foundation/'training-protocol-private.json').read_text()))
    data=json.loads((foundation/'data-ready-private.json').read_text());verify({'files':data['files']})
    step=fit['selected_update'];parent=foundation/'training-run/checkpoints'/f'step-{step:06d}'
    identity=bundle_identity(parent)
    audit=json.loads((foundation/'audit-private.json').read_text())
    if identity!=next(p['identity'] for p in audit['audits'] if p['update']==step):raise ValueError('Foundation identity changed')
    meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True)
    if not(meta.get('base_only_initialization') and meta['policy_version']=='mortal-base-v1'
           and meta['optimizer_step']==step):raise ValueError('Require the newly trained base lineage')
    readiness=json.loads((out/'raw-outcome-audit-private.json').read_text());verify({'files':readiness['source_files']})
    if readiness['train_positions']!=48 or readiness['monitor_positions']!=8:raise ValueError('Unexpected endgame membership')
    old=ROOT/c['retention_root'];manifest=json.loads((old/'manifest-private.json').read_text())
    old_ready=json.loads((old/'data-ready-private.json').read_text())
    if digest(manifest['positions'])!=manifest['positions_sha256']:raise ValueError('Endgame manifest changed')
    requests=[]
    for row in manifest['positions']:
        request=copy.deepcopy(row['request']);request['_meta']=dict(id='base-retention:'+row['id'])
        requests.append(request)
    priors=normal_predictions(parent,requests)
    if bundle_identity(parent)!=identity:raise ValueError('Prior inference modified the foundation')
    teacher_train=read_rows(foundation/'train.jsonl');teacher_val=read_rows(foundation/'validation.jsonl')
    train=select_replay(teacher_train,c['replay_quotas'],c['replay_group_cap']);val=copy.deepcopy(teacher_val)
    source_files={str(foundation/p):sha(foundation/p) for p in
        ('final-evidence-private.json','fit-result-private.json','pipeline-exit-private.json',
         'audit-private.json','comparison-games-private.jsonl','comparison-result-private.json','data-ready-private.json')}
    source_files[str(public_path)]=sha(public_path)
    for row in manifest['positions']:
        path=old/'labels'/(row['id'].rsplit(':',1)[1]+'.json');source_files[str(path)]=sha(path)
        if source_files[str(path)]!=old_ready['labels_sha256'][row['id']]:raise ValueError('Raw terminal outcome changed')
        label=json.loads(path.read_text())
        if label['source']!=row:raise ValueError('Endgame source mismatch')
        request=copy.deepcopy(row['request']);rid='base-retention:'+row['id']
        target,details,top2,escape=terminal_target(label,priors[rid],fourth_chase_action_keys(request))
        request['questions']['action'].update(src='retention_outcome',target=target,label=max(target,key=target.get))
        request['_meta']=dict(id=rid,group_id=row['group_id'],split=row['split'],category=row['category'],
            source='terminal_outcomes_with_fresh_foundation_prior',source_initial_placement=row['rank'],
            fresh_top2=top2,fresh_escape=escape,paired_uncertainty=details,
            old_continuation_dependency=True,not_mortal_teacher_label=True)
        (train if row['split']=='train' else val).append(request)
    if len(train)!=2096 or len(val)!=520:raise ValueError('Fixed2096/520 membership differs')
    isolated(train,val)
    for r in train+val:materialize(r)
    weights={k:v*c['teacher_loss_fraction'] for k,v in data['loss_mass'].items()}
    weights['retention_outcome']=1-c['teacher_loss_fraction']
    schedule=balanced_batches(train,c['batch_quotas'],c['max_updates'],seed=c['seed'])
    if set(i for batch in schedule[:c['minimum_observation_updates']] for i in batch)!=set(range(len(train))):
        raise ValueError('Observation floor misses admitted training examples')
    for split,rows in (('train',train),('validation',val)):
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    proof=source_registration(c);verify(proof);verify({'files':source_files});verify({'files':readiness['source_files']})
    write(out/'registered-data-code-private.json',proof)
    result=dict(status='fresh_lineage_retention_data_ready',parent_checkpoint=str(parent),parent_step=step,
        parent_identity=identity,train=len(train),validation=len(val),teacher_replay=2048,retention_train=48,
        teacher_validation=512,retention_monitor=8,loss_mass=weights,
        counts=dict(Counter(r['questions']['action']['src'] for r in train)),
        files={str(out/(s+'.jsonl')):sha(out/(s+'.jsonl')) for s in ('train','validation')},
        source_files=source_files,fresh_foundation_prior=True,no_old_prior_or_soft_targets=True,
        old_continuation_outcomes_reused=True,raw_publication=False,release_qualified=False)
    write(out/'data-ready-private.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','source_files','parent_identity')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_GPUS') and os.environ.get('SLURM_STEP_GPUS'):
        os.environ['SLURM_JOB_GPUS']=os.environ['SLURM_STEP_GPUS']
    require_gpu();c=configuration(a.config);os.chdir(ROOT);sys.path.insert(0,c['upstream_source']);prepare(c)
