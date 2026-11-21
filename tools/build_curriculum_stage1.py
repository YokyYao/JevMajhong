"""Prepare a reproducible source-group holdout and clean stage1 from public data."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import socket

SOURCE_SHA={'train':'b462fbacdf2b4cbad9517d096bc1418638b2ddfed866060562c4f2dc2d2d295f',
            'validation':'1179bb4a26a6fe5b01cd13dcaeec46cd9d67adbc1b1deddb306c60f5d52d3540'}
SPLIT_SEED=2026112001
CONTEXT=('seat','dealer','round_wind','honba','sticks','scores','riichi','rivers','melds','hand_number')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def holdout(group):
    return int(hashlib.sha256(f'{SPLIT_SEED}:{group}'.encode()).hexdigest()[:16],16)%100<20

def context_gaps(row):
    s=row['state'];gaps=[k for k in CONTEXT if s.get(k) is None]
    for k in ('scores','rivers','melds'):
        if s.get(k) is not None and (len(s[k])!=4 or any(v is None for v in s[k])):gaps.append(k)
    return sorted(set(gaps))

def stratum(row):
    q=row['questions']['action'];criteria=q['criteria']
    if any(c['type']=='reach' for c in criteria.values()):
        return 'riichi_yes' if criteria[q['label']]['type']=='reach' else 'riichi_no'
    return 'plain'

def public_input(row):
    return dict(state=row['state'],questions={k:{a:b for a,b in q.items() if a in ('type','instructions','criteria')}
                                           for k,q in row['questions'].items()})

def save_rows(path,rows):
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    return dict(path=str(path),records=len(rows),sha256=sha(path),groups=len({r['_meta']['group_id'] for r in rows}))

def build(source,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),'Scheduled GPU required'
    pools={}
    for split,expected in SOURCE_SHA.items():
        path=source/f'{split}.jsonl'
        if sha(path)!=expected:raise ValueError('Pinned public source changed')
        pools[split]=[json.loads(s) for s in path.read_text().splitlines()]
    original_groups={k:{r['_meta']['group_id'] for r in v} for k,v in pools.items()}
    if original_groups['train']&original_groups['validation']:raise ValueError('Original groups overlap')
    reserved={g for g in original_groups['train'] if holdout(g)}
    by_split={
        'train':[r for r in pools['train'] if r['_meta']['group_id'] not in reserved],
        'validation':[r for r in pools['train'] if r['_meta']['group_id'] in reserved]+pools['validation']}
    clean={};excluded={};gaps={}
    for split,rows in by_split.items():
        gaps[split]=Counter(k for r in rows for k in context_gaps(r))
        clean[split]=[r for r in rows if not context_gaps(r)]
        excluded[split]=len(rows)-len(clean[split])
    out.mkdir(parents=True,exist_ok=False);stages={};strata={}
    for stage in ('stage1','stage2','stage3'):
        directory=out/stage;directory.mkdir();parts={}
        for split,pool in clean.items():
            chosen=[r for r in pool if stage=='stage3' or (r['_meta']['label_type']=='expert_replay'
                and (stage=='stage2' or r['_meta']['phase']=='closed_discard'))]
            if stage=='stage1':
                tagged=[];before=[]
                for r in chosen:
                    item=copy.deepcopy(r);category=stratum(r);item['questions']['action']['src']='curriculum1:'+category
                    if public_input(r)!=public_input(item):raise ValueError('Loss-only category leaked into visible input')
                    tagged.append(item)
                chosen=tagged;strata[split]=dict(Counter(stratum(r) for r in chosen))
            parts[split]=save_rows(directory/f'{split}.jsonl',chosen)
        stages[stage]=parts
    for kind in ('riichi_yes','riichi_no'):
        if strata['validation'].get(kind,0)<16:raise ValueError('Insufficient distinct riichi validation examples')
    n=sum(strata['train'].values());windows=strata['train']['riichi_yes']+strata['train']['riichi_no']
    mass={'plain':.8,'riichi_yes':.2*strata['train']['riichi_yes']/windows,
          'riichi_no':.2*strata['train']['riichi_no']/windows}
    weights={k:mass[k]*n/strata['train'][k] for k in mass}
    if max(weights.values())>8:raise ValueError('Loss weight exceeds registered cap8')
    train_ids={r['_meta']['id'] for r in clean['train']};val_ids={r['_meta']['id'] for r in clean['validation']}
    if train_ids&val_ids:raise ValueError('Decision identity overlap')
    train_groups={r['_meta']['group_id'] for r in clean['train']};val_groups={r['_meta']['group_id'] for r in clean['validation']}
    if train_groups&val_groups:raise ValueError('New source group overlap')
    save_rows(out/'common-validation.jsonl',pools['validation'])
    write(out/'reserved-public-groups.json',sorted(reserved))
    result=dict(status='curriculum_stage1_data_ready',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        source_sha256=SOURCE_SHA,split_seed=SPLIT_SEED,label_blind_holdout_fraction=.2,
        reserved_groups=len(reserved),groups_removed_from_all_future_stage_training=True,
        missing_context_rows_excluded=excluded,missing_context_fields={k:dict(v) for k,v in gaps.items()},
        train_validation_group_overlap=0,train_validation_id_overlap=0,stages=stages,stage1_strata=strata,
        stage1_loss_mass=mass,stage1_loss_weights=weights,loss_only_tags_do_not_change_public_input=True,
        copies_added=0,win_choice_supervision=False,legal_wins_executed_by_rule=True,
        simulated_labels_unchanged=True,stage4_admitted=False,
        note='Stage1 teaches discard/riichi. It does not claim learned win/pass or yaku scoring.')
    write(out/'data-ready.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.source,a.out)
