"""Teacher-only opportunity-balanced data for adaptation from a fresh base."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def scope(row):
    q=row['questions']['action'];kinds={v['type'] for v in q['criteria'].values()}
    chosen=q['criteria'][row['_meta']['teacher_action']]['type']
    if 'hora' in kinds: raise ValueError('This source pool must retain its no-win-menu scope')
    if kinds&{'ankan','daiminkan','kakan'}: return 'teacher_kan'
    if 'reach' in kinds: return 'teacher_riichi_yes' if chosen=='reach' else 'teacher_riichi_no'
    if kinds&{'chi','pon'}: return 'teacher_call_yes' if chosen in ('chi','pon') else 'teacher_call_no'
    if any(p!=row['state']['seat'] for p in row['state'].get('riichi',[])): return 'teacher_defense'
    return 'teacher_plain'


def family_weights(counts,families):
    result={k:families[k] for k in ('teacher_plain','teacher_defense','teacher_kan')}
    for family in ('riichi','call'):
        keys=['teacher_'+family+'_'+v for v in ('yes','no')];total=sum(counts[k] for k in keys)
        if not total or any(counts[k]<=0 for k in keys): raise ValueError('Missing select/decline opportunity coverage')
        for k in keys: result[k]=families[family]*counts[k]/total
    if abs(sum(result.values())-1.)>1e-12:raise ValueError('Loss family mass is not normalized')
    return result


def select_validation(rows,quotas,cap):
    groups=Counter();selected=[];used=set()
    def key(r):return hashlib.sha256((r['_meta']['id']+':base-heldout-v1').encode()).hexdigest()
    for source in ('teacher_kan','teacher_riichi_yes','teacher_riichi_no','teacher_call_yes',
                   'teacher_call_no','teacher_defense','teacher_plain'):
        pool=sorted((r for r in rows if r['questions']['action']['src']==source),key=key)
        take=[]
        if source=='teacher_kan':
            for kind in ('ankan','daiminkan','kakan'):
                available=[r for r in pool if kind in r['_meta']['kan_types']]
                for r in available:
                    if sum(kind in v['_meta']['kan_types'] for v in take)>=8:break
                    rid=r['_meta']['id'];gid=r['_meta']['group_id']
                    if rid not in used and groups[gid]<cap:
                        take.append(r);used.add(rid);groups[gid]+=1
                if sum(kind in v['_meta']['kan_types'] for v in take)<8:raise ValueError('Kan validation coverage shortfall')
        for r in pool:
            if len(take)==quotas[source]:break
            rid=r['_meta']['id'];gid=r['_meta']['group_id']
            if rid not in used and groups[gid]<cap:
                take.append(r);used.add(rid);groups[gid]+=1
        if len(take)!=quotas[source]:raise ValueError('Validation source quota shortfall: '+source)
        selected+=take
    return selected


def prepare(c):
    from tools.build_mortal_teacher import sha,verify,registration
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.targeted_refresh import balanced_batches
    from jevany.data import materialize
    donor=ROOT/c['teacher_source'];out=ROOT/c['out'];out.mkdir(parents=True,exist_ok=True)
    if (out/'data-ready-private.json').exists():raise ValueError('Existing prepared dataset must be inspected')
    verify(json.loads((donor/'registered-data-code-private.json').read_text()))
    proof=json.loads((donor/'data-ready-private.json').read_text())
    expected='bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8'
    if proof['teacher_model_sha256']!=expected:raise ValueError('Teacher weight identity differs')
    pools=defaultdict(list);source_hashes={};source_groups=defaultdict(set)
    paths=sorted((donor/'sources').glob('*.json'))
    if len(paths)!=512:raise ValueError('Expected512 completed teacher source games')
    for path in paths:
        s=json.loads(path.read_text());split=s['split'];source_groups[split].add(s['seed']);source_hashes[str(path)]=sha(path)
        if not s['completed']:raise ValueError('Incomplete teacher source game')
        for raw in s['pool']:
            r=copy.deepcopy(raw);meta=r['_meta'];q=r['questions']['action'];label=meta['teacher_action']
            if meta['teacher_checkpoint_sha256']!=expected or meta['split']!=split or label not in q['criteria']:
                raise ValueError('Public teacher action/source identity differs')
            category=scope(r);keys=list(q['criteria']);epsilon=c['label_smoothing']
            q.update(src=category,label=label,target={k:epsilon/len(keys)+(1-epsilon if k==label else 0.) for k in keys})
            meta.update(id='cold-base:'+meta['id'],teacher_original_scope=raw['questions']['action']['src'],
                        source='mortal_teacher_only_no_v8_targets')
            materialize(r);pools[split].append(r)
    if (len(source_groups['train'])!=384 or len(source_groups['validation'])!=128
            or source_groups['train']&source_groups['validation']):raise ValueError('Whole-game source split changed')
    if len(pools['train'])!=22321 or len(pools['validation'])!=7305:raise ValueError('Stored teacher menu population differs')
    train=pools['train'];val=select_validation(pools['validation'],c['validation_quotas'],c['validation_group_cap'])
    if {r['_meta']['group_id'] for r in train}&{r['_meta']['group_id'] for r in val}:raise ValueError('Source leakage')
    if len({r['_meta']['id'] for r in train+val})!=len(train)+len(val):raise ValueError('Duplicate identities')
    counts=Counter(r['questions']['action']['src'] for r in train)
    weights=family_weights(counts,c['family_mass'])
    observation=max(c['minimum_observation_updates'],
        math.ceil(max(math.ceil(counts[k]/c['batch_quotas'][k]) for k in counts)/c['validation_every'])*c['validation_every'])
    if observation>=c['max_updates']:raise ValueError('Insufficient budget for full source exposure')
    schedule=balanced_batches(train,c['batch_quotas'],c['max_updates'],seed=c['seed'])
    if set(i for b in schedule[:observation] for i in b)!=set(range(len(train))):raise ValueError('Observation misses examples')
    for split,rows in [('train',train),('validation',val)]:
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    registration_proof=registration(c,('tools/build_mortal_base.py','tests/core/test_mortal_base.py',
        'configs/training/mortal-base-v1.json','docs/experiments/plans/mortal-base-training-20261120.md'))
    write(out/'registered-data-code-private.json',registration_proof)
    result=dict(status='fresh_base_teacher_data_ready',train=len(train),validation=len(val),
        source_games={s:len(v) for s,v in source_groups.items()},train_counts=dict(counts),loss_mass=weights,
        validation_counts=dict(Counter(r['questions']['action']['src'] for r in val)),
        observation_updates=observation,no_v8_replay_targets=True,explicit_teacher_labels=True,
        teacher_sha256=expected,files={str(out/(s+'.jsonl')):sha(out/(s+'.jsonl')) for s in ('train','validation')},
        source_hashes=source_hashes,raw_publication=False)
    write(out/'data-ready-private.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','files')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    from tools.build_mortal_teacher import require_gpu
    require_gpu();os.chdir(ROOT);c=json.loads(a.config.read_text());sys.path.insert(0,c['upstream_source']);prepare(c)
