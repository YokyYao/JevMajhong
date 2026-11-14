"""Build a private, quality-selected raw-request pool for fresh-base LoRA.

Cached encoder tensors are never training inputs here. Future outcomes are
used only to construct complete-menu soft targets, never serialized as state.
"""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from jevmahjong.core.io import digest,write

INSTRUCTION=('Choose the best legal action in four-player Mahjong Soul East-only riichi mahjong. '
    'Prioritize winning the current hand and winning sooner while avoiding deal-ins. '
    'Consider legal yaku, open or closed hand structure, scores, dealer, honba and riichi sticks. '
    'sh=shanten (lower is closer to tenpai); ukeire=visible remaining improving tiles; '
    'safe=opponent seats for which this tile is genbutsu. Probabilities are action preferences.')
STATE_KEYS={'seat','dealer','round_wind','honba','sticks','scores','hand','dora_indicators',
    'riichi','rivers','melds','discards_count','hand_number','tsumogiri_flags',
    'riichi_sutehais','last_tedashis','last_discard','screen_observation'}


def read(p):return json.loads(Path(p).read_text())
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def position(request,group):
    """Match views/feature revisions by observed tile position within a match."""
    s=request['state']
    fields=('seat','hand','rivers','melds','dora_indicators','discards_count','honba','sticks')
    return group+'|'+identity({k:s.get(k) for k in fields})


def phase(request):
    kinds={c['type'] for c in request['questions']['action']['criteria'].values()}
    if kinds&{'ankan','daiminkan','kakan'}:return 'kan_window'
    if kinds&{'chi','pon'}:return 'call_window'
    if 'dahai' in kinds:
        own=request['state']['melds'][request['state']['seat']]
        return 'open_discard' if any(m['type'] not in (3,'ankan') for m in own) else 'closed_discard'
    if 'hora' in kinds:return 'win_window'
    return 'other'


def normalized_request(request,group,label_type,ordinal):
    if not isinstance(request.get('state'),dict) or set(request['state'])-STATE_KEYS:
        raise ValueError('Only the shared public-state allowlist is permitted')
    if set(request.get('questions',{}))!={'action'}:raise ValueError('Only the public action question is permitted')
    result=copy.deepcopy(request);result['questions']['action']['instructions']=INSTRUCTION
    # Source annotations can contain outcome diagnostics; rebuild metadata.
    result['_meta']=dict(id=identity([group,position(result,group),label_type,ordinal]),
        group_id=group,source=label_type,variant='clean',phase=phase(result),label_type=label_type,
        view=request.get('_meta',{}).get('view','native'))
    from jevmahjong.policy.call_features import enrich_request
    return enrich_request(result)


def soft_target(credits,temperature):
    """Outcome-only win-compatible target; no inherited never-call prior."""
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive temperature required')
    values=[]
    for c in credits:
        u,w,s=(c[k] for k in ('utility','own_win','fast_reward'))
        if not all(math.isfinite(x) for x in (u,w,s)):raise ValueError('Finite target credits required')
        if u>0 and not (w>1e-12 or (abs(w)<=1e-12 and s>1e-12)):u=0.
        values.append(u/temperature)
    maximum=max(values);raw=[math.exp(x-maximum) for x in values];total=sum(raw)
    return [x/total for x in raw]


def complete_target(rows,reward,temperature):
    """Use every labelled action; unknown legal actions receive an explicit mask.

    Zero entries outside eligible mean UNLABELLED, not negative supervision.
    These records require the masked-soft-target training loss.
    """
    from jevmahjong.data.future_pairs import mean_credit
    first=rows[0];criteria=first['public_request']['questions']['action']['criteria'];keys=list(criteria)
    eligible=first['eligible'];teacher=first['teacher_action']
    if (not eligible or len(set(eligible))!=len(eligible) or teacher not in eligible
            or not set(eligible)<=set(range(len(keys)))):raise ValueError('Invalid labelled eligible menu')
    zero={k:0. for k in ('utility','own_win','fast_reward','dealin','net_points')};credits={teacher:zero};control=None
    for row in rows:
        if (row['public_request']!=first['public_request'] or row['eligible']!=eligible
                or row['teacher_action']!=teacher or row['action'] in credits):raise ValueError('Conflicting complete menu')
        if row['_source_reward']==reward:means=row['training_credit']
        else:means=mean_credit(row['counterfactual_worlds'],row,row['_source_config'],reward)
        if 'own_win' not in means['control']:raise ValueError('Actual own-win labels required')
        if control is not None and means['control']!=control:raise ValueError('Alternatives must share control futures')
        control=means['control']
        delta={k:means['branch'][k]-means['control'][k] for k in zero}
        credits[row['action']]=delta
    if set(credits)!=set(eligible):raise ValueError('Incomplete labelled future menu')
    probability=soft_target([credits[a] for a in eligible],temperature)
    target={k:0. for k in keys}
    for action,p in zip(eligible,probability):target[keys[action]]=p
    if any(p<=0 for p in probability):raise ValueError('Labelled target probability underflow')
    return target,[credits[a] for a in eligible]


def verified_prefix_source(dataset,provenance,inputs):
    """Only independently replayed raw expert data may use the new schema."""
    directory=Path(dataset['path']).resolve();train=directory/'train.jsonl';file=directory/'provenance.json'
    for name in dataset.get('audit_receipts',[]):
        result=Path(name);protocol=result.parent/'protocol.json'
        if digest(result)!=dataset['audit_sha256'][name]:raise ValueError('Expert prefix audit receipt changed')
        proof=read(result)
        if proof.get('audit_kind')!='observed_expert_kan_prefix_v1':continue
        if (proof.get('status')!='passed_complete' or not proof.get('eligible_for_pool_integration')
                or not proof.get('all_observed_eligible_windows_enumerated')
                or not proof.get('inputs_rebuilt_before_label') or not proof.get('alternate_shadow_seed')
                or not proof.get('censored_passes_excluded') or not proof.get('historical_held_groups_excluded')
                or proof.get('sealed_test_payloads_opened') is not False):
            raise ValueError('Complete observed prefix source audit required')
        if (proof['source_train_sha256']!=digest(train) or proof['source_provenance_sha256']!=digest(file)
                or proof['counts']['records']!=provenance['counts']['train']
                or proof['train_groups']!=len(provenance['matches']['train'])
                or provenance['matches']['validation'] or provenance['matches']['test']):
            raise ValueError('Audited expert train bytes or grouping differ')
        p=read(protocol)
        if Path(p['config']['producer_out']).resolve()!=directory:
            raise ValueError('Expert source audit targets a different directory')
        for path,checksum in p['source_files'].items():
            if digest(Path(path))!=checksum:raise ValueError('Audited expert source changed: '+path)
            if path in inputs and inputs[path]!=checksum:raise ValueError('Conflicting expert audit inputs')
            inputs[path]=checksum
        inputs[str(result)]=digest(result);inputs[str(protocol)]=digest(protocol)
        return
    raise ValueError('Complete observed prefix source audit required')


def prepare(cfg):
    import torch
    from jevmahjong.training.utility import validate_config
    validate_config(cfg['reward'])
    if cfg['reward']['mode']!='own_win_first_v1':raise ValueError('Registered own-win reward required')
    if cfg.get('soft_target_mode')!='outcome_only_win_compatible_masked_v2':raise ValueError('Explicit masked outcome-only target required')
    torch.set_num_threads(4)
    catalog=Path(cfg['catalog']);index=catalog/'index-private.sqlite'
    if (digest(index)!=cfg['catalog_index_sha256'] or digest(catalog/'result.json')!=cfg['catalog_result_sha256']
            or digest(catalog/'sources-private.json')!=cfg['catalog_sources_sha256']):
        raise ValueError('Registered completed catalog changed')
    sources=read(catalog/'sources-private.json');by_path={d['path']:d for d in sources}
    lookup=sqlite3.connect('file:'+str(index.resolve())+'?mode=ro',uri=True)
    blocked={g for (g,) in lookup.execute("SELECT DISTINCT group_key FROM groups WHERE split!='train'")};lookup.close()
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(out/'pool-private.sqlite')
    db.execute('CREATE TABLE cases (position TEXT,split TEXT,group_key TEXT,phase TEXT,label_type TEXT,choice TEXT,view TEXT,priority INTEGER,payload TEXT,PRIMARY KEY(position,split))')
    conflicts=set();inputs={str(index):cfg['catalog_index_sha256'],
        str(catalog/'result.json'):cfg['catalog_result_sha256'],
        str(catalog/'sources-private.json'):cfg['catalog_sources_sha256']};stats=Counter();origins=[]
    def insert(req,group,split,label_type,priority,ordinal):
        if split=='train' and group in blocked:stats['historically_held_train_rows_blocked']+=1;return
        if split=='validation' and group not in blocked:raise ValueError('Unregistered validation group')
        r=normalized_request(req,group,label_type,ordinal);q=r['questions']['action'];choice=q['criteria'][q['label']]['type']
        pos=position(r,group);signature=identity({k:v for k,v in q['criteria'][q['label']].items() if k in ('type','pai','consumed')})
        old=db.execute('SELECT priority,payload FROM cases WHERE position=? AND split=?',(pos,split)).fetchone()
        if old:
            previous=json.loads(old[1]);oldq=previous['questions']['action']
            oldsig=identity({k:v for k,v in oldq['criteria'][oldq['label']].items() if k in ('type','pai','consumed')})
            if label_type=='expert_replay' and oldsig!=signature:conflicts.add((pos,split));stats['conflicting_expert_views']+=1
            stats['duplicate_position_rows']+=1
            if old[0]>=priority:return
        db.execute('INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?,?)',
            (pos,split,group,r['_meta']['phase'],label_type,choice,r['_meta']['view'],priority,json.dumps(r,ensure_ascii=False)))
    for spec in cfg['sources']:
        d=by_path[str(Path(spec['path']).resolve())];priority=spec['priority']
        if d['catalog_status']!='indexed_not_training_admission':raise ValueError('Source is quarantined')
        if digest(Path(d['provenance']))!=d['provenance_sha256']:raise ValueError('Source provenance changed')
        inputs[d['provenance']]=d['provenance_sha256'];origins.append(dict(path=d['path'],kind=d['kind'],priority=priority))
        if d['kind']=='replay_jsonl':
            if not d['train_checksum_matches_provenance'] or not d['holdout_group_metadata_present']:
                raise ValueError('Verified replay checksums and historical split groups required')
            provenance=read(d['provenance'])
            prefix=provenance.get('schema')=='public-prefix-kan-v1'
            if prefix:
                if spec.get('validation'):raise ValueError('Observed prefix source is train-only')
                verified_prefix_source(d,provenance,inputs)
            for split in ('train','validation') if spec.get('validation') else ('train',):
                file=Path(d['path'])/(split+'.jsonl');checksum=digest(file)
                if provenance['files'][split]!=checksum:raise ValueError('Replay file changed')
                inputs[str(file)]=checksum
                with file.open() as stream:
                    for number,line in enumerate(stream):
                        if not line.strip():continue
                        r=json.loads(line);m=r['_meta'];schema=m.get('schema')
                        good=(schema=='clean-full-policy-v1' and m.get('call_repair_version')=='public-chi-pon-postcall-v2')
                        good|=(schema=='semantic-v3-reaction-only' and 'censored_or_unresolved_pass' in provenance.get('exclusions',{}))
                        good|=(prefix and schema=='public-prefix-kan-v1' and split=='train')
                        if not good:stats['unreviewed_replay_schema_rows_quarantined']+=1;continue
                        if m.get('split')!=split:raise ValueError('Replay split declaration differs')
                        if qlabel:=r['questions']['action'].get('label'):
                            if qlabel not in r['questions']['action']['criteria']:raise ValueError('Missing expert action')
                        else:raise ValueError('Missing expert label')
                        insert(r,'replay:'+str(m['group_id']),split,'expert_replay',priority,number)
        else:
            if not d['audit_receipts']:raise ValueError('Paired-future independent audit required')
            audited={}
            for name in d['audit_receipts']:
                result=Path(name);protocol=result.parent/'protocol.json'
                if digest(result)!=d['audit_sha256'][name] or read(result)['status']!='passed_complete':raise ValueError('Audit receipt changed')
                inputs[str(result)]=digest(result);inputs[str(protocol)]=digest(protocol)
                audited.update(read(protocol)['source_files'])
            p=read(d['provenance']);source_cfg=p['config'];start=source_cfg['training_seed_start']
            if not source_cfg.get('expand_from'):raise ValueError('Complete-option future data required')
            for path,checksum in d['shard_sha256'].items():
                seed=int(Path(path).stem);split='validation' if (seed-start)%4==3 else 'train'
                if split=='validation' and not spec.get('validation'):continue
                if audited.get(path)!=checksum or digest(Path(path))!=checksum:raise ValueError('Audited future shard changed')
                inputs[path]=checksum;payload=torch.load(path,map_location='cpu',weights_only=False);buckets=defaultdict(list)
                for raw in payload['rows']:
                    row=dict(raw);row.update(_source_reward=source_cfg['reward'],_source_config=source_cfg)
                    buckets[(row['episode_key'][0],row['episode_key'][1],row['request_sha256'])].append(row)
                for number,rows in enumerate(buckets.values()):
                    target,credits=complete_target(rows,cfg['reward'],cfg['target_temperature']);r=copy.deepcopy(rows[0]['public_request'])
                    keys=list(r['questions']['action']['criteria']);r['questions']['action'].update(target=target,label=max(keys,key=lambda k:target[k]),src='paired_future_masked')
                    stats['future_menu_states']+=1;stats['future_menu_negative_utility_options']+=sum(c['utility']<0 for c in credits)
                    if any(p==0 for p in target.values()):stats['future_states_with_unlabelled_legal_options_masked']+=1
                    group='native:'+str(seed)
                    insert(r,group,split,'paired_future_masked',priority,number)
                del payload;db.commit()
        db.commit();write(out/'status.json',dict(status='building_raw_request_pool',sources_complete=len(origins),sources_total=len(cfg['sources']),statistics=dict(stats)))
    for pos,split in conflicts:db.execute('DELETE FROM cases WHERE position=? AND split=?',(pos,split))
    rows=[dict(split=s,phase=p,label_type=l,choice=c,positions=n) for s,p,l,c,n in db.execute('SELECT split,phase,label_type,choice,count(*) FROM cases GROUP BY split,phase,label_type,choice')]
    overlaps=db.execute("SELECT count(*) FROM (SELECT group_key FROM cases GROUP BY group_key HAVING count(DISTINCT split)>1)").fetchone()[0]
    if overlaps:raise ValueError('Train/validation whole-group overlap')
    db.commit();db.close()
    for name,checksum in inputs.items():
        if digest(Path(name))!=checksum:raise ValueError('Pool source changed: '+name)
    write(out/'inputs-private.json',inputs);write(out/'origins-private.json',origins)
    result=dict(status='raw_request_pool_ready_not_yet_balanced',config=cfg,groups_overlap=overlaps,
        statistics=dict(stats),strata=rows,pool_sha256=digest(out/'pool-private.sqlite'),
        conflicting_expert_positions_quarantined=len(conflicts),cached_encoder_features_used=False,
        requires_masked_soft_target_loss=True,
        old_adapters_loaded=False,test_payloads_opened=False,raw_records_public=False,optimizer_updates=0)
    write(out/'result.json',result);write(out/'status.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    result=prepare(read(p.parse_args().config));print(json.dumps({k:v for k,v in result.items() if k!='config'}))


if __name__=='__main__':main()
