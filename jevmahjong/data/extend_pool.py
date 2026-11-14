"""Append audited native future menus while preserving an existing raw pool."""
from collections import defaultdict,Counter
import copy
import json
from pathlib import Path
import sqlite3
from jevmahjong.core.io import digest,write
from .base_lora_pool import complete_target,normalized_request,position,phase
from .native_pairs import load_sources
from jevmahjong.training.provenance import verify as verify_protocol


def audited_records(spec,cfg,inputs):
    directory=Path(spec['path']).resolve();audit=Path(spec['audit_result']).resolve()
    protocol_file=audit.parent/'protocol.json'
    for file,checksum in [(audit,spec['audit_result_sha256']),(protocol_file,spec['audit_protocol_sha256'])]:
        if digest(file)!=checksum:raise ValueError('Bound future audit changed')
        inputs[str(file)]=checksum
    proof=json.loads(audit.read_text());audit_protocol=json.loads(protocol_file.read_text());verify_protocol(audit_protocol)
    root=Path(audit_protocol['config']['collection']).resolve()
    marker=json.loads((root/'collection-result.json').read_text())
    if (proof.get('status')!='passed_complete' or marker['status']!='collection_complete'
            or directory not in {Path(p).resolve() for p in marker['source_outputs']}):
        raise ValueError('Complete matching native future audit required')
    source=json.loads((directory/'protocol.json').read_text());verify_protocol(source)
    if (source.get('collection_method')!='paired_future_ev_v1' or not source['config'].get('expand_from')
            or source['model_files']!=audit_protocol['model_files']):
        raise ValueError('Audited all-alternative future source required')
    for name,checksum in audit_protocol['source_files'].items():
        if digest(Path(name))!=checksum:raise ValueError('Audited future input changed')
        inputs[name]=checksum
    for file in [root/'protocol.json',root/'collection-result.json',directory/'protocol.json',directory/'collection-result.json']:
        inputs[str(file)]=digest(file)
    parts,hashes,_=load_sources([directory],source['model_files'],cfg['reward'],
        expected_call_baseline=source['config'].get('call_context_baseline'))
    for name,checksum in hashes.items():
        if name in inputs and inputs[name]!=checksum:raise ValueError('Conflicting native source receipt')
        inputs[name]=checksum
    for split,rows in parts.items():
        groups=defaultdict(list)
        for original in rows:
            row=dict(original,_source_reward=cfg['reward'],_source_config=source['config'])
            groups[(row['episode_key'][0],row['episode_key'][1],row['request_sha256'])].append(row)
        for ordinal,items in enumerate(groups.values()):
            target,_=complete_target(items,cfg['reward'],cfg['target_temperature'],
                cfg['soft_target_mode']=='outcome_only_win_risk_compatible_masked_v3')
            req=copy.deepcopy(items[0]['public_request']);keys=list(req['questions']['action']['criteria'])
            req['questions']['action'].update(target=target,label=max(keys,key=target.__getitem__),src='paired_future_masked')
            if phase(req)!=spec['expected_phase']:raise ValueError('New source phase differs from registered plan')
            group='native:'+str(items[0]['episode_key'][0])
            yield ('validation' if split=='held' else 'train'),normalized_request(req,group,'paired_future_masked',ordinal)


def append_records(db,records,blocked,priority):
    """No replacement, no validation migration, no copies to fill quotas."""
    counts=Counter()
    parent_train={r[0] for r in db.execute("SELECT DISTINCT group_key FROM cases WHERE split='train'")}
    parent_held={r[0] for r in db.execute("SELECT DISTINCT group_key FROM cases WHERE split='validation'")}
    for split,record in records:
        if split not in ('train','validation'):raise ValueError('Declared source split required')
        group=record['_meta']['group_id'];pos=position(record,group)
        if split=='train' and group in blocked|parent_held:
            counts['historically_held_train_rows_blocked']+=1;continue
        if split=='validation' and group in parent_train:raise ValueError('New validation overlaps parent training')
        if db.execute('SELECT 1 FROM cases WHERE position=?',(pos,)).fetchone():raise ValueError('New source repeats an existing physical position')
        q=record['questions']['action'];choice=q['criteria'][q['label']]['type']
        db.execute('INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?)',(pos,split,group,phase(record),
            'paired_future_masked',choice,record['_meta']['view'],priority,json.dumps(record,ensure_ascii=False)))
        counts[split]+=1
    overlap=db.execute('SELECT count(*) FROM (SELECT group_key FROM cases GROUP BY group_key HAVING count(DISTINCT split)>1)').fetchone()[0]
    if overlap:raise ValueError('Native additions leak whole groups')
    return counts


def extend(cfg):
    from jevmahjong.training.utility import validate_config
    validate_config(cfg['reward'])
    if cfg['reward']['mode']!='own_win_first_v1':raise ValueError('Own-win objective required')
    if cfg['soft_target_mode'] not in ('outcome_only_win_compatible_masked_v2','outcome_only_win_risk_compatible_masked_v3'):
        raise ValueError('Registered soft target conversion required')
    parent=Path(cfg['parent_pool']);index=parent/'pool-private.sqlite';result_file=parent/'result.json'
    catalog=Path(cfg['historical_catalog_index'])
    inputs={str(index):cfg['parent_pool_sha256'],str(result_file):cfg['parent_result_sha256'],str(catalog):cfg['historical_catalog_sha256']}
    def verify():
        for file,h in inputs.items():
            if digest(Path(file))!=h:raise ValueError('Bound raw-pool input changed: '+file)
    verify();previous=json.loads(result_file.read_text())
    if previous['status']!='raw_request_pool_ready_not_yet_balanced' or not previous['requires_masked_soft_target_loss']:
        raise ValueError('Completed masked raw parent pool required')
    source=sqlite3.connect('file:'+str(catalog.resolve())+'?mode=ro',uri=True)
    blocked={r[0] for r in source.execute("SELECT DISTINCT group_key FROM groups WHERE split!='train'")};source.close()
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    source=sqlite3.connect('file:'+str(index.resolve())+'?mode=ro',uri=True)
    db=sqlite3.connect(out/'pool-private.sqlite');source.backup(db);source.close();counts=Counter()
    for spec in cfg['sources']:
        counts.update(append_records(db,audited_records(spec,cfg,inputs),blocked,spec['priority']))
    db.execute('ATTACH DATABASE ? AS original',('file:'+str(index.resolve())+'?mode=ro',))
    if db.execute('SELECT count(*) FROM (SELECT * FROM original.cases EXCEPT SELECT * FROM cases)').fetchone()[0]:
        raise ValueError('Extension changed original records or validation')
    strata=[dict(split=s,phase=p,label_type=l,choice=c,positions=n) for s,p,l,c,n in db.execute(
        'SELECT split,phase,label_type,choice,count(*) FROM cases GROUP BY split,phase,label_type,choice')]
    db.commit();db.close();verify()
    result=dict(previous,config=dict(sources=previous['config']['sources']+cfg['sources'],extension=cfg),
        pool_sha256=digest(out/'pool-private.sqlite'),strata=strata,groups_overlap=0,
        extension=dict(counts,original_records_and_validation_preserved=True,new_sources_independently_audited=True,
            cached_encoder_features_used=False,optimizer_updates=0))
    write(out/'inputs-private.json',inputs);write(out/'result.json',result);write(out/'status.json',result)
    return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    result=extend(json.loads(p.parse_args().config.read_text()));print(json.dumps(result['extension']))


if __name__=='__main__':main()
