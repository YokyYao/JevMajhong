"""Freeze an explicit, bounded-repeat phase mixture from a completed private pool."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from jevmahjong.core.io import digest,write


def kan_route(record):
    kinds={c['type'] for c in record['questions']['action']['criteria'].values()}
    return next((k for k in ('kakan','ankan','daiminkan') if k in kinds),None)


def choose(rows,count,repeat_cap,seed):
    if not isinstance(count,int) or count<1 or not isinstance(repeat_cap,int) or repeat_cap<1:
        raise ValueError('Positive mixture count and repeat cap required')
    if not rows or count>len(rows)*repeat_cap:raise ValueError('Insufficient independent coverage for bounded mixture')
    ranked=sorted(rows,key=lambda r:hashlib.sha256((str(seed)+'|'+r[0]).encode()).hexdigest())
    return [ranked[i%len(ranked)] for i in range(count)]


def prepare(cfg):
    pool=Path(cfg['pool']);file=pool/'pool-private.sqlite'
    if digest(file)!=cfg['pool_sha256'] or digest(pool/'result.json')!=cfg['pool_result_sha256']:
        raise ValueError('Registered raw pool changed')
    receipt=json.loads((pool/'result.json').read_text())
    if receipt['status']!='raw_request_pool_ready_not_yet_balanced' or not receipt['requires_masked_soft_target_loss']:
        raise ValueError('Completed masked raw pool required')
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect('file:'+str(file.resolve())+'?mode=ro',uri=True)
    summaries=[];groups={};identities={};files={};manifest=[]
    for split in ('train','validation'):
        selected=[];used=set();groups[split]=set();identities[split]=set()
        for index,spec in enumerate(cfg['mixture'][split]):
            where=['split=?','phase=?','label_type=?'];values=[split,spec['phase'],spec['label_type']]
            if 'choice' in spec:
                if spec['label_type']!='expert_replay':raise ValueError('Never quota future samples by their outcome/argmax label')
                where.append('choice=?');values.append(spec['choice'])
            rows=db.execute('SELECT position,payload FROM cases WHERE '+' AND '.join(where),values).fetchall()
            if spec.get('kan_kind'):rows=[r for r in rows if kan_route(json.loads(r[1]))==spec['kan_kind']]
            cap=spec.get('max_repeats',1)
            if split=='validation' and cap!=1:raise ValueError('Validation positions must stay unique')
            picked=choose(rows,spec['count'],cap,str(cfg['seed'])+':'+split+':'+str(index))
            positions={p for p,_ in picked}
            if used&positions:raise ValueError('Overlapping mixture strata')
            used.update(positions)
            for repeat,(pos,payload) in enumerate(picked):
                record=json.loads(payload);record['_meta']['id']+=':'+split+':'+str(index)+':'+str(repeat)
                record['_meta']['split']=split
                selected.append(record);groups[split].add(record['_meta']['group_id']);identities[split].add(pos)
                manifest.append(dict(split=split,position=pos,record_id=record['_meta']['id']))
            summaries.append(dict(split=split,**spec,available_unique_positions=len(rows),
                selected_unique_positions=len(positions),effective_max_repetition=max(Counter(p for p,_ in picked).values())))
        selected.sort(key=lambda r:hashlib.sha256((str(cfg['seed'])+r['_meta']['id']).encode()).hexdigest())
        path=out/(split+'.jsonl')
        with path.open('w') as stream:
            for r in selected:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        files[split]=dict(path=str(path.resolve()),sha256=digest(path),records=len(selected),
            groups=len(groups[split]),unique_positions=len(identities[split]))
    db.close()
    if groups['train']&groups['validation'] or identities['train']&identities['validation']:
        raise ValueError('Selected whole-group partitions overlap')
    if digest(file)!=cfg['pool_sha256']:raise ValueError('Pool changed during materialization')
    write(out/'selection-private.json',manifest)
    result=dict(status='balanced_raw_data_ready',config=cfg,files=files,strata=summaries,
        train_validation_group_overlap=0,train_validation_position_overlap=0,
        requires_masked_soft_target_loss=True,expert_labels_changed=False,unknown_actions_are_not_negative_labels=True,
        reward_sign_filtering=False,test_payloads_opened=False,raw_records_public=False,optimizer_updates=0)
    write(out/'result.json',result);write(out/'status.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    result=prepare(json.loads(p.parse_args().config.read_text()));print(json.dumps({k:v for k,v in result.items() if k!='config'}))


if __name__=='__main__':main()
