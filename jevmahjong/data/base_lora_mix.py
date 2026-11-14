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


def validation_reference(db,spec):
    """Keep previously frozen validation records byte-for-byte as a prefix."""
    from .base_lora_pool import position
    path=Path(spec['path'])
    if digest(path)!=spec['sha256']:raise ValueError('Frozen validation reference changed')
    text=path.read_text();records=[];positions=set()
    def normalized(record):
        value=copy.deepcopy(record);value['_meta'].pop('id',None);value['_meta'].pop('split',None)
        return value
    for line in text.splitlines():
        if not line.strip():continue
        record=json.loads(line);pos=position(record,record['_meta']['group_id'])
        if pos in positions:raise ValueError('Frozen validation reference repeats a position')
        saved=db.execute("SELECT payload FROM cases WHERE position=? AND split='validation'",(pos,)).fetchone()
        if saved is None or normalized(record)!=normalized(json.loads(saved[0])):
            raise ValueError('Frozen validation reference differs from held-out raw pool')
        positions.add(pos);records.append(record)
    if not records:raise ValueError('Nonempty frozen validation reference required')
    return dict(text=text,records=records,positions=positions,path=str(path.resolve()),sha256=spec['sha256'])


def prepare(cfg):
    pool=Path(cfg['pool']);file=pool/'pool-private.sqlite'
    if digest(file)!=cfg['pool_sha256'] or digest(pool/'result.json')!=cfg['pool_result_sha256']:
        raise ValueError('Registered raw pool changed')
    receipt=json.loads((pool/'result.json').read_text())
    if receipt['status']!='raw_request_pool_ready_not_yet_balanced' or not receipt['requires_masked_soft_target_loss']:
        raise ValueError('Completed masked raw pool required')
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect('file:'+str(file.resolve())+'?mode=ro',uri=True)
    summaries=[];groups={};identities={};files={};manifest=[];reference=None
    for split in ('train','validation'):
        selected=[];used=set();groups[split]=set();identities[split]=set()
        frozen_positions=set();frozen_count=0
        if split=='validation' and cfg.get('preserve_validation') is not None:
            from .base_lora_pool import position
            reference=validation_reference(db,cfg['preserve_validation'])
            selected=reference['records'];frozen_positions=reference['positions'];frozen_count=len(selected)
            used.update(frozen_positions);identities[split].update(frozen_positions)
            for r in selected:
                groups[split].add(r['_meta']['group_id'])
                manifest.append(dict(split=split,position=position(r,r['_meta']['group_id']),record_id=r['_meta']['id']))
            summaries.append(dict(split=split,frozen_reference=True,count=frozen_count,
                selected_unique_positions=frozen_count,effective_max_repetition=1))
        for index,spec in enumerate(cfg['mixture'][split]):
            where=['split=?','phase=?','label_type=?'];values=[split,spec['phase'],spec['label_type']]
            if 'choice' in spec:
                if spec['label_type']!='expert_replay':raise ValueError('Never quota future samples by their outcome/argmax label')
                where.append('choice=?');values.append(spec['choice'])
            rows=db.execute('SELECT position,payload FROM cases WHERE '+' AND '.join(where),values).fetchall()
            if frozen_positions:rows=[r for r in rows if r[0] not in frozen_positions]
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
        selected=selected[:frozen_count]+sorted(selected[frozen_count:],key=lambda r:hashlib.sha256((str(cfg['seed'])+r['_meta']['id']).encode()).hexdigest())
        if len({r['_meta']['id'] for r in selected})!=len(selected):raise ValueError('Prepared record IDs repeat')
        path=out/(split+'.jsonl')
        with path.open('w') as stream:
            if frozen_count:
                stream.write(reference['text'])
                if not reference['text'].endswith('\n'):stream.write('\n')
            for r in selected[frozen_count:]:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        files[split]=dict(path=str(path.resolve()),sha256=digest(path),records=len(selected),
            groups=len(groups[split]),unique_positions=len(identities[split]))
    db.close()
    if groups['train']&groups['validation'] or identities['train']&identities['validation']:
        raise ValueError('Selected whole-group partitions overlap')
    if digest(file)!=cfg['pool_sha256']:raise ValueError('Pool changed during materialization')
    if reference is not None and (digest(reference['path'])!=reference['sha256']
            or not (out/'validation.jsonl').read_bytes().startswith(Path(reference['path']).read_bytes())):
        raise ValueError('Frozen validation prefix changed during materialization')
    write(out/'selection-private.json',manifest)
    result=dict(status='balanced_raw_data_ready',config=cfg,files=files,strata=summaries,
        train_validation_group_overlap=0,train_validation_position_overlap=0,
        requires_masked_soft_target_loss=True,expert_labels_changed=False,unknown_actions_are_not_negative_labels=True,
        reward_sign_filtering=False,test_payloads_opened=False,raw_records_public=False,optimizer_updates=0)
    if reference is not None:result['fixed_validation_reference']=dict(path=reference['path'],sha256=reference['sha256'],
        records=len(reference['records']),original_bytes_preserved_as_prefix=True)
    write(out/'result.json',result);write(out/'status.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    result=prepare(json.loads(p.parse_args().config.read_text()));print(json.dumps({k:v for k,v in result.items() if k!='config'}))


if __name__=='__main__':main()
