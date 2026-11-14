"""Inventory replay and native training data without opening sealed test payloads.

The SQLite index and source paths are private. Counts distinguish rows, public
requests and known physical positions; a future-world branch is not an expert
action label. This catalog does not itself authorize any source for training.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def discover(roots):
    """Use completed collection markers; directories alone are not datasets."""
    datasets={};audit_targets={}
    for root in map(Path,roots):
        for p in (root/'runs').glob('*/protocol.json'):
            j=read(p);c=j.get('config',{});target=c.get('collection');result=p.parent/'result.json'
            if target and result.exists() and read(result).get('status')=='passed_complete':
                t=Path(target);t=t if t.is_absolute() else root/t
                audit_targets.setdefault(str(t.resolve()),[]).append(str(result.resolve()))
        for p in (root/'data').glob('*/provenance.json'):
            directory=p.parent.resolve()
            if (directory/'train.jsonl').exists():
                datasets[str(directory)]=dict(kind='replay_jsonl',path=str(directory),workspace=str(root.resolve()),
                    provenance=str(p.resolve()),provenance_sha256=sha(p),complete=read(p).get('complete',True))
        paths=list((root/'runs').glob('*/source_*/protocol.json'))+list((root/'runs').glob('*/protocol.json'))
        for p in paths:
            directory=p.parent.resolve();j=read(p);method=j.get('collection_method')
            if not method or not (directory/'walls-private').is_dir():continue
            marker=directory/'collection-result.json'
            datasets[str(directory)]=dict(kind='native_pt',path=str(directory),workspace=str(root.resolve()),
                provenance=str(p.resolve()),provenance_sha256=sha(p),method=method,
                complete=marker.exists() and read(marker).get('status')=='collection_complete',
                declared_counts=read(marker) if marker.exists() else None)
    for d in datasets.values():
        d['audit_receipts']=audit_targets.get(d['path'],[])+audit_targets.get(str(Path(d['path']).parent),[])
        d['audit_sha256']={p:sha(p) for p in d['audit_receipts']}
    return sorted(datasets.values(),key=lambda d:d['path'])


def source_groups(provenance):
    result={}
    matches=provenance.get('matches',{})
    if isinstance(matches,dict):
        for split,ids in matches.items():
            if split in ('train','validation','test') and isinstance(ids,list):result[split]=ids
    for split in ('train','validation','test'):
        if isinstance(provenance.get(split+'_matches'),list):result[split]=provenance[split+'_matches']
    return result


def classify(request):
    state=request.get('state',{});q=request.get('questions',{}).get('action',{});criteria=q.get('criteria',{})
    if not isinstance(criteria,dict) or not criteria:return 'unknown',[],False,'missing_menu'
    kinds=sorted({c.get('type','unknown') for c in criteria.values()})
    valid=True;reason=''
    for c in criteria.values():
        typ=c.get('type');consumed=c.get('consumed',[])
        # MJAI kakan.consumed is the existing three-tile pon. The added
        # owned tile is kakan.pai, unlike the owned-consumption feature vector.
        needed={'chi':2,'pon':2,'ankan':4,'daiminkan':3,'kakan':3}.get(typ)
        if needed is not None and len(consumed)!=needed:valid=False;reason='consumed_count_requires_review'
        if typ=='kakan' and not isinstance(c.get('pai'),str):valid=False;reason='missing_added_kan_tile'
    if set(kinds)&{'ankan','daiminkan','kakan'}:phase='kan_window'
    elif set(kinds)&{'chi','pon'}:phase='call_window'
    elif 'dahai' in kinds:
        seat=state.get('seat');melds=state.get('melds')
        own=melds[seat] if isinstance(seat,int) and isinstance(melds,list) and 0<=seat<len(melds) else None
        phase='unknown_discard' if own is None else ('open_discard' if any(m.get('type') not in (3,'ankan') for m in own) else 'closed_discard')
    elif 'hora' in kinds:phase='win_window'
    else:phase='other'
    return phase,kinds,valid,reason


def public_identity(request):
    """Exclude labels, targets, annotations and instruction wording from identity."""
    return fingerprint(dict(state=request.get('state'),criteria=request.get('questions',{}).get('action',{}).get('criteria')))


def initialize(db):
    db.executescript('''
        CREATE TABLE groups (group_key TEXT, split TEXT, dataset INTEGER,
            PRIMARY KEY(group_key,split,dataset));
        CREATE TABLE entries (dataset INTEGER, split TEXT, group_key TEXT,
            position_key TEXT, request_hash TEXT, phase TEXT, view TEXT, kinds TEXT,
            observed_kind TEXT, label_type TEXT, schema_valid INTEGER, reason TEXT,
            locator TEXT);
        CREATE INDEX group_idx ON entries(group_key);
        CREATE INDEX request_idx ON entries(request_hash);
    ''')


def add(db,index,dataset,split,request,locator,*,native_row=None):
    meta=request.get('_meta',{});phase,kinds,valid,reason=classify(request)
    q=request.get('questions',{}).get('action',{});criteria=q.get('criteria',{})
    if native_row is None:
        group=meta.get('group_id');group='replay:'+str(group) if group is not None else None
        original=meta.get('parent_id',meta.get('id'))
        # These pipelines retain the replay decision ordinal before view suffixes.
        position=str(original).split(':',1)[0] if original is not None else None
        label=q.get('label');chosen=criteria.get(label,{}).get('type','missing')
        label_type='replay_action' if chosen!='missing' else 'unlabelled'
    else:
        key=native_row.get('episode_key',[])
        group='native:'+str(key[0]) if key else None
        position=':'.join(map(str,key[:3])) if len(key)>=3 else None
        action=native_row.get('action',native_row.get('teacher_action'))
        chosen=list(criteria.values())[action].get('type','unknown') if isinstance(action,int) and 0<=action<len(criteria) else 'missing'
        label_type='paired_future_branch_not_gold' if dataset['method']=='paired_future_ev_v1' else 'unlabelled_native'
    if group is None or position is None:valid=False;reason='missing_physical_group_identity'
    group=group or 'unidentified:'+fingerprint(locator);position=position or public_identity(request)
    db.execute('INSERT OR IGNORE INTO groups VALUES (?,?,?)',(group,split,index))
    db.execute('INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (index,split,group,group+'|'+position,public_identity(request),phase,meta.get('view','native'),
         json.dumps(kinds),chosen,label_type,int(valid),reason,json.dumps(locator)))


def scan_replay(db,index,dataset):
    root=Path(dataset['path']);p=read(dataset['provenance']);declared=source_groups(p)
    for split,ids in declared.items():
        db.executemany('INSERT OR IGNORE INTO groups VALUES (?,?,?)',
            [('replay:'+str(g),split,index) for g in ids])
    train=root/'train.jsonl';expected=p.get('files',{}).get('train');actual=sha(train)
    if isinstance(expected,str) and expected!=actual:raise ValueError('Replay training checksum changed: '+str(train))
    dataset['train_sha256']=actual;dataset['train_checksum_matches_provenance']=actual==expected
    dataset['validation_test_payloads_opened']=False
    dataset['declared_match_counts']={s:len(ids) for s,ids in declared.items()}
    dataset['holdout_group_metadata_present']=all(
        not (root/(split+'.jsonl')).exists() or (root/(split+'.jsonl')).stat().st_size==0
        or bool(declared.get(split)) for split in ('validation','test'))
    count=0
    with train.open('rb') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            if not line.strip():continue
            request=json.loads(line);split=request.get('_meta',{}).get('split','train')
            # A validation/test row stored in train.jsonl remains nontraining.
            if split not in ('train','validation','test'):split='unknown'
            add(db,index,dataset,split,request,dict(format='jsonl',file=str(train),byte_offset=offset))
            count+=1
    if sha(train)!=actual:raise ValueError('Replay input changed while indexing')
    dataset['indexed_train_file_rows']=count


def scan_native(db,index,dataset):
    import torch
    root=Path(dataset['path']);protocol=read(dataset['provenance']);cfg=protocol['config']
    start=cfg['training_seed_start'];seeds=cfg['training_seeds_per_stage']
    expected=set(range(start,start+seeds));files={};rows=0;held=0
    actual={int(p.stem):p for p in (root/'walls-private').glob('*.pt')}
    if set(actual)!=expected:raise ValueError('Native wall coverage differs: '+str(root))
    for seed,path in sorted(actual.items()):
        split='validation' if (seed-start)%4==3 else 'train'
        db.execute('INSERT OR IGNORE INTO groups VALUES (?,?,?)',('native:'+str(seed),split,index))
        checksum=sha(path);receipt=path.with_suffix('.audit.json')
        if not receipt.exists() or read(receipt).get('sha256')!=checksum:
            raise ValueError('Native shard checksum differs: '+str(path))
        files[str(path)]=checksum
        if split!='train':held+=1;continue
        payload=torch.load(path,map_location='cpu',weights_only=False)
        for ordinal,row in enumerate(payload['rows']):
            request=row.get('public_request')
            if not isinstance(request,dict):dataset['unsupported_rows']=dataset.get('unsupported_rows',0)+1;continue
            add(db,index,dataset,split,request,dict(format='native_pt',file=str(path),row=ordinal),native_row=row);rows+=1
        del payload
        if sha(path)!=checksum:raise ValueError('Native shard changed while indexing')
    dataset.update(indexed_train_rows=rows,held_shards_unopened=held,shard_sha256=files)


def run(config):
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False)
    datasets=discover(config['roots']);write(out/'discovery-private.json',datasets)
    db=sqlite3.connect(out/'index-private.sqlite');initialize(db);completed=[]
    for index,d in enumerate(datasets):
        # Historical holdouts remain blocked even when that dataset's train
        # payload is incomplete, invalid or quarantined below.
        provenance=read(d['provenance'])
        if d['kind']=='replay_jsonl':
            for split,ids in source_groups(provenance).items():
                db.executemany('INSERT OR IGNORE INTO groups VALUES (?,?,?)',
                    [('replay:'+str(g),split,index) for g in ids])
        else:
            cfg=provenance['config'];start=cfg.get('training_seed_start');n=cfg.get('training_seeds_per_stage')
            if isinstance(start,int) and isinstance(n,int):
                db.executemany('INSERT OR IGNORE INTO groups VALUES (?,?,?)',
                    [('native:'+str(s),'validation' if (s-start)%4==3 else 'train',index) for s in range(start,start+n)])
        db.commit()
        if not d['complete']:
            d['catalog_status']='quarantined_incomplete';completed.append(d);continue
        if sha(d['provenance'])!=d['provenance_sha256']:raise ValueError('Source provenance changed')
        try:
            db.execute('SAVEPOINT dataset_scan')
            if d['kind']=='replay_jsonl':scan_replay(db,index,d)
            else:scan_native(db,index,d)
            if sha(d['provenance'])!=d['provenance_sha256']:raise ValueError('Source provenance changed')
            db.execute('RELEASE dataset_scan');db.commit();d['catalog_status']='indexed_not_training_admission'
        except Exception as error:
            db.execute('ROLLBACK TO dataset_scan');db.execute('RELEASE dataset_scan');db.commit()
            d['catalog_status']='quarantined_invalid_or_unsupported';d['error']=str(error)
        completed.append(d)
        write(out/'status.json',dict(status='cataloguing',datasets_complete=len(completed),datasets_total=len(datasets)))
        print(json.dumps(dict(dataset=index,status=d['catalog_status'],complete=len(completed),total=len(datasets))),flush=True)
    db.executescript('''
        CREATE VIEW training_candidates AS SELECT e.* FROM entries e
        WHERE e.split='train' AND e.schema_valid=1
          AND NOT EXISTS(SELECT 1 FROM groups g WHERE g.group_key=e.group_key AND g.split!='train');
    ''')
    totals=lambda table:dict(rows=db.execute('SELECT count(*) FROM '+table).fetchone()[0],
        public_requests=db.execute('SELECT count(DISTINCT request_hash) FROM '+table).fetchone()[0],
        known_positions=db.execute('SELECT count(DISTINCT position_key) FROM '+table).fetchone()[0])
    phases=[dict(phase=p,rows=n,requests=k) for p,n,k in db.execute('SELECT phase,count(*),count(DISTINCT request_hash) FROM training_candidates GROUP BY phase')]
    collisions=db.execute("SELECT count(*) FROM (SELECT group_key FROM groups GROUP BY group_key HAVING count(DISTINCT split)>1)").fetchone()[0]
    db.commit();db.close();write(out/'sources-private.json',completed)
    summary=dict(status='catalogued_private_index_not_training_admission',datasets=len(datasets),
        dataset_statuses=dict(Counter(d['catalog_status'] for d in completed)),all_indexed=None,
        training_phases=phases,cross_split_groups_quarantined=collisions,
        test_payloads_opened=False,native_validation_payloads_opened=False,
        labels_or_rewards_changed=False,raw_records_public=False,optimizer_updates=0)
    # Open afresh after flushing to prove the persisted index is readable.
    db=sqlite3.connect(out/'index-private.sqlite');summary['all_indexed']=totals('entries');summary['training_candidates_before_quality_selection']=totals('training_candidates');db.close()
    summary['private_index_sha256']=sha(out/'index-private.sqlite');write(out/'result.json',summary);write(out/'status.json',summary)
    return summary


def reclassify(config):
    """Repair classification in a copied completed index, without full rescans."""
    original=Path(config['input_catalog']);previous=read(original/'result.json')
    index=original/'index-private.sqlite';before=sha(index)
    if before!=previous['private_index_sha256']:raise ValueError('Completed input catalog changed')
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False)
    source=sqlite3.connect('file:'+str(index.resolve())+'?mode=ro',uri=True)
    db=sqlite3.connect(out/'index-private.sqlite');source.backup(db);source.close()
    datasets=read(original/'sources-private.json');expected={}
    for d in datasets:
        expected.update(d.get('shard_sha256',{}))
        if d.get('train_sha256'):expected[str(Path(d['path'])/'train.jsonl')]=d['train_sha256']
    records={}
    for rowid,locator in db.execute("SELECT rowid,locator FROM entries WHERE reason='consumed_count_requires_review'"):
        loc=json.loads(locator);records.setdefault(loc['file'],[]).append((rowid,loc))
    checked=corrected=0;input_files={}
    for file,items in records.items():
        checksum=sha(file)
        if expected.get(file)!=checksum:raise ValueError('Reclassification source differs: '+file)
        payload=None
        if items[0][1]['format']=='native_pt':
            import torch
            payload=torch.load(file,map_location='cpu',weights_only=False)
        with Path(file).open('rb') as stream:
            for rowid,loc in items:
                if loc['format']=='native_pt':request=payload['rows'][loc['row']]['public_request']
                else:stream.seek(loc['byte_offset']);request=json.loads(stream.readline())
                phase,kinds,valid,reason=classify(request);checked+=1
                if valid:
                    db.execute('UPDATE entries SET phase=?,kinds=?,schema_valid=1,reason=? WHERE rowid=?',
                        (phase,json.dumps(kinds),'',rowid));corrected+=1
        del payload
        if sha(file)!=checksum:raise ValueError('Reclassification input changed')
        input_files[file]=checksum;db.commit()
    summary={k:v for k,v in previous.items() if k not in ('private_index_sha256','training_phases','training_candidates_before_quality_selection')}
    summary.update(status='reclassified_private_catalog_not_training_admission',
        classification_revision='mjai-kakan-existing-pon-v2',schema_flagged_records_rechecked=checked,
        schema_records_corrected=corrected,original_catalog_result_sha256=sha(original/'result.json'),
        original_catalog_index_sha256=before,source_payloads_changed=False,
        training_candidates_before_quality_selection=dict(
            rows=db.execute('SELECT count(*) FROM training_candidates').fetchone()[0],
            public_requests=db.execute('SELECT count(DISTINCT request_hash) FROM training_candidates').fetchone()[0],
            known_positions=db.execute('SELECT count(DISTINCT position_key) FROM training_candidates').fetchone()[0]),
        training_phases=[dict(phase=p,rows=n,requests=k) for p,n,k in db.execute(
            'SELECT phase,count(*),count(DISTINCT request_hash) FROM training_candidates GROUP BY phase')])
    db.close()
    if sha(index)!=before:raise ValueError('Original catalog changed during repair')
    summary['private_index_sha256']=sha(out/'index-private.sqlite')
    write(out/'sources-private.json',datasets)
    write(out/'reclassification-inputs-private.json',input_files)
    write(out/'result.json',summary);write(out/'status.json',summary)
    return summary


def append_replay(config):
    """Copy a completed catalog and append one independently audited source."""
    original=Path(config['input_catalog']);directory=Path(config['append_replay_source']).resolve()
    index=original/'index-private.sqlite';previous=read(original/'result.json')
    inputs={str(index):previous['private_index_sha256'],
        str(original/'result.json'):config['input_result_sha256'],
        str(original/'sources-private.json'):config['input_sources_sha256'],
        config['audit_result']:config['audit_result_sha256'],
        config['audit_protocol']:config['audit_protocol_sha256']}
    def verify():
        for name,checksum in inputs.items():
            if sha(name)!=checksum:raise ValueError('Bound catalog/source input changed: '+name)
    verify();proof=read(config['audit_result']);audit_protocol=read(config['audit_protocol'])
    if (proof.get('status')!='passed_complete' or proof.get('audit_kind')!='observed_expert_kan_prefix_v1'
            or not proof.get('eligible_for_pool_integration') or not proof.get('all_observed_eligible_windows_enumerated')
            or not proof.get('historical_held_groups_excluded') or proof.get('sealed_test_payloads_opened') is not False):
        raise ValueError('Complete expert-prefix source audit required')
    provenance=directory/'provenance.json';train=directory/'train.jsonl'
    inputs.update({str(provenance):proof['source_provenance_sha256'],str(train):proof['source_train_sha256']})
    verify()
    for name,checksum in audit_protocol['source_files'].items():
        if name in inputs and inputs[name]!=checksum:raise ValueError('Conflicting audit file receipts')
        inputs[name]=checksum
    verify()
    if Path(read(provenance)['config']['catalog_index']).resolve()!=index.resolve():
        raise ValueError('Producer/audit used a different historical split catalog')
    datasets=read(original/'sources-private.json')
    if any(Path(d['path']).resolve()==directory for d in datasets):raise ValueError('Source already catalogued')
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False)
    source=sqlite3.connect('file:'+str(index.resolve())+'?mode=ro',uri=True)
    db=sqlite3.connect(out/'index-private.sqlite');source.backup(db);source.close()
    d=dict(kind='replay_jsonl',path=str(directory),workspace=str(Path.cwd()),
        provenance=str(provenance),provenance_sha256=proof['source_provenance_sha256'],complete=True,
        audit_receipts=[str(Path(config['audit_result']).resolve())],
        audit_sha256={str(Path(config['audit_result']).resolve()):config['audit_result_sha256']})
    scan_replay(db,len(datasets),d)
    if (d['indexed_train_file_rows']!=proof['counts']['records'] or not d['train_checksum_matches_provenance']
            or db.execute('SELECT count(*) FROM entries WHERE dataset=? AND (schema_valid!=1 OR phase!=?)',
                (len(datasets),'kan_window')).fetchone()[0]):
        raise ValueError('Audited kan source index differs')
    d['catalog_status']='indexed_not_training_admission';datasets.append(d)
    def totals(table):
        return dict(rows=db.execute('SELECT count(*) FROM '+table).fetchone()[0],
            public_requests=db.execute('SELECT count(DISTINCT request_hash) FROM '+table).fetchone()[0],
            known_positions=db.execute('SELECT count(DISTINCT position_key) FROM '+table).fetchone()[0])
    summary=dict(status='catalogued_private_index_not_training_admission',datasets=len(datasets),
        dataset_statuses=dict(Counter(d['catalog_status'] for d in datasets)),
        all_indexed=totals('entries'),training_candidates_before_quality_selection=totals('training_candidates'),
        training_phases=[dict(phase=p,rows=n,requests=k) for p,n,k in db.execute(
            'SELECT phase,count(*),count(DISTINCT request_hash) FROM training_candidates GROUP BY phase')],
        cross_split_groups_quarantined=db.execute("SELECT count(*) FROM (SELECT group_key FROM groups GROUP BY group_key HAVING count(DISTINCT split)>1)").fetchone()[0],
        test_payloads_opened=False,native_validation_payloads_opened=False,source_payloads_changed=False,
        original_catalog_index_sha256=inputs[str(index)],appended_audited_records=proof['counts']['records'],
        appended_source_audit_sha256=config['audit_result_sha256'],raw_records_public=False,optimizer_updates=0)
    db.commit();db.close();verify();summary['private_index_sha256']=sha(out/'index-private.sqlite')
    write(out/'sources-private.json',datasets);write(out/'append-inputs-private.json',inputs)
    write(out/'result.json',summary);write(out/'status.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');a=p.parse_args();cfg=read(a.config)
    if a.metadata_only:
        result=discover(cfg['roots']);write(cfg['metadata_out'],result)
        print(json.dumps(dict(datasets=len(result),kinds=dict(Counter(d['kind'] for d in result)),complete=sum(d['complete'] for d in result))))
    else:print(json.dumps(append_replay(cfg) if cfg.get('append_replay_source') else reclassify(cfg) if cfg.get('input_catalog') else run(cfg),ensure_ascii=False))


if __name__=='__main__':main()
