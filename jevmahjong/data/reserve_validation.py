"""Derive an expert-kan validation pool unseen by the current model family."""
import argparse
import json
from pathlib import Path
import sqlite3

from jevmahjong.core.io import digest, write


def reserve(cfg):
    read = lambda p: json.loads(Path(p).read_text())
    parent = Path(cfg['parent_pool']); original = parent/'pool-private.sqlite'
    manifest = Path(cfg['reservation']); marker = Path(cfg['holdout_metadata'])
    inputs = {str(original):cfg['parent_pool_sha256'], str(parent/'result.json'):cfg['parent_result_sha256'],
        str(manifest):cfg['reservation_sha256'], str(marker):cfg['holdout_metadata_sha256']}
    def verify():
        for file, checksum in inputs.items():
            if digest(Path(file)) != checksum:
                raise ValueError('Bound validation reservation input changed: '+file)
    verify(); previous = read(parent/'result.json'); reservation = read(manifest); metadata = read(marker)
    if (previous['status'] != 'raw_request_pool_ready_not_yet_balanced'
            or previous['pool_sha256'] != cfg['parent_pool_sha256']
            or not previous['requires_masked_soft_target_loss']
            or reservation['schema'] != 'family-unseen-expert-prefix-validation-v1'):
        raise ValueError('Completed pool and explicit family reservation required')
    groups = set(reservation['groups'])
    if not groups or len(groups) != len(reservation['groups']):
        raise ValueError('Unique nonempty validation groups required')
    if (metadata.get('complete') is not False or metadata['matches']['train'] or metadata['matches']['test']
            or {'replay:'+g for g in metadata['matches']['validation']} != groups
            or metadata['reservation_sha256'] != cfg['reservation_sha256']):
        raise ValueError('Sticky historical holdout metadata differs')
    inputs.update(reservation['family_input_sha256']); verify()
    seen = set()
    for name, checksum in reservation['family_runs'].items():
        run = Path(name); protocol_file = run/'protocol.json'
        if digest(protocol_file) != checksum:
            raise ValueError('Family protocol changed')
        inputs[str(protocol_file)] = checksum
        protocol = read(protocol_file); init_file = run/'initialization-audit.json'; init = read(init_file)
        if (init.get('old_adapters_loaded') is not False or init.get('external_residual_networks') != 0
                or protocol['trainer_args'].get('init_from') or protocol['trainer_args'].get('resume')):
            raise ValueError('Fresh-base family lineage required')
        inputs[str(init_file)] = digest(init_file)
        for key in ('data','eval_data'):
            file = Path(protocol['trainer_args'][key]); expected = protocol['input_sha256'][str(file)]
            if inputs.get(str(file.resolve())) != expected or digest(file) != expected:
                raise ValueError('Complete family data/validation receipts required')
            with file.open() as stream:
                for line in stream:
                    if line.strip():seen.add(json.loads(line)['_meta']['group_id'])
    if groups & seen:
        raise ValueError('Reserved group was seen by model family')
    directory = Path(reservation['source_directory']).resolve()
    train = directory/'train.jsonl'; provenance = directory/'provenance.json'
    inputs.update({str(train):reservation['source_train_sha256'],
        str(provenance):reservation['source_provenance_sha256']}); verify()
    if not groups <= {'replay:'+g for g in read(provenance)['matches']['train']}:
        raise ValueError('Reserved group is absent from audited expert source')
    specs = previous['config']['sources']
    spec = [s for s in specs if Path(s['path']).resolve() == directory]
    if len(spec) != 1 or sum(s['priority'] == spec[0]['priority'] for s in specs) != 1:
        raise ValueError('Unambiguous expert source priority required')
    priority = spec[0]['priority']
    out = Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    source = sqlite3.connect('file:'+str(original.resolve())+'?mode=ro',uri=True)
    db = sqlite3.connect(str(out/'pool-private.sqlite'),uri=True);source.backup(db);source.close()
    if db.execute('SELECT COUNT(*) FROM (SELECT group_key FROM cases GROUP BY group_key HAVING COUNT(DISTINCT split)>1)').fetchone()[0]:
        raise ValueError('Parent pool already leaks groups')
    before = dict(db.execute('SELECT split,COUNT(*) FROM cases GROUP BY split'))
    db.execute('CREATE TEMP TABLE reserved_groups (group_key TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO reserved_groups VALUES (?)',[(g,) for g in groups])
    db.execute('CREATE INDEX IF NOT EXISTS case_groups ON cases(group_key,split)')
    kept = "phase='kan_window' AND label_type='expert_replay' AND priority=?"
    where = "split='train' AND group_key IN (SELECT group_key FROM reserved_groups)"
    moved = db.execute('SELECT COUNT(*) FROM cases WHERE '+where+' AND '+kept,(priority,)).fetchone()[0]
    if moved < 1:raise ValueError('No reserved expert kan records')
    removed = db.execute('DELETE FROM cases WHERE '+where+' AND NOT ('+kept+')',(priority,)).rowcount
    db.execute("UPDATE cases SET split='validation' WHERE "+where+' AND '+kept,(priority,))
    after = dict(db.execute('SELECT split,COUNT(*) FROM cases GROUP BY split'))
    if after.get('train',0) != before.get('train',0)-moved-removed or after['validation'] != before.get('validation',0)+moved:
        raise ValueError('Partition row counts differ')
    db.execute('ATTACH DATABASE ? AS original',('file:'+str(original.resolve())+'?mode=ro',))
    differences = db.execute('''SELECT COUNT(*) FROM cases n LEFT JOIN original.cases o ON n.position=o.position
        WHERE o.position IS NULL OR n.group_key IS NOT o.group_key OR n.phase IS NOT o.phase OR n.label_type IS NOT o.label_type
        OR n.choice IS NOT o.choice OR n.view IS NOT o.view OR n.priority IS NOT o.priority OR n.payload IS NOT o.payload
        OR (n.split IS NOT o.split AND NOT(n.split='validation' AND o.split='train'
            AND n.group_key IN(SELECT group_key FROM reserved_groups)
            AND n.phase='kan_window' AND n.label_type='expert_replay' AND n.priority=?))''',(priority,)).fetchone()[0]
    overlap = db.execute('SELECT COUNT(*) FROM (SELECT group_key FROM cases GROUP BY group_key HAVING COUNT(DISTINCT split)>1)').fetchone()[0]
    if differences or overlap:raise ValueError('Reservation changed labels/payloads or leaks groups')
    strata = [dict(split=s,phase=p,label_type=l,choice=c,positions=n) for s,p,l,c,n in db.execute(
        'SELECT split,phase,label_type,choice,COUNT(*) FROM cases GROUP BY split,phase,label_type,choice')]
    db.commit();db.close();verify()
    result = dict(previous,status='raw_request_pool_ready_not_yet_balanced',
        config=dict(sources=specs,reservation_derivation=cfg),pool_sha256=digest(out/'pool-private.sqlite'),
        groups_overlap=0,strata=strata,validation_reservation=dict(groups=len(groups),family_group_overlap=0,
            expert_kan_records_moved=moved,other_source_train_records_excluded=removed,
            before=before,after=after,original_validation_preserved=True,payloads_and_labels_changed=False,
            source_pool_unchanged=True,training_started=False))
    write(out/'inputs-private.json',inputs);write(out/'result.json',result);write(out/'status.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    result=reserve(json.loads(parser.parse_args().config.read_text()))
    print(json.dumps(dict(status=result['status'],reservation=result['validation_reservation'])))


if __name__=='__main__':main()
