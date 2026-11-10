"""Prepare audited repaired natural records and a separate encoder warm start."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from call_policy import digest,is_call_request,request_hash,screenshot_view
from head_rl import write


def views(record,names):
    for view in names:
        result=copy.deepcopy(record) if view=='native' else screenshot_view(record,unknown=view=='screen_unknown')
        result['_meta'].update(id=record['_meta']['id']+':encoder:'+view,view=view,variant='clean',
            parent_id=record['_meta']['id'],schema='call-encoder-v1-correct-candidates')
        result['_meta']['text_sha256']=request_hash(result)
        if not is_call_request(result) or any(len(c.get('consumed',[]))!=2 for c in result['questions']['action']['criteria'].values() if c['type'] in ('chi','pon')):
            raise ValueError('Candidate leak/legality scope')
        yield result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());source=ROOT/cfg['data_source'];data=ROOT/cfg['data'];out=ROOT/cfg['out']
    provenance=json.loads((source/'provenance.json').read_text())
    if not provenance['complete']:raise ValueError('Source incomplete')
    train_matches=set(provenance['matches']['train']);val_matches=set(provenance['matches']['validation']);test_matches=set(provenance['matches']['test'])
    if train_matches&val_matches or train_matches&test_matches or val_matches&test_matches:raise ValueError('Source match overlap')
    if data.exists() or out.exists():raise ValueError('Refuse overwrite; use a new experiment')
    records={}
    for split in ('train','validation'):
        path=source/(split+'.jsonl')
        if digest(path)!=provenance['files'][split]:raise ValueError('Source digest changed')
        records[split]=[json.loads(l) for l in path.read_text().splitlines()]
    if len(records['train'])!=8192 or len(records['validation'])!=2048:raise ValueError('Unexpected natural data coverage')
    # Fixed ID hash, independent of action labels and model predictions.
    validation=sorted(records['validation'],key=lambda r:hashlib.sha256((str(cfg['seed'])+r['_meta']['id']).encode()).hexdigest())[:cfg['validation_raw_records']]
    chosen={'train':records['train'],'validation':validation};data.mkdir(parents=True);out.mkdir(parents=True)
    counts={};matches={}
    for split,rows in chosen.items():
        counts[split]=dict(Counter(r['questions']['action']['criteria'][r['questions']['action']['label']]['type'] for r in rows))
        if min(counts[split].get(k,0) for k in ('chi','pon','none'))<10:raise ValueError('Insufficient natural class coverage')
        matches[split]=sorted({r['_meta']['group_id'] for r in rows})
        with (data/(split+'.jsonl')).open('w') as f:
            for record in rows:
                if request_hash(record)!=record['_meta']['text_sha256']:raise ValueError('Source request identity differs')
                for result in views(record,cfg['views']):f.write(json.dumps(result,ensure_ascii=False)+'\n')
    reference=ROOT/cfg['reference'];source_protocol=json.loads((ROOT/cfg['reference_training_protocol']).read_text())
    for name,h in source_protocol['reference_files'].items():
        if digest(reference/name)!=h:raise ValueError('Incumbent changed')
    warm=ROOT/cfg['warm_start']
    if digest(warm)!=cfg['warm_start_sha256']:raise ValueError('Call warm start changed')
    payload=torch.load(warm,map_location='cpu',weights_only=True);meta=torch.load(reference/'head.pt',map_location='cpu',weights_only=True)
    if payload['reference_head_sha256']!=cfg['reference_head_sha256']:raise ValueError('Call source mismatch')
    initial=ROOT/cfg['initialization'];initial.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        shutil.copy2(reference/name,initial/name)
    meta['head']=payload['head'];meta['encoder_pilot_initialization']={'reference_head_sha256':cfg['reference_head_sha256'],'call_head_sha256':cfg['warm_start_sha256']}
    torch.save(meta,initial/'head.pt')
    report={'complete':True,'config':cfg,'raw_counts':{k:len(v) for k,v in chosen.items()},'action_types':counts,'matches':matches,
        'views':cfg['views'],'source_provenance_sha256':digest(source/'provenance.json'),
        'files':{k:digest(data/(k+'.jsonl')) for k in chosen},'records':{k:len(v)*len(cfg['views']) for k,v in chosen.items()},
        'reference_files':source_protocol['reference_files'],'initialization_files':{p.name:digest(p) for p in initial.iterdir()},
        'human_test_opened':False,'validation_sampling':'fixed request-ID hash; no label quota or prediction selection'}
    write(data/'provenance.json',report);write(out/'preparation.json',report)
    print(json.dumps({k:report[k] for k in ('complete','raw_counts','action_types','records','human_test_opened')}),flush=True)


if __name__=='__main__':main()
