"""Restore the pinned public v8 dataset and audit prospective curriculum splits.

Run in a scheduled GPU allocation on this cluster. Source archives and derived
JSONL remain local. No model weights are loaded and no optimizer is run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import urllib.request

REVISION='7ee572904bfd30387d2451a2dd1d770f90ffebf7'
URL=f'https://raw.githubusercontent.com/YokyYao/JevMajhong/{REVISION}/datasets/jev-play-v2/'
EXPECTED={'train':'b462fbacdf2b4cbad9517d096bc1418638b2ddfed866060562c4f2dc2d2d295f',
          'validation':'1179bb4a26a6fe5b01cd13dcaeec46cd9d67adbc1b1deddb306c60f5d52d3540'}

def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def download(name):
    if Path(name).name!=name:raise ValueError('Invalid archive member')
    with urllib.request.urlopen(URL+name,timeout=40) as response:return response.read()

def summary(rows):
    phases=Counter();labels=Counter();positive=Counter();opportunities=Counter();sources=Counter()
    for r in rows:
        meta=r['_meta'];q=r['questions']['action'];phases[meta['phase']]+=1;sources[meta['label_type']]+=1
        kinds={v['type'] for v in q['criteria'].values()};opportunities.update(kinds)
        if meta['label_type']=='expert_replay':
            action=q['criteria'][q['label']]['type'];positive[action]+=1
        labels.update([meta['phase']+':'+meta['label_type']])
    return dict(records=len(rows),groups=len({r['_meta']['group_id'] for r in rows}),
        phases=dict(phases),sources=dict(sources),phase_source=dict(labels),
        expert_action_counts=dict(positive),legal_action_opportunities=dict(opportunities))

def prepare(output):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('A scheduled GPU allocation is required on this cluster')
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(download('manifest.json'));rows={};files={}
    archives=output/'archives';archives.mkdir()
    for split,spec in manifest['splits'].items():
        chunks=[]
        for part in spec['parts']:
            data=download(part['name'])
            if len(data)!=part['bytes'] or sha(data)!=part['sha256']:raise ValueError('Part checksum mismatch')
            (archives/part['name']).write_bytes(data);chunks.append(data)
        packed=b''.join(chunks)
        if len(packed)!=spec['archive_bytes'] or sha(packed)!=spec['archive_sha256']:raise ValueError('Archive checksum mismatch')
        raw=gzip.decompress(packed)
        if len(raw)!=spec['bytes'] or sha(raw)!=spec['sha256'] or sha(raw)!=EXPECTED[split]:
            raise ValueError('Public JSONL checksum mismatch')
        rows[split]=[json.loads(line) for line in raw.splitlines()]
        if len(rows[split])!=spec['records']:raise ValueError('Wrong restored record count')
        path=output/f'{split}.jsonl';path.write_bytes(raw);files[split]=dict(path=str(path),sha256=sha(raw))
    groups={k:{r['_meta']['group_id'] for r in v} for k,v in rows.items()}
    if groups['train']&groups['validation']:raise ValueError('Whole-game source split overlaps')
    stages={};selection={
        'stage1':lambda r:r['_meta']['label_type']=='expert_replay' and r['_meta']['phase']=='closed_discard',
        'stage2':lambda r:r['_meta']['label_type']=='expert_replay',
        'stage3':lambda r:True}
    for name,predicate in selection.items():
        directory=output/name;directory.mkdir();parts={}
        for split,pool in rows.items():
            admitted=[r for r in pool if predicate(r)]
            # Preserve public file order: derivation does not depend on private IDs.
            indices=[i for i,r in enumerate(pool) if predicate(r)]
            raw=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in admitted).encode()
            path=directory/f'{split}.jsonl';path.write_bytes(raw)
            parts[split]=dict(summary(admitted),path=str(path),sha256=sha(raw),
                public_row_indices_sha256=sha(json.dumps(indices,separators=(',',':')).encode()))
        stages[name]=parts
    report=dict(status='public_curriculum_data_audited_not_training',public_revision=REVISION,
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],gpu=os.environ['SLURM_JOB_GPUS'],
        source_files=files,source_summary={k:summary(v) for k,v in rows.items()},stages=stages,
        whole_source_group_overlap=0,stage4_admitted=False,optimizer_updates=0,
        warning='Coverage counts describe demonstrations, not expert optimality or playing strength.')
    write(output/'manifest.json',manifest);write(output/'coverage.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out)
