"""Prepare later-course public-context views; never admit a transfer parent."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_riichi_context import VERSION,FIELD,enrich_request
from tools.curriculum_parent import require_replay


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())


def prepare(source_data_ready,out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    source_data_ready=source_data_ready.resolve();out=out.resolve();out.relative_to(ROOT)
    source=read(source_data_ready)
    if source['status'] not in ('stage2_data_prepared_pending_parent_and_fit_registration',
                                'stage3_data_prepared_pending_parent_and_fit_registration'):
        raise ValueError('Prepared later-course data required')
    if source.get('required_input_schema'):raise ValueError('Source already has an input contract')
    out.mkdir(parents=True,exist_ok=False);files={};counts={}
    visible=lambda r:dict(state=r['state'],questions={name:{k:q[k] for k in ('type','instructions','criteria')}
        for name,q in r['questions'].items()})
    for split,spec in source['files'].items():
        path=ROOT/spec['path']
        if sha(path)!=spec['sha256']:raise ValueError('Prepared source changed')
        rows=[json.loads(s) for s in path.read_text().splitlines() if s];enriched=[];summary=Counter()
        for row in rows:
            item=enrich_request(row);q=row['questions']['action'];newq=item['questions']['action']
            assert row['_meta']==item['_meta']
            assert {k:v for k,v in q.items() if k!='criteria'}=={k:v for k,v in newq.items() if k!='criteria'}
            assert visible(item)==enrich_request(visible(row))
            restored=copy.deepcopy(item)
            for value in restored['questions']['action']['criteria'].values():value.pop(FIELD,None)
            assert restored==row
            summary['records']+=1
            summary['reach_windows']+=any(v['type']=='reach' for v in q['criteria'].values())
            summary['unchanged_windows']+=item==row
            enriched.append(item)
        assert len(rows)==spec['records']
        file=out/f'{split}.jsonl'
        with file.open('x') as f:
            for row in enriched:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        files[split]=dict(spec,path=str(file.relative_to(ROOT)),sha256=sha(file))
        counts[split]=dict(summary)
    result=copy.deepcopy(source)
    result.update(files=files,required_input_schema=VERSION,feature_code_sha256=sha(ROOT/'tools/curriculum_riichi_context.py'),
        context_preparation_code_sha256=sha(Path(__file__)),source_data_ready=str(source_data_ready.resolve()),
        source_data_ready_sha256=sha(source_data_ready),source_files=source['files'],context_counts=counts,
        model_visible_inputs_unchanged=False,model_visible_difference='only_versioned_public_riichi_context',
        labels_targets_ids_groups_and_loss_weights_unchanged=True,parent_model_admitted=False,optimizer_updates=0,
        variant_fit_admitted=False,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    if source['status'].startswith('stage2_'):
        parent=read(ROOT/'runs/curriculum-context-v3/fit-plan/data-ready.json')
        for split in ('train','validation'):
            parent_rows=[json.loads(s) for s in (ROOT/parent['stages']['stage1'][split]['path']).read_text().splitlines() if s]
            child_rows=[json.loads(s) for s in (ROOT/files[split]['path']).read_text().splitlines() if s]
            require_replay(parent_rows,child_rows)
        result['context_stage1_train_and_validation_replay_identical']=True
    with (out/'data-ready.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],required_input_schema=VERSION,counts=counts,
        parent_admitted=False,variant_fit_admitted=False,optimizer_updates=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-data-ready',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.source_data_ready,a.out)
