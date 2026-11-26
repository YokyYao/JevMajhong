"""Verify two canonical source collections without any private job wrapper."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def compatible_source_specs(first,second):
    a=dict(first['config']);b=dict(second['config']);a.pop('out');b.pop('out')
    if (a!=b or a.get('method')!='stage4_mortal_grouped_score_context_inputs_v4'
        or a.get('simultaneous_seat_order')!='ascending_absolute_seat'
        or first['sources']!=second['sources'] or len(first['sources'])!=64
        or first['source_package_sha256']!=second['source_package_sha256']):
        raise ValueError('Only the output path may differ between canonical source runs')
    return True


def verify(parent,repeat,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    if out.exists():raise ValueError('Never overwrite a source repeat receipt')
    read=lambda path:json.loads(path.read_text())
    directories=[root if (root/'source-registration-private.json').exists() else root/'data' for root in (parent,repeat)]
    registrations=[read(directory/'source-registration-private.json') for directory in directories]
    compatible_source_specs(*registrations)
    ready=[read(directory/'data-ready-private.json') for directory in directories]
    for reg in registrations:
        if not all(sha(path)==digest for path,digest in reg['files'].items()):raise ValueError('Source dependencies changed')
    if any(item['source_legal_win_declines']!=0 or item['source_ledgers_passed'] is not True
           or item['all_score_replays_verified'] is not True or item['labels_generated']!=0 for item in ready):
        raise ValueError('Incomplete source collection or already labelled inputs')
    if ready[0]['counts']!=ready[1]['counts'] or ready[0]['source_receipts']!=ready[1]['source_receipts']:
        raise ValueError('Repeated source counts/receipts differ')
    counts={}
    for split in ('train','validation'):
        paths=[ROOT/item['files'][split]['path'] for item in ready]
        for p,item in zip(paths,ready):
            if sha(p)!=item['files'][split]['sha256']:raise ValueError('Collected input file changed')
        first,second=[p.read_bytes() for p in paths]
        if first!=second:raise ValueError('Repeated input rows are not byte-identical')
        counts[split]=len(first.splitlines())
    receipt=dict(status='canonical_source_repeat_verified',repeated_all_rows_identical=True,
        repeated_runs_are_same64_source_families=True,counts=counts,
        parent_data_ready_sha256=sha(directories[0]/'data-ready-private.json'),
        repeat_data_ready_sha256=sha(directories[1]/'data-ready-private.json'),
        labels_generated=0,teacher_selected=False,stage4_fit_admitted=False,
        native_input_audit_separate_requirement=True,source_script_sha256=sha(__file__),
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-root','repeat-root','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();verify(a.parent_root.resolve(),a.repeat_root.resolve(),a.out.resolve())
