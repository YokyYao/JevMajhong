"""Finite original/v3 shard terminal capture, bound before writer execution."""
import argparse
import json
import os
import re
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields,terminal_record
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read,sha
from tools import curriculum_targeted_win_resume_v2 as original
from tools import curriculum_targeted_win_resume_v3 as recovery

RUN=ROOT/'runs/curriculum-targeted-shard-terminals-v3'
ENTRY=ROOT/'cluster/curriculum-targeted-shard-terminal-v3.slurm'
PLAN=ROOT/'docs/experiments/plans/curriculum-targeted-gpu-replacement-20261125.md'


def descriptor(kind,shard,job,attempt=0,array_job=None):
    if (kind not in ('original','v3') or type(shard) is not int or shard not in range(4)
            or not str(job).isdigit() or type(attempt) is not int
            or kind=='original' and attempt!=0 or kind=='v3' and attempt<1):
        raise ValueError('Exact original or v3 shard/attempt/job identity required')
    if kind=='original':
        array_job=str(job if array_job is None else array_job)
        if not array_job.isdigit() or int(array_job)<=0:raise ValueError('Actual positive array master ID required')
    elif array_job is not None:raise ValueError('A v3 recovery is not an original array task')
    if kind=='original':
        folder=original.RUN/'shards'/f'shard-{shard:02d}'
        command=ROOT/'cluster/curriculum-targeted-win-labels-v1.slurm'
        name=f'original-shard{shard:02d}-job{job}'
    else:
        folder=recovery.RUN/f'shard-{shard:02d}'/f'attempt-{attempt:03d}'
        command=recovery.ENTRY;name=f'v3-shard{shard:02d}-attempt{attempt:03d}-job{job}'
    return dict(kind=kind,shard=shard,job=str(job),attempt=attempt,folder=str(folder),command=str(command),
        array_job=array_job,binding=str(RUN/(name+'-binding-private.json')),archive=str(RUN/(name+'-terminal-private.json')))


def single_array_task(value,shard):
    match=re.fullmatch(r'(\d+)(?:%\d+)?',str(value))
    return match is not None and int(match.group(1))==shard


def require_pending(v,d):
    if (v.get('JobId')!=d['job'] or v.get('Command')!=d['command']
            or v.get('JobState')!='PENDING' or v.get('RunTime')!='00:00:00'
            or v.get('AllocTRES') not in ('(null)','')
            or v.get('StartTime') not in ('Unknown','N/A','(null)')
            or not v.get('UserId','').endswith(f'({os.getuid()})')):
        raise ValueError('Actual own pending, never allocated writer required')
    if d['kind']=='original' and (v.get('ArrayJobId')!=d['array_job'] or not single_array_task(v.get('ArrayTaskId'),d['shard'])):
        raise ValueError('Wrong original array shard')


def bind(d):
    allocation=resource();path=Path(d['binding'])
    if path.exists():raise ValueError('Never overwrite a writer/capture binding')
    observed=scheduler(d['job'])
    if observed.returncode:raise ValueError('Cannot observe the actual pending writer')
    require_pending(fields(observed.stdout),d)
    if Path(d['folder']).exists():raise ValueError('Pending writer output already exists')
    regpath=original.RUN/'registered-private.json';reg=read(regpath)
    selected=reg['selected'][d['shard']*27:(d['shard']+1)*27]
    if len(selected)!=27:raise ValueError('The fixed original27-view population is required')
    paths=[Path(__file__),ENTRY,PLAN,regpath,Path(d['command']),
        ROOT/'tools/curriculum_targeted_win_resume_v2.py',ROOT/'tools/curriculum_targeted_win_resume_v3.py',
        ROOT/'tools/capture_curriculum_paired_terminal_v2.py',
        ROOT/'tests/core/test_curriculum_targeted_shard_terminal_v3.py']
    RUN.mkdir(exist_ok=True)
    write(path,dict(status='actual_pending_targeted_writer_terminal_capture_bound',descriptor=d,
        observed_pending_writer=observed.stdout,selected=selected,model=reg['continuation_model'],
        files={str(p):sha(p) for p in paths},allocation=allocation,student_updates=0))
    print(json.dumps({'status':'actual_pending_targeted_writer_terminal_capture_bound','kind':d['kind'],
        'shard':d['shard'],'original_job':d['job']}),flush=True)


def capture(d):
    allocation=resource();b=read(d['binding'])
    if b['descriptor']!=d or b['status']!='actual_pending_targeted_writer_terminal_capture_bound':
        raise ValueError('Exact before-execution descriptor binding required')
    if any(sha(p)!=h for p,h in b['files'].items()):raise ValueError('Bound terminal source changed')
    folder=Path(d['folder']);launch=read(folder/'registered-private.json')
    if (str(launch['job_id'])!=d['job'] or launch['selected']!=b['selected'] or launch['model']!=b['model']):
        raise ValueError('Actual initialized writer has a different population, teacher or job')
    proof=(original.require_terminal(folder) if d['kind']=='original' else recovery.require_terminal(folder))
    if proof['job_id']!=d['job']:raise ValueError('Wrong actual terminal writer')
    if d['kind']=='original':
        v=fields(proof['raw'])
        if v.get('ArrayJobId')!=d['array_job'] or not single_array_task(v.get('ArrayTaskId'),d['shard']):
            raise ValueError('Wrong terminal original array element')
    archive=Path(d['archive'])
    write(archive,dict(raw=proof['raw'],exit_receipt=proof['exit_receipt'],terminal_proof=proof,
        capture_allocation=allocation,capture_binding_sha256=sha(d['binding']),
        actual_scheduler_state_is_authoritative=True,script_receipt_alone_never_implies_success=True,
        genuine_scheduler_record=True,inferred_or_synthetic_terminal=False))
    files=dict(b['files']);files[str(Path(d['binding']))]=sha(d['binding']);files[str(archive)]=sha(archive)
    write(archive.with_name(archive.name.replace('-terminal-private.json','-completion-binding-private.json')),dict(files=files))
    print(json.dumps({'status':'actual_targeted_writer_terminal_archived','kind':d['kind'],'shard':d['shard'],
        'job_id':proof['job_id'],'state':proof['state'],'original_successful':proof['successful'],
        'model_loaded':False,'new_games':0}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('bind','capture'),required=True)
    p.add_argument('--kind',choices=('original','v3'),required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--job',type=int,required=True);p.add_argument('--attempt',type=int,default=0)
    p.add_argument('--array-job',type=int)
    a=p.parse_args();d=descriptor(a.kind,a.shard,a.job,a.attempt,a.array_job)
    bind(d) if a.phase=='bind' else capture(d)
