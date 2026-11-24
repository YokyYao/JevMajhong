"""Finite GPU job preserving a genuine original label-shard terminal.

Slurm MinJobAge is300 seconds; preserve the real scheduler record before a
longer recovery or teacher job can occupy the next available GPU slot.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields,terminal_record
from tools.curriculum_targeted_win_resume_v2 import require_terminal,RUN as ORIGINAL
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write

RUN=ROOT/'runs/curriculum-targeted-win-terminal-v1'
ENTRY=ROOT/'cluster/curriculum-original-shard-terminal-v1.slurm'
PLAN=ROOT/'docs/experiments/plans/curriculum-original-shard-terminal-20261125.md'
ORIGINAL_ENTRY=ROOT/'cluster/curriculum-targeted-win-labels-v1.slurm'


def bind():
    allocation=resource()
    target=RUN/'original-shard00-binding-private.json'
    if target.exists():raise ValueError('Never overwrite a terminal binding')
    folder=ORIGINAL/'shards/shard-00';launch=read(folder/'registered-private.json')
    job=str(launch['job_id']);observed=scheduler(job);v=fields(observed.stdout)
    if (observed.returncode or job!='322030' or v.get('JobId')!=job
        or v.get('ArrayJobId')!='319759' or v.get('ArrayTaskId')!='0'
        or v.get('JobState')!='RUNNING' or v.get('Command')!=str(ORIGINAL_ENTRY)
        or v.get('NodeList')!=launch['node'].split('.')[0]):
        raise ValueError('Bind only the actual running original shard00 writer')
    # The real running observation must not be accepted as a terminal proof.
    try:terminal_record(observed.stdout,job,ORIGINAL_ENTRY)
    except ValueError:pass
    else:raise ValueError('A live writer was accepted as terminal')
    files={str(p.resolve()):sha(p) for p in (Path(__file__),ENTRY,PLAN,
        ROOT/'tools/capture_curriculum_paired_terminal_v2.py',
        ROOT/'tools/curriculum_targeted_win_resume_v2.py',
        ORIGINAL_ENTRY,folder/'registered-private.json')}
    RUN.mkdir(exist_ok=True)
    write(target,{'status':'actual_live_original_shard_terminal_capture_bound',
        'original_job':job,'folder':str(folder),'files':files,
        'actual_running_observation':observed.stdout,'allocation':allocation,
        'running_writer_rejected_as_terminal':True,'model_loaded':False,'new_games':0})
    print(json.dumps({'status':'original_shard00_terminal_capture_bound','original_job':job}),flush=True)


def capture():
    allocation=resource();binding_path=RUN/'original-shard00-binding-private.json'
    b=read(binding_path)
    if b['status']!='actual_live_original_shard_terminal_capture_bound':raise ValueError('Exact live binding required')
    if any(sha(p)!=h for p,h in b['files'].items()):raise ValueError('Bound capture input changed')
    # No polling, inferred terminal, synthetic receipt or alternative writer.
    proof=require_terminal(Path(b['folder']))
    if proof['job_id']!=b['original_job']:raise ValueError('Wrong original writer')
    v=fields(proof['raw'])
    if v.get('ArrayJobId')!='319759' or v.get('ArrayTaskId')!='0':raise ValueError('Wrong original array element')
    archive=RUN/'original-shard00-terminal-private.json'
    write(archive,{'raw':proof['raw'],'exit_receipt':proof['exit_receipt'],
        'terminal_proof':proof,'capture_allocation':allocation,
        'capture_binding_sha256':sha(binding_path),
        'genuine_scheduler_record':True,'inferred_or_synthetic_terminal':False})
    files=dict(b['files']);files[str(binding_path)]=sha(binding_path);files[str(archive)]=sha(archive)
    write(RUN/'original-shard00-completion-binding-private.json',{'files':files})
    print(json.dumps({'status':'actual_original_shard00_terminal_archived',
        'original_job':proof['job_id'],'state':proof['state'],
        'original_successful':proof['successful'],'model_loaded':False,'new_games':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('bind','capture'),required=True)
    args=parser.parse_args();bind() if args.phase=='bind' else capture()
