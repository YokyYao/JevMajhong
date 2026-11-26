"""Capture actual queued case-worker/assembly/merge terminals, including failures."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import curriculum_targeted_parallel_worker_v1 as worker
from tools import curriculum_targeted_parallel_assembly_v1 as assembly
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,terminal_record,fields
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write

ENTRY=ROOT/'cluster/curriculum-targeted-case-capture-v1.slurm'
ENTRIES=dict(worker=worker.ENTRY,assembly=assembly.ENTRY,merge=assembly.MERGE_ENTRY)
PREFIXES=dict(worker=worker.EXIT_PREFIX,assembly='curriculum-targeted-case-assembly-v1-exit-',
    merge='curriculum-targeted-case-merge-v1-exit-')


def archive_path(job):
    return worker.RUN/'terminals'/f'job-{job}-terminal-private.json'


def capture(kind,job):
    allocation=resource(); worker.verify_binding()
    queue=read(worker.RUN/'queue-private.json')
    descriptors=[item for item in queue['descriptors'] if item['job']==job and item['kind']==kind]
    if len(descriptors) != 1 or kind not in ENTRIES:
        raise ValueError('Exactly the actual before-execution queued job descriptor is required')
    descriptor=descriptors[0]; pending=fields(descriptor['observed_pending_raw'])
    if (pending['JobId'] != job or pending['Command'] != str(ENTRIES[kind])
            or pending['JobState'] != 'PENDING' or pending['RunTime'] != '00:00:00'
            or not pending['UserId'].endswith(f'({os.getuid()})')):
        raise ValueError('Original actual own never-executed job binding differs')
    actual=scheduler(job)
    if actual.returncode: raise ValueError('Actual queued job terminal expired; never invent it')
    proof=terminal_record(actual.stdout,job,ENTRIES[kind])
    receipt_path=ROOT/'logs'/f'{PREFIXES[kind]}{job}.json'; receipt=read(receipt_path)
    if str(receipt['job_id']) != job or proof['successful'] and receipt['exit_code'] != 0:
        raise ValueError('Actual script receipt and successful terminal differ')
    initialized=False
    if descriptor.get('folder'):
        path=Path(descriptor['folder'])/'registered-private.json'
        if path.exists():
            launch=read(path); initialized=True
            if str(launch['job_id']) != job or fields(actual.stdout)['NodeList'] != launch['node'].split('.')[0]:
                raise ValueError('Actual initialized producer job/node differs')
    out=archive_path(job); out.parent.mkdir(parents=True,exist_ok=True)
    write(out,dict(raw=actual.stdout,exit_receipt=receipt,terminal_proof=proof,
        capture_allocation=allocation,genuine_scheduler_record=True,
        inferred_or_synthetic_terminal=False,initialized=initialized,
        before_execution_descriptor=descriptor,queue_sha256=sha(worker.RUN/'queue-private.json'),
        worker_source_binding_sha256=sha(worker.BINDING)))
    print(dict(status='actual_disjoint_case_job_terminal_preserved',kind=kind,job=job,
        successful=proof['successful'],initialized=initialized),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=tuple(ENTRIES),required=True); p.add_argument('--job',required=True)
    args=p.parse_args(); capture(args.kind,args.job)
