"""Bind/freeze a concrete finite case-parallel queue; never cancel live jobs.

The queue remains held for explicit scheduler-level handoff after inspection.
Only the execution partition changes; full108 data and Stage4 remain unchanged.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import curriculum_targeted_parallel_worker_v1 as worker
from tools import curriculum_targeted_parallel_assembly_v1 as assembly
from tools import curriculum_targeted_parallel_capture_v1 as capture
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write

ENTRY=ROOT/'cluster/curriculum-targeted-case-control-v1.slurm'
PUBLIC=('tools/curriculum_targeted_case_partition_v1.py',
    'tools/curriculum_targeted_parallel_worker_v1.py','tools/curriculum_targeted_parallel_assembly_v1.py',
    'tools/curriculum_targeted_parallel_capture_v1.py','tools/launch_curriculum_targeted_parallel_v1.py',
    'tests/core/test_curriculum_targeted_case_partition_v1.py',
    'tests/core/test_curriculum_targeted_parallel_lifecycle_v1.py',
    'cluster/curriculum-targeted-case-worker-v1.slurm','cluster/curriculum-targeted-case-assembly-v1.slurm',
    'cluster/curriculum-targeted-case-merge-v1.slurm','cluster/curriculum-targeted-case-capture-v1.slurm',
    'cluster/curriculum-targeted-case-control-v1.slurm')
ORIGINAL=worker.previous.RUN/'shards/shard-03'
LAST=worker.cached.RUN/'shard-03/attempt-001'
OLD_ARCHIVE=ROOT/'runs/curriculum-targeted-shard-terminals-v3/original-shard03-job324259-terminal-private.json'
LAST_ARCHIVE=ROOT/'runs/curriculum-targeted-shard-terminals-v3/v3-shard03-attempt001-job324261-terminal-private.json'


def bind():
    allocation=resource()
    if worker.BINDING.exists() or worker.DISPATCH.exists() or (worker.RUN/'queue-private.json').exists():
        raise ValueError('Never overwrite any actual parallel source entry, dispatch or queue')
    parity=ROOT/'runs/curriculum-targeted-case-parallel-runtime-preflight-v1'
    receipt_path=parity/'completion-binding-private.json'; receipt=read(receipt_path)
    if any(sha(path) != expected for path,expected in receipt['files'].items()):
        raise ValueError('Actual case-order/teacher-proposal parity sources changed')
    proof=read(parity/'summary-private.json')
    if (proof['status'] != 'actual_three_source_case_order_six_public_world_blocks_exact_not_parallel_launch_admission'
            or proof['actual_proposal_blocks'] != 6 or proof['original_world_allocations_reproduced'] != 96
            or not proof['original_complete_current_menus_exact'] or proof['student_updates']):
        raise ValueError('Actual unchanged-teacher case-order proposal proof required')
    tests=ROOT/'logs/targeted-case-partition-check-v1-326733.log'
    text=tests.read_text()
    if 'Ran 6 tests' not in text or not text.rstrip().endswith('OK'):
        raise ValueError('Actual complete ownership and unchanged numeric-body checks required')
    core=read(worker.cached.BINDING); files=dict(core['files'])
    worker.combine_files(files,receipt['files'])
    for name in PUBLIC:
        path=ROOT/name
        if path.suffix=='.py': compile(path.read_text(),str(path),'exec')
        elif path.suffix=='.slurm': subprocess.run(['bash','-n',str(path)],check=True,timeout=10)
        files[str(path)]=sha(path)
    original=assembly.pipeline.pipeline_function('merge')
    if original.__code__ is not assembly.pipeline.previous.merge.__code__:
        raise ValueError('The complete original full108 merge body must be reused')
    checked=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_targeted_parallel_lifecycle_v1'],cwd=ROOT,
        capture_output=True,text=True,timeout=60)
    if checked.returncode: raise RuntimeError(checked.stdout+checked.stderr)
    for path in (worker.cached.BINDING,receipt_path,parity/'summary-private.json',
            parity/'scheduler-terminal-private.txt',tests,
            ROOT/'logs/targeted-case-partition-check-v1-326733-terminal-private.txt'):
        files[str(path)]=sha(path)
    if any(sha(path) != expected for path,expected in files.items()):
        raise ValueError('Actual numeric or execution preparation input changed')
    worker.RUN.mkdir(parents=True,exist_ok=False)
    write(worker.BINDING,dict(status='same_numeric_producer_parallel_case_entry_bound_not_dispatched',
        files=files,source4_only=True,maximum_parallel_case_workers=3,maximum_collection_gpus=4,
        complete_original108_and_original_partitions_required=True,
        original_numeric_case_body_unchanged=True,original_full108_merge_body_unchanged=True,
        actual_case_partition_checks=6,actual_case_order_world_allocation_matches=96,
        actual_new_lifecycle_boundary_checks=3,lifecycle_check_output=checked.stdout+checked.stderr,
        new_simulations=0,model_loaded=False,student_updates=0,
        original_live_jobs_not_cancelled=True,held_queue_requires_scheduler_handoff=True,allocation=allocation))
    print('Bound complete same-producer case-parallel lifecycle; no old jobs changed',flush=True)


def submit(entry,args,dependencies=()):
    command=['sbatch','--parsable','--hold','--no-requeue','--kill-on-invalid-dep=yes']
    if dependencies: command.append('--dependency='+','.join(dependencies))
    command.extend([str(entry),*[str(value) for value in args]])
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
    if process.returncode: raise RuntimeError(process.stderr)
    job=process.stdout.strip().split(';')[0]
    if not job.isdigit(): raise ValueError('An actual scheduler job ID is required')
    return job


def describe(job,kind,entry,folder=None,**extra):
    actual=scheduler(job); v=fields(actual.stdout)
    if (actual.returncode or v.get('JobId') != job or v.get('Command') != str(entry)
            or v.get('JobState') != 'PENDING' or v.get('RunTime') != '00:00:00'
            or v.get('AllocTRES') not in ('(null)','')
            or not v.get('UserId','').endswith(f'({os.getuid()})')):
        raise ValueError('Actual own pending never-allocated queue job required')
    return dict(job=job,kind=kind,folder=str(folder) if folder else None,
        entry=str(entry),observed_pending_raw=actual.stdout,**extra)


def freeze_queue():
    allocation=resource(); worker.verify_binding()
    queue_path=worker.RUN/'queue-private.json'
    if worker.DISPATCH.exists() or queue_path.exists(): raise ValueError('Never overwrite an actual dispatch or partial queue')
    regpath=worker.previous.RUN/'registered-private.json'; reg=read(regpath); specs=reg['selected'][81:108]
    recovered={}; files={}; terminals=[]
    for folder,archive in ((ORIGINAL,OLD_ARCHIVE),(LAST,LAST_ARCHIVE)):
        terminal=worker.cached.require_terminal(folder,archive)
        terminals.append(terminal)
        incoming,bound=worker.previous.completed_from(folder,specs,reg)
        worker.previous.combine(recovered,incoming); worker.combine_files(files,bound)
        files[str(archive)]=sha(archive)
    assignments=worker.partition.partition_missing(specs,set(recovered),3)
    files[str(regpath)]=sha(regpath); files[str(worker.BINDING)]=sha(worker.BINDING)
    dispatch=dict(status='actual_ended_source4_missing_cases_frozen_for_disjoint_workers',
        shard=3,selected=specs,completed_source_ids=sorted(recovered),assignments=assignments,
        original_priors=[str(ORIGINAL),str(LAST)],original_archives=[str(OLD_ARCHIVE),str(LAST_ARCHIVE)],
        original_terminals=terminals,collection_sha256=sha(regpath),worker_binding_sha256=sha(worker.BINDING),
        numeric_producer_binding_sha256=sha(worker.cached.BINDING),files=files,
        assignment_depends_only_on_original_case_order_and_complete_source_ids=True,
        original_worlds_teacher_targets_and_partitions_unchanged=True,
        maximum_parallel_case_workers=3,maximum_collection_gpus=4,allocation=allocation)
    write(worker.DISPATCH,dispatch)
    del recovered
    jobs=[]; descriptors=[]; final_folders=[]; final_archives=[]; final_caps=[]

    def add(entry,args,kind,folder=None,dependencies=(),**extra):
        job=submit(entry,args,dependencies); jobs.append(job)
        descriptors.append(describe(job,kind,entry,folder,**extra))
        return job

    def terminal_capture(job,kind):
        return add(capture.ENTRY,['--kind',kind,'--job',job],'capture',
            dependencies=('afterany:'+job,),producer_job=job,producer_kind=kind)

    try:
        for lane,ids in enumerate(assignments):
            prior_folders=[]; prior_archives=[]; predecessor=None
            for attempt in (1,2):
                folder=worker.RUN/f'lane-{lane:02d}'/f'attempt-{attempt:03d}'
                args=['--lane',lane,'--attempt',attempt]
                for path in prior_folders: args.extend(['--prior',path])
                for path in prior_archives: args.extend(['--archive',path])
                dependencies=('afterok:'+predecessor,) if predecessor else ()
                job=add(worker.ENTRY,args,'worker',folder,dependencies,lane=lane,attempt=attempt,
                    assigned_source_ids=ids)
                cap=terminal_capture(job,'worker')
                prior_folders.append(folder); prior_archives.append(capture.archive_path(job)); predecessor=cap
            final_folders.append(folder); final_archives.append(capture.archive_path(job)); final_caps.append(cap)
        args=[]
        for path in final_folders: args.extend(['--folder',path])
        for path in final_archives: args.extend(['--archive',path])
        assembly_job=add(assembly.ENTRY,args,'assembly',assembly.FULL,
            tuple('afterok:'+job for job in final_caps))
        assembly_cap=terminal_capture(assembly_job,'assembly')
        folders=[worker.cached.RUN/'shard-00/attempt-001',worker.cached.RUN/'shard-01/attempt-004',
            worker.cached.RUN/'shard-02/attempt-001',assembly.FULL]
        archives=[ROOT/'runs/curriculum-targeted-shard-terminals-v3/v3-shard00-attempt001-job322941-terminal-private.json',
            ROOT/'runs/curriculum-targeted-shard-terminals-v3/v3-shard01-attempt004-job324265-terminal-private.json',
            ROOT/'runs/curriculum-targeted-shard-terminals-v3/v3-shard02-attempt001-job324257-terminal-private.json',
            capture.archive_path(assembly_job)]
        args=[]
        for path in folders: args.extend(['--folder',path])
        for path in archives: args.extend(['--archive',path])
        merge_job=add(assembly.MERGE_ENTRY,args,'merge',
            dependencies=('afterok:'+assembly_cap,'afterok:324258'))
        merge_cap=terminal_capture(merge_job,'merge')
        write(queue_path,dict(status='actual_disjoint_case_jobs_queued_held_for_scheduler_handoff',
            descriptors=descriptors,jobs=jobs,assembly=assembly_job,assembly_capture=assembly_cap,
            merge=merge_job,merge_capture=merge_cap,dispatch_sha256=sha(worker.DISPATCH),
            binding_sha256=sha(worker.BINDING),all_jobs_initially_held=True,
            old_training_dependencies_not_changed=True,old_future_jobs_not_cancelled=True,
            old_numeric_recipe_not_changed=True,actual_new_cases_not_yet_collected=True,
            final_frozen_v8_confirmation_not_opened=True,allocation=allocation))
        print(json.dumps(dict(status='actual_disjoint_case_queue_held_ready_for_handoff',
            completed_parent_cases=len(dispatch['completed_source_ids']),
            assigned_case_counts=[len(ids) for ids in assignments],jobs=jobs,
            merge=merge_job,assembly=assembly_job)),flush=True)
    except BaseException:
        for job in reversed(jobs):
            subprocess.run(['scancel',job],capture_output=True,text=True,timeout=20)
        write(worker.RUN/'queue-failure-private.json',dict(actual_created_jobs=jobs,
            cancelled_only_jobs_created_by_this_submission=True,original_future_queue_not_changed=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',choices=('bind','freeze-queue'),required=True)
    args=p.parse_args(); (bind if args.phase=='bind' else freeze_queue)()
