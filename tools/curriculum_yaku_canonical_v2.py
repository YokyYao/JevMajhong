"""Qualified canonical-replay2472 -> unchanged yaku2500 continuation.

Reuse numerical function code objects. Explicitly adapt parent identity,
registration/receipt/output paths and internal lifecycle module imports only.
No failed grade, interim peak or missing final can admit a child.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from types import CodeType, FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_yaku_balanced_parent_v1 as old_parent
from tools import register_curriculum_yaku_balanced_v1 as old_registration
from tools import train_curriculum_yaku_balanced_v1 as old_training
from tools import audit_curriculum_yaku_balanced_v1 as old_auditing
from tools import launch_curriculum_yaku_balanced_v1 as old_launch
from tools import curriculum_ri_balanced_replay_v2 as canonical
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

METHOD='audited_canonical_replay2472_same_policy_yaku_continuation_v2'
STATUS='actual_canonical_replay2472_final_sealed_for_yaku_course'
RUN=ROOT/'runs/curriculum-yaku-canonical-parent-v2'
PARENT_RUN=canonical.RUN
CONFIG=ROOT/'configs/training/curriculum-yaku-canonical-fit-v2.json'
ENTRY=ROOT/'cluster/curriculum-yaku-canonical-v2.slurm'
BINDING=PARENT_RUN/'canonical-yaku-entry-binding-v2-private.json'
QUEUE=PARENT_RUN/'canonical-yaku-queue-v2-private.json'
BRIDGE_QUEUE=PARENT_RUN/'canonical-yaku-bridge-queue-v2-private.json'
PREFLIGHT=ROOT/'runs/curriculum-canonical-yaku-course-v2/preflight-v2'
ARCHIVE='canonical-yaku-parent-capture-terminal-private.json'
PUBLIC=(
    'tools/curriculum_yaku_canonical_v2.py',
    'tests/core/test_curriculum_yaku_canonical_v2.py',
    'configs/training/curriculum-yaku-canonical-fit-v2.json',
    'cluster/curriculum-yaku-canonical-v2.slurm',
    'docs/experiments/plans/curriculum-yaku-canonical-continuation-20261125.md',
)
STRINGS={
    'runs/curriculum-ri-balanced-replay-fit-v1':'runs/curriculum-ri-balanced-replay-fit-v2',
    'cluster/curriculum-ri-balanced-replay-v1.slurm':'cluster/curriculum-ri-balanced-replay-v2.slurm',
    'curriculum-ri-balanced-replay-exit-':'curriculum-ri-balanced-replay-v2-exit-',
    'runs/curriculum-yaku-balanced-parent-v1':'runs/curriculum-yaku-canonical-parent-v2',
    'jev-curriculum-yaku-balanced-parent-v1':'jev-curriculum-yaku-canonical-parent-v2',
    'stage3B-yaku-balanced-parent-v1':'stage3B-yaku-canonical-parent-v2',
    'curriculum-yaku-balanced-exit-':'curriculum-yaku-canonical-v2-exit-',
    'actual_failed_control_standby_jobs_bound_while_held':'canonical_retry_jobs_bound_while_held',
    'capture-terminal-for-yaku-private.json':ARCHIVE,
    'balanced_yaku_full_entry_preflight_passed_not_parent_or_fit_admitted':'canonical_yaku_full_entry_preflight_passed_not_parent_or_fit_admitted',
    'balanced_yaku_entry_bound_prospectively_not_parent_or_fit_admitted':'canonical_yaku_entry_bound_prospectively_not_parent_or_fit_admitted',
}
IMPORTS={
    'tools.curriculum_ri_balanced_replay_runtime_v1':'tools.curriculum_ri_balanced_replay_v2',
    'tools.register_curriculum_yaku_balanced_v1':'tools.curriculum_yaku_canonical_v2',
    'tools.train_curriculum_yaku_balanced_v1':'tools.curriculum_yaku_canonical_v2',
    'tools.audit_curriculum_yaku_balanced_v1':'tools.curriculum_yaku_canonical_v2',
}


def adapt_code(code):
    constants=[]
    for value in code.co_consts:
        if isinstance(value,CodeType):value=adapt_code(value)
        elif isinstance(value,str):
            for old,new in STRINGS.items():value=value.replace(old,new)
        constants.append(value)
    return code.replace(co_consts=tuple(constants),co_names=tuple(IMPORTS.get(name,name) for name in code.co_names))


def scoped(function,**overrides):
    return FunctionType(adapt_code(function.__code__),dict(function.__globals__,**overrides),
        function.__name__,function.__defaults__,function.__closure__)


require_qualified_parent=scoped(old_parent.require_qualified_parent,
    require_parent_registration=canonical.require_registration)
actual_terminals=scoped(old_parent.actual_terminals)
seal=scoped(old_parent.seal,METHOD=METHOD,STATUS=STATUS,__file__=__file__,
    require_qualified_parent=require_qualified_parent,require_launch=canonical.require_launch,
    actual_terminals=actual_terminals)
verify_parent=scoped(old_parent.verify_parent,METHOD=METHOD,STATUS=STATUS)


@contextmanager
def parent_loader(initialization):
    previous=initialization.verified_parent;initialization.verified_parent=verify_parent
    try:yield
    finally:initialization.verified_parent=previous


register=scoped(old_registration.register,PARENT_METHOD=METHOD,STATUS=STATUS,
    verify_parent=verify_parent,PUBLIC_FILES=(*old_registration.PUBLIC_FILES,*PUBLIC))
train=scoped(old_training.train,verify_parent=verify_parent,parent_loader=parent_loader)
audit=scoped(old_auditing.audit,verify_parent=verify_parent)
require_same_recipe=scoped(old_launch.require_same_recipe)

# Functions share their own isolated globals; old modules remain unmodified.
_scope=dict(old_launch.__dict__,RUN=RUN,PARENT_RUN=PARENT_RUN,CONFIG=CONFIG,ENTRY=ENTRY,
    BINDING=BINDING,QUEUE=QUEUE,PREFLIGHT=PREFLIGHT,BRIDGE_QUEUE=BRIDGE_QUEUE,
    PUBLIC=(*old_launch.PUBLIC,*PUBLIC),require_qualified_parent=require_qualified_parent,
    require_same_recipe=require_same_recipe,seal=seal)
_lifecycle={}
for _name in ('qualified_parent','verify_binding','bind','archive_parent_capture','submit',
              'bridge','prepare','train','audit','capture'):
    _function=getattr(old_launch,_name)
    _lifecycle[_name]=FunctionType(adapt_code(_function.__code__),_scope,
        _function.__name__,_function.__defaults__,_function.__closure__)
_scope.update(_lifecycle)


def preflight():
    allocation=resource()
    if PREFLIGHT.exists():raise ValueError('Never overwrite canonical continuation evidence')
    c=read(CONFIG);require_same_recipe(c,read(ROOT/'configs/training/curriculum-yaku-parent-fit-v1.json'))
    canonical.require_registration(read(PARENT_RUN/'fit-plan/registered.json'))
    previous=old_launch.PREFLIGHT/'completion-binding-private.json';binding=read(previous)
    if any(sha(p)!=h for p,h in binding['files'].items()):
        raise ValueError('Unchanged numerical yaku prerequisites changed')
    result=subprocess.run([sys.executable,'-m','unittest','tests.core.test_curriculum_yaku_canonical_v2'],
        capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    subprocess.run(['bash','-n',str(ENTRY)],check=True,timeout=15)
    try:_lifecycle['qualified_parent'](c)
    except FileNotFoundError as error:
        if Path(error.filename)!=PARENT_RUN/'stage3-fit/completion.json':raise
    else:raise ValueError('Expected the actual still-incomplete parent to be refused')
    if RUN.exists():raise ValueError('Prospective checks created a child run')
    PREFLIGHT.mkdir(parents=True)
    summary=dict(status='canonical_yaku_full_entry_preflight_passed_not_parent_or_fit_admitted',
        parent_method=METHOD,old_numerical_data_and_objective_preflight_reused=True,
        actual_missing_final_refused=True,student_updates=0,child_parent_admitted=False,
        allocation=allocation,test_output=result.stdout+result.stderr)
    write(PREFLIGHT/'summary-private.json',summary)
    files=dict(binding['files']);files[str(previous)]=sha(previous)
    for p in [ROOT/name for name in PUBLIC]+[PREFLIGHT/'summary-private.json',PARENT_RUN/'fit-plan/registered.json']:
        files[str(p)]=sha(p)
    write(PREFLIGHT/'completion-binding-private.json',dict(files=files))
    print(dict(status=summary['status'],student_updates=0),flush=True)


def enqueue_bridge():
    allocation=resource();_lifecycle['verify_binding']()
    if BRIDGE_QUEUE.exists() or QUEUE.exists():raise ValueError('No duplicate canonical child bridge')
    parent=read(PARENT_RUN/'fit-queue-private.json')
    fit,cap=(scheduler(parent[role+'_job']) for role in ('fit','capture'))
    fv,cv=fields(fit.stdout),fields(cap.stdout)
    if (fit.returncode or cap.returncode or fv.get('JobState')!='RUNNING'
            or cv.get('JobState')!='PENDING' or fv.get('Command')!=str(canonical.ENTRY)
            or cv.get('Command')!=fv.get('Command')
            or any(f'afterany:{parent[role+"_job"]}(' not in cv.get('Dependency','') for role in ('fit','audit'))):
        raise ValueError('Actual live canonical parent and its bound pending capture required')
    jobs=[]
    def submit(arguments):
        r=subprocess.run(['sbatch','--parsable','--hold','--time=00:05:00',
            '--cpus-per-task=1','--mem=4G',*arguments,str(ENTRY)],cwd=ROOT,
            capture_output=True,text=True,timeout=30)
        job=r.stdout.strip().split(';')[0]
        if r.returncode or not job.isdigit():raise RuntimeError(r.stderr or r.stdout)
        jobs.append(job);return job
    try:
        archive=submit(['--job-name=jev-canonical-yaku-archive',
            '--dependency=afterany:'+str(parent['capture_job']),'--kill-on-invalid-dep=yes',
            '--export=ALL,JEV_YAKU_PARENT_PHASE=archive-parent'])
        bridge=submit(['--job-name=jev-canonical-yaku-bridge','--dependency=afterok:'+archive,
            '--kill-on-invalid-dep=yes','--export=ALL,JEV_YAKU_PARENT_PHASE=bridge'])
        observations={role:scheduler(job) for role,job in (('archive',archive),('bridge',bridge))}
        for role,job,kind,predecessor in (('archive',archive,'afterany',str(parent['capture_job'])),
                                          ('bridge',bridge,'afterok',archive)):
            actual=observations[role];v=fields(actual.stdout)
            if (actual.returncode or v.get('JobId')!=job or v.get('Reason')!='JobHeldUser'
                    or v.get('Command')!=str(ENTRY) or v.get('TimeLimit')!='00:05:00'
                    or f'{kind}:{predecessor}(' not in v.get('Dependency','')):
                raise ValueError('Actual finite archive/bridge held graph differs')
        write(BRIDGE_QUEUE,dict(status='canonical_yaku_archive_bridge_bound_during_actual_parent_fit',
            archive_job=archive,bridge_job=bridge,parent_jobs=parent,binding_sha256=sha(BINDING),
            allocation=allocation,observed_fit=fit.stdout,observed_capture=cap.stdout,
            observed_held_jobs={role:r.stdout for role,r in observations.items()}))
        for job in (bridge,archive):
            subprocess.run(['scontrol','release',job],check=True,capture_output=True,text=True,timeout=20)
        print(dict(status='finite_canonical_parent_archive_and_bridge_released'),flush=True)
    except BaseException as error:
        cleanup=[]
        for job in jobs:
            r=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=20)
            cleanup.append(dict(job=job,returncode=r.returncode,stderr=r.stderr))
        write(PARENT_RUN/'canonical-yaku-enqueue-failure-private.json',dict(error=repr(error),cleanup=cleanup))
        raise


def archive_parent_terminal():
    resource();_lifecycle['verify_binding']();queue=read(BRIDGE_QUEUE)
    if queue['archive_job']!=os.environ['SLURM_JOB_ID'] or queue['binding_sha256']!=sha(BINDING):
        raise ValueError('Exact bound canonical parent archive required')
    job=queue['parent_jobs']['capture_job'];observed=scheduler(job)
    receipt_path=ROOT/'logs'/f'curriculum-ri-balanced-replay-v2-exit-{job}.json'
    receipt=read(receipt_path) if receipt_path.exists() else None
    write(PARENT_RUN/'canonical-yaku-parent-capture-observation-private.json',dict(
        stdout=observed.stdout,stderr=observed.stderr,returncode=observed.returncode,
        exit_receipt=receipt,parent_admitted=False))
    if observed.returncode or receipt is None or receipt.get('phase')!='capture':
        raise ValueError('Actual parent capture terminal and phase required')
    scheduler_exit(receipt,observed.stdout,canonical.ENTRY)
    write(PARENT_RUN/ARCHIVE,dict(raw=observed.stdout,exit=receipt))
    print(dict(status='actual_canonical_parent_capture_terminal_archived_not_parent_admitted'),flush=True)


def bind_and_enqueue():
    preflight();_lifecycle['bind']();enqueue_bridge()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('bind-enqueue','archive-parent','bridge','prepare','fit','audit','capture'),required=True)
    parser.add_argument('--fit-job',type=int)
    args=parser.parse_args()
    if args.phase=='bind-enqueue':bind_and_enqueue()
    elif args.phase=='archive-parent':archive_parent_terminal()
    elif args.phase=='fit':_lifecycle['train']()
    elif args.phase=='audit':
        if args.fit_job is None:parser.error('--fit-job is required for audit')
        _lifecycle['audit'](args.fit_job)
    else:_lifecycle[args.phase]()
