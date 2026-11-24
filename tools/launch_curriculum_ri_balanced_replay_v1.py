"""Finite dependency-gated standby launch; no polling or automatic promotion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields,terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.curriculum_ri_balanced_replay_runtime_v1 import RUN,CONTROL,require_registration,require_launch
from tools.register_curriculum_ri_balanced_replay_v1 import authorize

ENTRY=ROOT/'cluster/curriculum-ri-balanced-replay-v1.slurm'
REGISTERED=RUN/'fit-plan/registered.json'
BRIDGE_QUEUE=RUN/'bridge-queue-private.json'
FIT_QUEUE=RUN/'fit-queue-private.json'


def submit(arguments):
    result=subprocess.run(['sbatch','--parsable',*arguments,str(ENTRY)],cwd=ROOT,
        capture_output=True,text=True,timeout=30)
    value=result.stdout.strip().split(';')[0]
    if result.returncode or not value.isdigit():raise RuntimeError(result.stderr or result.stdout)
    return value


def verify_files(binding):
    if any(sha(p)!=h for p,h in binding['files'].items()):raise ValueError('Bound source or evidence changed')


def enqueue_bridge():
    allocation=resource();require_registration(read(REGISTERED))
    if BRIDGE_QUEUE.exists() or FIT_QUEUE.exists():raise ValueError('No duplicate bridge/fit submission')
    parent=read(ROOT/'runs/curriculum-progression-riichi-paired-v1/parent-retention-queue-v1-private.json')
    live=scheduler(parent['fit_job']);capture=scheduler(parent['capture_job'])
    fv,cv=fields(live.stdout),fields(capture.stdout)
    if (live.returncode or capture.returncode or fv.get('JobState')!='RUNNING'
            or cv.get('JobState')!='PENDING'
            or cv.get('Command')!=str(ROOT/'cluster/curriculum-parent-retention-terminal-v1.slurm')
            or any(f'afterany:{parent[r+"_job"]}(' not in cv.get('Dependency','') for r in ('fit','audit'))):
        raise ValueError('Bind finite bridge while actual control is live with exact pending capture')
    job=None
    try:
        job=submit(['--hold','--job-name=jev-ri-replay-bridge','--time=00:05:00',
            '--cpus-per-task=1','--mem=4G','--dependency=afterany:'+str(parent['capture_job']),
            '--kill-on-invalid-dep=yes','--export=ALL,JEV_RI_REPLAY_PHASE=bridge'])
        actual=scheduler(job);v=fields(actual.stdout)
        if (actual.returncode or v.get('JobState')!='PENDING' or v.get('Reason')!='JobHeldUser'
                or v.get('Command')!=str(ENTRY) or v.get('TimeLimit')!='00:05:00'
                or f'afterany:{parent["capture_job"]}(' not in v.get('Dependency','')):
            raise ValueError('Held finite bridge contract differs')
        bound=read(RUN/'launch-binding-private.json')
        files=dict(bound['files']);files[str(RUN/'launch-binding-private.json')]=sha(RUN/'launch-binding-private.json')
        write(BRIDGE_QUEUE,dict(status='finite_standby_bridge_bound_before_control_completion',
            bridge_job=job,parent_jobs=parent,observed_live_fit=live.stdout,
            observed_capture=capture.stdout,observed_held_bridge=actual.stdout,
            files=files,allocation=allocation,student_updates=0))
        subprocess.run(['scontrol','release',job],check=True,capture_output=True,text=True,timeout=20)
        print(json.dumps({'status':'finite_standby_bridge_released','bridge_job':job}),flush=True)
    except BaseException:
        if job:subprocess.run(['scancel',job],check=False,capture_output=True,text=True,timeout=20)
        raise


def bridge():
    allocation=resource();bound=read(BRIDGE_QUEUE);verify_files(bound)
    if str(bound['bridge_job'])!=os.environ['SLURM_JOB_ID']:raise ValueError('Wrong finite bridge allocation')
    if (RUN/'bridge-decision-private.json').exists() or FIT_QUEUE.exists():raise ValueError('No duplicate bridge decision')
    captured=read(CONTROL/'parent-retention-terminal-v1/terminal-evidence-private.json')
    grade=captured.get('original_strategy_grade')
    if grade and grade.get('passed') is True:
        write(RUN/'bridge-decision-private.json',dict(status='standby_skipped_control_qualified',
            unchanged_grade=grade,allocation=allocation,student_updates=0))
        print(json.dumps({'status':'standby_skipped_control_qualified'}),flush=True);return
    authorize()  # Refuses incomplete, unsuccessful, un-audited or stale controls.
    require_launch(REGISTERED);jobs=[]
    try:
        fit=submit(['--hold','--export=ALL,JEV_RI_REPLAY_PHASE=fit']);jobs.append(fit)
        audit=submit(['--job-name=jev-ri-replay-audit','--time=02:00:00',
            '--dependency=afterok:'+fit,'--kill-on-invalid-dep=yes',
            '--export=ALL,JEV_RI_REPLAY_PHASE=audit,JEV_RI_REPLAY_FIT_JOB='+fit]);jobs.append(audit)
        cap=submit(['--job-name=jev-ri-replay-terminal','--time=00:05:00','--cpus-per-task=1','--mem=4G',
            '--dependency=afterany:'+fit+':'+audit,'--kill-on-invalid-dep=yes',
            '--export=ALL,JEV_RI_REPLAY_PHASE=capture']);jobs.append(cap)
        observed={r:scheduler(j) for r,j in [('fit',fit),('audit',audit),('capture',cap)]}
        if any(x.returncode for x in observed.values()):raise ValueError('Cannot verify actual child job dependencies')
        fv,av,cv=(fields(observed[r].stdout) for r in ('fit','audit','capture'))
        if (fv.get('Reason')!='JobHeldUser' or fv.get('JobState')!='PENDING'
                or any(v.get('Command')!=str(ENTRY) for v in (fv,av,cv))
                or f'afterok:{fit}(' not in av.get('Dependency','')
                or any(f'afterany:{j}(' not in cv.get('Dependency','') for j in (fit,audit))):
            raise ValueError('Held fit/audit/terminal actual scheduler contract differs')
        files=dict(bound['files'])
        for p in (BRIDGE_QUEUE,RUN/'launch-gate-private.json'):files[str(p)]=sha(p)
        write(FIT_QUEUE,dict(status='actual_failed_control_standby_jobs_bound_while_held',
            fit_job=fit,audit_job=audit,capture_job=cap,files=files,allocation=allocation,
            actual_observations={r:x.stdout for r,x in observed.items()}))
        subprocess.run(['scontrol','release',fit],check=True,capture_output=True,text=True,timeout=20)
        write(RUN/'bridge-decision-private.json',dict(status='one_separate_replay_fit_released',
            fit_job=fit,audit_job=audit,capture_job=cap,allocation=allocation))
        print(json.dumps({'status':'one_separate_replay_fit_released','fit_job':fit,'audit_job':audit,'capture_job':cap}),flush=True)
    except BaseException as error:
        cleanup=[]
        for job in jobs:
            r=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=20)
            cleanup.append(dict(job_id=job,returncode=r.returncode,stderr=r.stderr))
        write(RUN/'bridge-launch-failure-private.json',dict(error=repr(error),cleanup=cleanup,allocation=allocation))
        raise


def fit():
    resource();queue=read(FIT_QUEUE);verify_files(queue)
    if str(queue['fit_job'])!=os.environ['SLURM_JOB_ID']:raise ValueError('Wrong explicitly bound fit job')
    from tools.train_curriculum_ri_balanced_replay_v1 import train
    train(REGISTERED)


def audit():
    resource();queue=read(FIT_QUEUE);verify_files(queue)
    if (str(queue['audit_job'])!=os.environ['SLURM_JOB_ID']
            or str(queue['fit_job'])!=os.environ['JEV_RI_REPLAY_FIT_JOB']):raise ValueError('Wrong bound independent audit')
    job=queue['fit_job'];observed=scheduler(job)
    receipt=read(ROOT/'logs'/f'curriculum-ri-balanced-replay-exit-{job}.json')
    if observed.returncode or receipt.get('phase')!='fit':raise ValueError('Actual completed fit and exit required')
    scheduler_exit(receipt,observed.stdout,ENTRY)
    write(RUN/'fit-terminal-scheduler-private.json',dict(raw=observed.stdout,exit=receipt))
    from tools.audit_curriculum_ri_balanced_replay_v1 import audit as check
    check(REGISTERED,RUN/'final-evidence-private.json')
    write(RUN/'audit-execution-private.json',dict(status='actual_balanced_replay2472_full_evidence_independently_audited',
        actual_fit_job=job,automatic_release=False,automatic_stage3B=False))


def capture():
    allocation=resource();queue=read(FIT_QUEUE);verify_files(queue)
    if str(queue['capture_job'])!=os.environ['SLURM_JOB_ID']:raise ValueError('Wrong bound terminal job')
    out=RUN/'terminal-v1';out.mkdir(exist_ok=False);results={}
    for role in ('fit','audit'):
        job=queue[role+'_job'];actual=scheduler(job)
        raw=actual.stdout
        if actual.returncode:
            if role!='fit':raise ValueError('Actual audit scheduler terminal expired before capture')
            prior=read(RUN/'fit-terminal-scheduler-private.json');raw=prior['raw']
        record=terminal_record(raw,job,ENTRY)
        receipt_path=ROOT/'logs'/f'curriculum-ri-balanced-replay-exit-{job}.json'
        receipt=read(receipt_path) if receipt_path.exists() else None
        if record['successful']:
            if receipt is None or receipt.get('phase')!=role:raise ValueError('Exact successful phase receipt required')
            scheduler_exit(receipt,raw,ENTRY)
        record['exit_receipt']=receipt;results[role]=record
        write(out/(role+'-terminal-private.json'),record)
    evidence=RUN/'final-evidence-private.json'
    write(out/'terminal-evidence-private.json',dict(status='balanced_replay_actual_terminals_preserved_not_promoted',
        scheduler_proofs=results,strategy_grade=(read(evidence)['stage3_advancement'] if evidence.exists() else None),
        allocation=allocation,automatic_stage3B=False,automatic_release=False))
    paths=[FIT_QUEUE,*out.glob('*-private.json')]
    write(out/'completion-binding-private.json',dict(files={str(p):sha(p) for p in paths}))
    print(json.dumps({'status':'balanced_replay_actual_terminals_preserved_not_promoted'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('enqueue-bridge','bridge','fit','audit','capture'),required=True)
    a=p.parse_args()
    {'enqueue-bridge':enqueue_bridge,'bridge':bridge,'fit':fit,'audit':audit,'capture':capture}[a.phase]()
