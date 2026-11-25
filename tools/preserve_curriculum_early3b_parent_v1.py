"""Preserve the actual3B capture terminal for the later placement continuation.

Finite GPU preparation/capture only. This never loads a model, admits a parent,
opens confirmation games, submits placement training or changes3B parameters.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_placement_early_course_v1 as course
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields,terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

CONTROL=ROOT/'runs/curriculum-early3B-parent-preservation-v1'
BINDING=CONTROL/'source-binding-private.json'
QUEUE=CONTROL/'queue-private.json'
ENTRY=ROOT/'cluster/curriculum-early3b-parent-preservation-v1.slurm'
PUBLIC=('tools/preserve_curriculum_early3b_parent_v1.py',
    'cluster/curriculum-early3b-parent-preservation-v1.slurm')


def verify(path):
    saved=read(path)
    if any(sha(name)!=value for name,value in saved['files'].items()):
        raise ValueError('Bound preservation/course sources changed')
    return saved


def bind():
    allocation=resource()
    if BINDING.exists() or course.BINDING.exists():raise ValueError('Never overwrite source preparation')
    tests=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_placement_early_parent_v1',
        'tests.core.test_curriculum_placement_early_course_v1'],cwd=ROOT,
        capture_output=True,text=True,timeout=90)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    subprocess.run(['bash','-n',str(ENTRY)],check=True,timeout=10)
    course.parent.require_lineage(read(course.parent.PARENT))
    if course.REGISTERED.exists() or (ROOT/course.IDENTITY['out']).exists():
        raise ValueError('Prepare source before any actual placement registration')
    public=set((*course.registration_function().__globals__['PUBLIC_FILES'],
        *course.integration_function().__globals__['PUBLIC_FILES'],*course.parent.producer.PUBLIC,
        *course.parent.PUBLIC,*course.PUBLIC,*PUBLIC))
    # Bind reused numeric/parent source as well as the explicit new adapters.
    public.update(('tools/curriculum_placement_balanced_course_v1.py',
        'tools/curriculum_placement_retained_parent_v1.py',
        'tools/curriculum_placement_parent_runtime_v1.py',
        'tools/curriculum_placement_parent_loss_v2.py',
        'tools/train_curriculum_placement_parent_v1.py',
        'tools/audit_curriculum_placement_parent_v1.py',
        'tools/register_curriculum_placement_parent_v1.py',
        'tools/integrate_curriculum_placement_balanced_v1.py'))
    files={str((ROOT/name).resolve()):sha(ROOT/name) for name in sorted(public)}
    course.BINDING.parent.mkdir(parents=True,exist_ok=False)
    summary=dict(status='early_route_placement_source_ready_not_parent_or_fit_admitted',
        actual_parent_admitted=False,actual_placement_fit_registered=False,student_updates=0,new_games=0,
        actual_parent_3A_step=1800,actual_parent_3B_step_required=2500,placement_stage_offset=5296,
        intermediate_grade_is_report_not_veto=True,all57_sources_and_exact_two_epoch_recipe_preserved=True,
        final_v8_confirmation_opened=False,allocation=allocation,test_output=tests.stdout+tests.stderr)
    summary_path=course.BINDING.parent/'summary-private.json';write(summary_path,summary)
    files[str(summary_path.resolve())]=sha(summary_path)
    write(course.BINDING,dict(files=files,actual_model_or_data_or_fit_admitted=False))
    CONTROL.mkdir(parents=True,exist_ok=False)
    files=dict(files);files[str(course.BINDING.resolve())]=sha(course.BINDING)
    files[str(course.parent.QUEUE.resolve())]=sha(course.parent.QUEUE)
    write(BINDING,dict(files=files,allocation=allocation,actual_placement_fit_started=False,
        automatic_release=False,final_confirmation_opened=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('allocation','test_output')}),flush=True)


def submit():
    resource();verify(BINDING)
    if QUEUE.exists() or course.parent.ARCHIVE.exists():raise ValueError('No duplicate producer capture preservation')
    parent_queue=read(course.parent.QUEUE);fit=scheduler(parent_queue['fit_job']);cap=scheduler(parent_queue['capture_job'])
    if (fit.returncode or fields(fit.stdout).get('JobState')!='RUNNING'
            or fields(fit.stdout).get('Command')!=str(course.parent.ENTRY)
            or cap.returncode or fields(cap.stdout).get('JobState')!='PENDING'
            or fields(cap.stdout).get('Command')!=str(course.parent.ENTRY)):
        raise ValueError('Bind to the actual running3B fit and its pending own capture')
    job=None;released=False
    try:
        result=subprocess.run(['sbatch','--hold','--parsable',
            '--dependency=afterany:'+str(parent_queue['capture_job']),
            '--kill-on-invalid-dep=yes',str(ENTRY)],cwd=ROOT,
            capture_output=True,text=True,check=True,timeout=30)
        job=result.stdout.strip().split(';')[0];held=scheduler(job);v=fields(held.stdout)
        if (not job.isdigit() or held.returncode or v.get('JobState')!='PENDING'
                or v.get('Reason')!='JobHeldUser' or v.get('Command')!=str(ENTRY)
                or v.get('TimeLimit')!='00:05:00'
                or f'afterany:{parent_queue["capture_job"]}(' not in v.get('Dependency','')):
            raise ValueError('Exact held finite producer capture preservation required')
        write(QUEUE,dict(job=job,parent_capture_job=str(parent_queue['capture_job']),
            actual_fit=fit.stdout,actual_capture=cap.stdout,actual_held_job=held.stdout,
            binding_sha256=sha(BINDING),parent_queue_sha256=sha(course.parent.QUEUE)))
        subprocess.run(['scontrol','release',job],check=True,capture_output=True,text=True,timeout=20)
        released=True
        print(json.dumps(dict(status='finite_actual3B_capture_preservation_submitted',job=job,
            afterany=parent_queue['capture_job'],placement_fit_started=False)),flush=True)
    finally:
        if job and job.isdigit() and not released:
            subprocess.run(['scancel',job],capture_output=True,timeout=20)


def capture():
    allocation=resource();verify(BINDING);q=read(QUEUE)
    if (str(q['job'])!=os.environ.get('SLURM_JOB_ID') or q['binding_sha256']!=sha(BINDING)
            or q['parent_queue_sha256']!=sha(course.parent.QUEUE)):
        raise ValueError('Exact queued producer-preservation identity required')
    if course.parent.ARCHIVE.exists():raise ValueError('Never overwrite the actual producer terminal')
    job=str(q['parent_capture_job']);actual=scheduler(job)
    if actual.returncode:raise ValueError('Actual producer capture terminal expired; never reconstruct it')
    receipt=read(ROOT/'logs'/f'{course.parent.PREFIX}{job}.json')
    if receipt.get('phase')!='capture':raise ValueError('Actual producer capture phase required')
    terminal_record(actual.stdout,job,course.parent.ENTRY)
    scheduler_exit(receipt,actual.stdout,course.parent.ENTRY)
    write(course.parent.ARCHIVE,dict(raw=actual.stdout,exit=receipt))
    write(CONTROL/'summary-private.json',dict(status='actual_early3B_capture_terminal_preserved',
        capture_job=job,allocation=allocation,student_updates=0,model_loaded=False,
        placement_parent_admitted=False,placement_fit_started=False,final_confirmation_opened=False))
    print(json.dumps(dict(status='actual_early3B_capture_terminal_preserved',capture_job=job)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('bind','submit','capture'),required=True)
    a=p.parse_args();globals()[a.phase]()
