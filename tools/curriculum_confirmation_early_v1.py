"""Consume a genuinely sealed final candidate from the active early3B route.

This is a source/lineage adapter, not a seal producer or permission to open
reserved games. Keep original native accounting, source and joint statistics.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_confirmation_canonical_v2 as previous
from tools import curriculum_placement_early_course_v1 as course
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource

original=previous.original
runner=previous.runner
auditor=previous.auditor
finisher=previous.finisher
REGISTERED=ROOT/'runs/curriculum-final-confirmation-early-v1/registered.json'
RUN_ENTRY=ROOT/'cluster/curriculum-confirmation-early-run-v1.slurm'
AUDIT_ENTRY=ROOT/'cluster/curriculum-confirmation-early-audit-v1.slurm'
BINDING=ROOT/'runs/curriculum-confirmation-early-consumer-v1/source-binding-private.json'
PUBLIC=('tools/curriculum_confirmation_early_v1.py',
    'cluster/curriculum-confirmation-early-run-v1.slurm',
    'cluster/curriculum-confirmation-early-audit-v1.slurm',
    'tests/core/test_curriculum_confirmation_early_v1.py',
    'docs/experiments/plans/curriculum-confirmation-early-consumer-20261125.md',
    *previous.PUBLIC)


def scoped(function):
    names=('ROOT','course','REGISTERED','PUBLIC','require_route','verify_function',
        'verify','run_function','audit_function','finish_function')
    scope=dict(function.__globals__,**{name:globals()[name] for name in names})
    return FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)


def require_route(path,registration):return scoped(previous.require_route)(path,registration)
def verify_function():return previous.verify_function()
def verify(reg,registered):return scoped(previous.verify)(reg,registered)
def run_function():return scoped(previous.run_function)()
def audit_function():return scoped(previous.audit_function)()


def finish_function():
    code=previous.remap(finisher.finish,names={
        'tools.audit_curriculum_confirmation_v1':'tools.curriculum_confirmation_early_v1'},
        literals={'curriculum-confirmation-run-exit-':'curriculum-confirmation-early-run-exit-',
            'cluster/curriculum-confirmation-run-v1.slurm':'cluster/curriculum-confirmation-early-run-v1.slurm'})
    return FunctionType(code,dict(finisher.finish.__globals__),finisher.finish.__name__,
        finisher.finish.__defaults__,finisher.finish.__closure__)


def run(registered=REGISTERED):return run_function()(Path(registered).resolve())
def audit(registered,out):return audit_function()(Path(registered).resolve(),Path(out).resolve())
def finish(registered,run_job):return finish_function()(Path(registered).resolve(),run_job)


def preflight():
    allocation=resource()
    if BINDING.exists() or REGISTERED.exists():raise ValueError('Never overwrite a consumer binding or actual seal')
    result=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_confirmation_early_v1'],cwd=ROOT,
        capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    for entry in (RUN_ENTRY,AUDIT_ENTRY):subprocess.run(['bash','-n',str(entry)],check=True,timeout=10)
    try:verify({},REGISTERED)
    except ValueError:pass
    else:raise ValueError('An unsealed plan cannot admit any final game')
    files={}
    course_binding=read(course.BINDING)
    if any(sha(path)!=digest for path,digest in course_binding['files'].items()):
        raise ValueError('Actual course source binding changed')
    files.update(course_binding['files']);files[str(course.BINDING.resolve())]=sha(course.BINDING)
    paths=(*[ROOT/name for name in (*PUBLIC,*course.PUBLIC,*course.parent.PUBLIC)],
        Path(original.__file__),Path(runner.__file__),Path(auditor.__file__),Path(finisher.__file__),
        ROOT/'tools/curriculum_placement_parent_runtime_v1.py')
    for path in paths:files[str(path.resolve())]=sha(path)
    BINDING.parent.mkdir(parents=True,exist_ok=False)
    summary=dict(status='actual_early_route_final_consumer_source_ready_not_sealed_or_opened',
        actual_final_registration_created=False,actual_candidate_admitted=False,
        original_final_criteria_native_source_and_statistics_unchanged=True,
        actual_full_parent_and_complete_source_closure_still_required=True,
        reserved_games_generated=0,model_loaded=False,student_updates=0,
        allocation=allocation,test_output=result.stdout+result.stderr)
    path=BINDING.parent/'summary-private.json';write(path,summary);files[str(path.resolve())]=sha(path)
    write(BINDING,dict(files=files,actual_candidate_admitted=False,final_games_opened=False))
    print({k:v for k,v in summary.items() if k not in ('allocation','test_output')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('preflight','run','finish'),required=True)
    p.add_argument('--registered',type=Path,default=REGISTERED);p.add_argument('--run-job',type=int)
    a=p.parse_args()
    if a.phase=='preflight':preflight()
    elif a.phase=='run':run(a.registered)
    else:
        if a.run_job is None:p.error('--run-job is required')
        finish(a.registered,a.run_job)
