"""Finite placement jobs for the active selected1800 -> completed3B2500 route.

Reuse the existing full data/register/integration/train/audit lifecycle, with
the actual ancestry offset. Submission still requires real completed parent
evidence and the complete57-source corpus. No confirmation is opened here.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from types import CodeType,FunctionType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_placement_early_course_v1 as course
from tools import launch_curriculum_placement_balanced_v1 as previous
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler

CONTROL=ROOT/'runs/curriculum-placement-early-launch-v1'
RUN=course.RUN
ENTRY=ROOT/'cluster/curriculum-placement-early-course-v1.slurm'
BINDING=CONTROL/'source-binding-private.json'
QUEUE=CONTROL/'queue-private.json'
LAUNCH_CONFIG=RUN/'launch-config-private.json'
DATA=previous.DATA
ORIGINAL_DATA=previous.ORIGINAL_DATA
SEAL=RUN/'completed-yaku-parent-v1/parent-seal-private.json'
PREFIX='curriculum-placement-early-course-exit-'
ROLES=previous.ROLES
LIMITS=previous.LIMITS
PUBLIC=('tools/launch_curriculum_placement_early_v1.py',
    'cluster/curriculum-placement-early-course-v1.slurm',
    'tests/core/test_curriculum_placement_early_launch_v1.py',
    'docs/experiments/plans/curriculum-placement-early-launch-20261125.md')
integration=SimpleNamespace(loss=course.control.loss,BINDING=course.INTEGRATION_BINDING,
    REGISTERED=course.REGISTERED,OUT=course.INTEGRATION_OUT,PUBLIC=course.PUBLIC,
    integration_function=course.integration_function,integrate=course.integrate)


def scoped(function):
    allowed=(previous.prepare_tree,previous.require_integration,previous.prepare_function,
        previous.verify_entry,previous.receipt,previous.archive_success,previous.verify_queue,
        previous.submit,previous.run)
    if function not in allowed:raise ValueError('Review the exact lifecycle function before adapting')
    names=('course','integration','CONTROL','RUN','ENTRY','BINDING','QUEUE','LAUNCH_CONFIG',
        'DATA','ORIGINAL_DATA','SEAL','PREFIX','ROLES','LIMITS','PUBLIC','prepare_tree',
        'require_integration','prepare_function','verify_entry','receipt','archive_success',
        'verify_queue','submit','run','read','sha','resource','scheduler','subprocess','write')
    scope=dict(function.__globals__,**{name:globals()[name] for name in names})
    if function is previous.require_integration:return course.replace_offset(function,**scope)
    code=function.__code__
    if function is previous.run:
        replacements={'placement-balanced-terminal-v1':'placement-early-terminal-v1',
            'actual_balanced_placement_terminals_preserved_not_confirmed':
                'actual_early_route_placement_terminals_preserved_not_confirmed'}
        counts=dict.fromkeys(replacements,0)
        def literals(c):
            values=[]
            for value in c.co_consts:
                if isinstance(value,CodeType):value=literals(value)
                elif isinstance(value,str) and value in replacements:
                    counts[value]+=1;value=replacements[value]
                values.append(value)
            return c.replace(co_consts=tuple(values))
        code=literals(code)
        if counts!=dict.fromkeys(replacements,1):raise ValueError('Review exact original capture-label sites')
    return FunctionType(code,scope,function.__name__,function.__defaults__,function.__closure__)


def prepare_tree():return scoped(previous.prepare_tree)()
def require_integration(proof,reg,job):return scoped(previous.require_integration)(proof,reg,job)
def prepare_function(job):return scoped(previous.prepare_function)(job)
def verify_entry():return scoped(previous.verify_entry)()
def receipt(role,job):return scoped(previous.receipt)(role,job)
def archive_success(role,q):return scoped(previous.archive_success)(role,q)
def verify_queue(role):return scoped(previous.verify_queue)(role)
def submit():return scoped(previous.submit)()
def run(role):return scoped(previous.run)(role)


def bind():
    allocation=resource()
    if BINDING.exists():raise ValueError('Never overwrite finite lifecycle sources')
    bound=read(course.BINDING)
    if any(sha(path)!=digest for path,digest in bound['files'].items()):
        raise ValueError('Actual early-route source preparation changed')
    result=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_placement_early_launch_v1'],cwd=ROOT,
        capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    subprocess.run(['bash','-n',str(ENTRY)],check=True,timeout=10)
    if QUEUE.exists() or course.REGISTERED.exists() or (ROOT/course.IDENTITY['out']).exists():
        raise ValueError('Bind before any actual placement launch')
    files=dict(bound['files']);files[str(course.BINDING.resolve())]=sha(course.BINDING)
    for path in (*[ROOT/name for name in PUBLIC],Path(previous.__file__),
            Path(previous.original.__file__)):
        files[str(path.resolve())]=sha(path)
    BINDING.parent.mkdir(parents=True,exist_ok=False)
    summary=dict(status='early_route_finite_placement_lifecycle_source_bound_not_submitted',
        actual_parent_admitted=False,actual_placement_registered=False,student_updates=0,new_games=0,
        intermediate_grade_is_report_not_veto=True,actual_stage_offset=5296,
        completed_actual_parent_and_full57_data_still_required=True,
        final_confirmation_opened=False,automatic_release=False,
        allocation=allocation,test_output=result.stdout+result.stderr)
    path=BINDING.parent/'summary-private.json';write(path,summary);files[str(path.resolve())]=sha(path)
    write(BINDING,dict(files=files,actual_placement_submitted=False))
    print({k:v for k,v in summary.items() if k not in ('allocation','test_output')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('bind','submit',*ROLES,'capture'),required=True)
    a=p.parse_args()
    if a.phase=='bind':bind()
    elif a.phase=='submit':submit()
    else:run(a.phase)
