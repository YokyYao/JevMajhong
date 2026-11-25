"""Finite canonical placement lifecycle; no parent/data admission is invented."""
import argparse
from collections import Counter
import dis
from pathlib import Path
import subprocess
import sys
from types import CodeType,FunctionType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_placement_canonical_course_v2 as course
from tools import launch_curriculum_placement_balanced_v1 as previous
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler

CONTROL=ROOT/'runs/curriculum-placement-canonical-launch-v2'
RUN=course.RUN
ENTRY=ROOT/'cluster/curriculum-placement-canonical-course-v2.slurm'
BINDING=CONTROL/'entry-preflight-v1/completion-binding-private.json'
QUEUE=CONTROL/'queue-private.json'
PARENT_TERMINAL_QUEUE=CONTROL/'parent-terminal-queue-private.json'
LAUNCH_CONFIG=RUN/'launch-config-private.json'
DATA=previous.DATA
ORIGINAL_DATA=previous.ORIGINAL_DATA
SEAL=RUN/'qualified-yaku-parent-v1/parent-seal-private.json'
PREFIX='curriculum-placement-canonical-course-exit-'
ROLES=previous.ROLES
LIMITS=previous.LIMITS
PUBLIC=(
    'tools/launch_curriculum_placement_canonical_v2.py',
    'cluster/curriculum-placement-canonical-course-v2.slurm',
    'tests/core/test_curriculum_placement_canonical_launch_v2.py',
    'docs/experiments/plans/curriculum-placement-canonical-launch-20261125.md',
)
integration=SimpleNamespace(loss=course.control.loss,BINDING=course.INTEGRATION_BINDING,
    REGISTERED=course.REGISTERED,OUT=course.INTEGRATION_OUT,PUBLIC=course.PUBLIC,
    integration_function=course.integration_function,integrate=course.integrate)
REPLACEMENTS={
    'curriculum-yaku-balanced-exit-':'curriculum-yaku-canonical-v2-exit-',
    'placement-balanced-terminal-v1':'placement-canonical-terminal-v2',
    'actual_balanced_placement_terminals_preserved_not_confirmed':
        'actual_canonical_placement_terminals_preserved_not_confirmed',
}


def scoped(function):
    counts=Counter()
    def literals(code):
        references=Counter(instruction.arg for instruction in dis.get_instructions(code)
            if instruction.opname=='LOAD_CONST')
        values=[]
        for index,value in enumerate(code.co_consts):
            if isinstance(value,CodeType):value=literals(value)
            elif isinstance(value,str) and value in REPLACEMENTS:
                counts[value]+=references[index];value=REPLACEMENTS[value]
            values.append(value)
        return code.replace(co_consts=tuple(values)) if counts else code
    expected={previous.preserve_parent_terminal:{'curriculum-yaku-balanced-exit-':1},
        previous.run:{'placement-balanced-terminal-v1':1,
            'actual_balanced_placement_terminals_preserved_not_confirmed':1}}
    code=literals(function.__code__)
    if dict(counts)!=expected.get(function,{}):
        raise ValueError('Review changed original lifecycle literal sites')
    names=('course','integration','CONTROL','RUN','ENTRY','BINDING','QUEUE',
        'PARENT_TERMINAL_QUEUE','LAUNCH_CONFIG','DATA','ORIGINAL_DATA','SEAL','PREFIX',
        'ROLES','LIMITS','PUBLIC','prepare_tree','require_integration','prepare_function',
        'verify_entry','receipt','archive_success','verify_queue','preserve_parent_terminal',
        'enqueue_parent_terminal','submit','run','read','sha','resource','scheduler','subprocess','write')
    scope=dict(function.__globals__,**{name:globals()[name] for name in names})
    return FunctionType(code,scope,function.__name__,function.__defaults__,function.__closure__)


def prepare_tree():return scoped(previous.prepare_tree)()
def require_integration(proof,reg,job):return scoped(previous.require_integration)(proof,reg,job)
def prepare_function(job):return scoped(previous.prepare_function)(job)
def verify_entry():return scoped(previous.verify_entry)()
def receipt(role,job):return scoped(previous.receipt)(role,job)
def archive_success(role,q):return scoped(previous.archive_success)(role,q)
def verify_queue(role):return scoped(previous.verify_queue)(role)
def preserve_parent_terminal():return scoped(previous.preserve_parent_terminal)()
def enqueue_parent_terminal():return scoped(previous.enqueue_parent_terminal)()
def submit():return scoped(previous.submit)()
def run(role):return scoped(previous.run)(role)


def bind():
    allocation=resource()
    if BINDING.parent.exists():raise ValueError('Never overwrite a lifecycle binding')
    dependencies=(previous.BINDING,course.BINDING)
    files={}
    for path in dependencies:
        binding=read(path)
        if any(sha(p)!=h for p,h in binding['files'].items()):
            raise ValueError('Earlier immutable lifecycle/course preparation changed')
        files.update(binding['files']);files[str(path.resolve())]=sha(path)
    result=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_placement_canonical_launch_v2'],cwd=ROOT,
        capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    subprocess.run(['bash','-n',str(ENTRY)],check=True,timeout=10)
    try:course.parent.parent_artifacts(course.parent.PARENT)
    except FileNotFoundError as error:
        if Path(error.filename)!=course.parent.PARENT:raise
    else:raise ValueError('Unavailable actual canonical parent must still be refused')
    if QUEUE.exists() or course.REGISTERED.exists() or (ROOT/course.IDENTITY['out']).exists():
        raise ValueError('Bind before any placement course is submitted')
    BINDING.parent.mkdir(parents=True,exist_ok=False)
    summary=dict(status='canonical_finite_placement_lifecycle_bound_not_submitted',
        actual_parent_admitted=False,actual_registration_or_fit_started=False,
        full_data_and_parent_still_required=True,student_updates=0,new_games=0,
        allocation=allocation,test_output=result.stdout+result.stderr)
    write(BINDING.parent/'summary-private.json',summary)
    for path in [*[ROOT/name for name in PUBLIC],Path(previous.__file__),
            BINDING.parent/'summary-private.json']:
        files[str(path.resolve())]=sha(path)
    write(BINDING,dict(files=files,allocation=allocation,student_updates=0,
        actual_registration_or_fit_started=False,full_data_and_parent_still_required=True))
    print({k:v for k,v in summary.items() if k not in ('allocation','test_output')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('bind','submit','enqueue-parent-terminal','parent-terminal',*ROLES,'capture'),required=True)
    args=parser.parse_args()
    if args.phase=='bind':bind()
    elif args.phase=='submit':submit()
    elif args.phase=='enqueue-parent-terminal':enqueue_parent_terminal()
    elif args.phase=='parent-terminal':preserve_parent_terminal()
    else:run(args.phase)
