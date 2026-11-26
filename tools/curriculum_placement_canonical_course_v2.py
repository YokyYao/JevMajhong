"""Placement course and isolated integration for canonical2472 -> yaku2500.

Reuse the original register/train/audit/control code objects with explicit
canonical parent verification. Preparation never substitutes for actual parent,
complete57 data, real two-update integration or a bound finite launch.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_placement_canonical_parent_v2 as parent
from tools import curriculum_placement_balanced_course_v1 as balanced
from tools import integrate_curriculum_placement_balanced_v1 as control
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource

registration=balanced.registration
training=balanced.training
auditing=balanced.auditing
runtime=balanced.runtime
CONFIG=ROOT/'configs/training/curriculum-placement-canonical-fit-v2.json'
RUN=ROOT/'runs/curriculum-placement-canonical-parent-v2'
BINDING=ROOT/'runs/curriculum-placement-canonical-course-v2/preflight-v1/completion-binding-private.json'
INTEGRATION_BINDING=BINDING
IDENTITY=dict(run_id='jev-curriculum-placement-canonical-parent-v2',
    stage='stage4-placement-canonical-parent-v2',
    plan_root='runs/curriculum-placement-canonical-parent-v2/fit-plan',
    out='runs/curriculum-placement-canonical-parent-v2/stage4-fit')
REGISTERED=ROOT/IDENTITY['plan_root']/'registered.json'
INTEGRATION_OUT=RUN/'upstream-integration-v1'
PUBLIC=(
    'tools/curriculum_placement_canonical_course_v2.py',
    'configs/training/curriculum-placement-canonical-fit-v2.json',
    'tests/core/test_curriculum_placement_canonical_course_v2.py',
    'docs/experiments/plans/curriculum-placement-canonical-course-20261125.md',
)


def course_function(function):
    scope=dict(function.__globals__,parent=parent,CONFIG=CONFIG,BINDING=BINDING,
        IDENTITY=IDENTITY,PUBLIC=(*PUBLIC,*parent.canonical.PUBLIC),
        require_config=require_config,require_recipe=require_recipe,
        require_registration=require_registration,require_entry=require_entry,
        registration_function=registration_function,train_function=train_function,
        audit_function=audit_function)
    return FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)


def require_config(candidate):
    return course_function(balanced.require_config)(candidate)


def require_recipe(config,data,sealed):
    return course_function(balanced.require_recipe)(config,data,sealed)


def require_registration(reg):
    return course_function(balanced.require_registration)(reg)


def registration_function():
    return course_function(balanced.registration_function)()


def train_function():
    return course_function(balanced.train_function)()


def audit_function():
    return course_function(balanced.audit_function)()


def integration_function():
    function=control.integration_function
    scope=dict(function.__globals__,course=sys.modules[__name__],
        BINDING=INTEGRATION_BINDING,PUBLIC=(*PUBLIC,*parent.canonical.PUBLIC))
    return FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)()


def require_entry():
    resource();binding=read(BINDING)
    if any(sha(path)!=digest for path,digest in binding['files'].items()):
        raise ValueError('Frozen canonical placement course/control entry changed')


def register(config,seal_path):
    return course_function(balanced.register)(config,seal_path)


def train(registered):
    return course_function(balanced.train)(registered)


def audit(registered,out):
    return course_function(balanced.audit)(registered,out)


def integrate(registered=REGISTERED,out=INTEGRATION_OUT):
    require_entry()
    if Path(registered).resolve()!=REGISTERED.resolve() or Path(out).resolve()!=INTEGRATION_OUT.resolve():
        raise ValueError('Only the canonical registration and isolated control output are allowed')
    integration_function()(Path(registered).resolve(),Path(out).resolve())


def preflight():
    allocation=resource()
    if BINDING.parent.exists():raise ValueError('Never overwrite a course/control preparation')
    tests=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_placement_canonical_course_v2'],cwd=ROOT,
        capture_output=True,text=True,timeout=90)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    dependencies=[balanced.BINDING,control.BINDING,
        ROOT/'runs/curriculum-placement-canonical-parent-v2/preflight-v1/completion-binding-private.json']
    files={}
    for path in dependencies:
        binding=read(path)
        if any(sha(p)!=h for p,h in binding['files'].items()):
            raise ValueError('Earlier immutable placement preparation changed')
        files.update(binding['files']);files[str(path.resolve())]=sha(path)
    # The actual canonical parent is still absent. A fixture/test pass must not
    # manufacture admission or invoke registration, model loading or updates.
    try:parent.parent_artifacts(parent.PARENT)
    except FileNotFoundError as error:
        if Path(error.filename)!=parent.PARENT:raise
    else:raise ValueError('Unavailable actual canonical parent must be refused')
    if REGISTERED.exists() or (ROOT/IDENTITY['out']).exists() or INTEGRATION_OUT.exists():
        raise ValueError('Preparation must precede actual course/control outputs')
    BINDING.parent.mkdir(parents=True,exist_ok=False)
    summary=dict(status='canonical_placement_course_and_control_software_ready_not_actual_fit_admission',
        actual_parent_admitted=False,actual_registration_created=False,
        real_two_update_control_completed=False,student_updates=0,new_games=0,
        inherited_register_train_audit_and_integration_code_objects_unchanged=True,
        all57_sources_two_epochs_and_fresh_child_baseline_preserved=True,
        allocation=allocation,test_output=tests.stdout+tests.stderr)
    write(BINDING.parent/'summary-private.json',summary)
    public=(*registration_function().__globals__['PUBLIC_FILES'],
        *integration_function().__globals__['PUBLIC_FILES'])
    for path in [*[ROOT/name for name in public],BINDING.parent/'summary-private.json']:
        files[str(path.resolve())]=sha(path)
    write(BINDING,dict(files=files,actual_model_or_dataset_or_fit_admitted=False))
    print({k:v for k,v in summary.items() if k not in ('allocation','test_output')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('preflight','register','integration','fit','audit'),required=True)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--parent-seal',type=Path)
    parser.add_argument('--registered',type=Path,default=REGISTERED)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.phase=='preflight':preflight()
    elif args.phase=='register':
        if args.parent_seal is None:parser.error('--parent-seal is required')
        register(args.config,args.parent_seal)
    elif args.phase=='integration':integrate(args.registered,args.out or INTEGRATION_OUT)
    elif args.phase=='fit':train(args.registered)
    else:
        if args.out is None:parser.error('--out is required for audit')
        audit(args.registered,args.out)
