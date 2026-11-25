"""Consume an actually sealed canonical placement candidate, never create one.

The old final verifier imports the pre-retained-loss placement guard. Change
that one import to the retained-loss runtime, add canonical lineage/source
guards, and retain every original source, native and statistical requirement.
"""
import argparse
from collections import Counter
import dis
from pathlib import Path
import subprocess
import sys
from types import CodeType,FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_placement_canonical_course_v2 as course
from tools import curriculum_confirmation_runtime_v1 as original
from tools import run_curriculum_confirmation_v1 as runner
from tools import audit_curriculum_confirmation_v1 as auditor
from tools import finish_curriculum_confirmation_audit_v1 as finisher
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource

REGISTERED=ROOT/'runs/curriculum-final-confirmation-canonical-v2/registered.json'
RUN_ENTRY=ROOT/'cluster/curriculum-confirmation-canonical-run-v2.slurm'
AUDIT_ENTRY=ROOT/'cluster/curriculum-confirmation-canonical-audit-v2.slurm'
PUBLIC=(
    'tools/curriculum_confirmation_canonical_v2.py',
    'cluster/curriculum-confirmation-canonical-run-v2.slurm',
    'cluster/curriculum-confirmation-canonical-audit-v2.slurm',
    'tests/core/test_curriculum_confirmation_canonical_v2.py',
    'docs/experiments/plans/curriculum-confirmation-canonical-consumer-20261125.md',
)


def remap(function,names=None,literals=None):
    names=names or {};literals=literals or {};counts=Counter();imports=Counter()
    def rewrite(code):
        for instruction in dis.get_instructions(code):
            if instruction.opname=='IMPORT_NAME' and instruction.argval in names:
                imports[instruction.argval]+=1
            elif instruction.opname=='LOAD_CONST' and isinstance(instruction.argval,str) and instruction.argval in literals:
                counts[instruction.argval]+=1
        return code.replace(co_names=tuple(names.get(value,value) for value in code.co_names),
            co_consts=tuple(rewrite(value) if isinstance(value,CodeType)
                else literals.get(value,value) if isinstance(value,str) else value for value in code.co_consts))
    code=rewrite(function.__code__)
    if dict(imports)!={key:1 for key in names} or dict(counts)!={key:1 for key in literals}:
        raise ValueError('Review original final consumer import/literal reference sites')
    return code


def verify_function():
    code=remap(original.verify,names={
        'tools.curriculum_placement_runtime_v1':'tools.curriculum_placement_parent_runtime_v1'})
    return FunctionType(code,dict(original.verify.__globals__),original.verify.__name__,
        original.verify.__defaults__,original.verify.__closure__)


def require_route(placement_registration,placement):
    if Path(placement_registration).resolve()!=course.REGISTERED.resolve():
        raise ValueError('Only the actual canonical placement registration may enter this final consumer')
    course.require_registration(placement)


def verify(reg,registered):
    original.require_header(reg)
    if Path(registered).resolve()!=REGISTERED.resolve():
        raise ValueError('Only the isolated canonical final seal is allowed')
    require_route(reg['placement_registration'],read(reg['placement_registration']))
    # Bind the exact consumer and finite execution paths before any reserved
    # wall reset or model load. The original verifier hashes this full union.
    for name in (*PUBLIC,*course.PUBLIC,*course.parent.PUBLIC):
        path=(ROOT/name).resolve()
        if reg['files'].get(str(path))!=sha(path):
            raise ValueError('Canonical final source or execution entry unbound: '+name)
    return verify_function()(reg,registered)


def run_function():
    function=runner.run
    return FunctionType(function.__code__,dict(function.__globals__,verify=verify),
        function.__name__,function.__defaults__,function.__closure__)


def audit_function():
    function=auditor.audit
    return FunctionType(function.__code__,dict(function.__globals__,verify=verify),
        function.__name__,function.__defaults__,function.__closure__)


def finish_function():
    code=remap(finisher.finish,names={
        'tools.audit_curriculum_confirmation_v1':'tools.curriculum_confirmation_canonical_v2'},
        literals={'curriculum-confirmation-run-exit-':'curriculum-confirmation-canonical-run-exit-',
            'cluster/curriculum-confirmation-run-v1.slurm':'cluster/curriculum-confirmation-canonical-run-v2.slurm'})
    return FunctionType(code,dict(finisher.finish.__globals__),finisher.finish.__name__,
        finisher.finish.__defaults__,finisher.finish.__closure__)


def run(registered=REGISTERED):return run_function()(Path(registered).resolve())
def audit(registered,out):return audit_function()(Path(registered).resolve(),Path(out).resolve())
def finish(registered,run_job):return finish_function()(Path(registered).resolve(),run_job)


def preflight():
    allocation=resource();out=ROOT/'runs/curriculum-confirmation-canonical-consumer-v2/preflight-v1'
    if out.exists():raise ValueError('Never overwrite final consumer preparation')
    if REGISTERED.exists():raise ValueError('Prepare before an actual final seal exists')
    result=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_confirmation_canonical_v2'],cwd=ROOT,
        capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    for entry in (RUN_ENTRY,AUDIT_ENTRY):subprocess.run(['bash','-n',str(entry)],check=True,timeout=10)
    try:verify({},REGISTERED)
    except ValueError:pass
    else:raise ValueError('An empty/unsealed plan cannot admit reserved execution')
    out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='canonical_final_consumer_software_checked_not_sealed_or_opened',
        actual_final_registration_created=False,actual_candidate_admitted=False,
        full_source_producer_and_ancestry_closure_still_required=True,
        original_final_criteria_source_native_and_statistical_checks_preserved=True,
        reserved_games_generated=0,model_loaded=False,student_updates=0,
        allocation=allocation,test_output=result.stdout+result.stderr)
    write(out/'summary-private.json',summary)
    paths=[*[ROOT/name for name in (*PUBLIC,*course.PUBLIC,*course.parent.PUBLIC)],
        Path(original.__file__),Path(runner.__file__),Path(auditor.__file__),Path(finisher.__file__),
        ROOT/'tools/curriculum_placement_parent_runtime_v1.py',out/'summary-private.json']
    write(out/'completion-binding-private.json',dict(files={str(p.resolve()):sha(p) for p in paths}))
    print({k:v for k,v in summary.items() if k not in ('allocation','test_output')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('preflight','run','finish'),required=True)
    parser.add_argument('--registered',type=Path,default=REGISTERED)
    parser.add_argument('--run-job',type=int)
    args=parser.parse_args()
    if args.phase=='preflight':preflight()
    elif args.phase=='run':run(args.registered)
    else:
        if args.run_job is None:parser.error('--run-job is required')
        finish(args.registered,args.run_job)
