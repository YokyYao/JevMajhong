"""Admit only an actual canonical2472 -> yaku2500 parent to placement.

Preserve the original full evidence, replay, partition and 57-pool requirements.
Only explicit producer identity/path constants and lineage metadata differ.
"""
import argparse
from contextlib import contextmanager
import copy
from collections import Counter
import dis
from pathlib import Path
import subprocess
import sys
from types import CodeType,FunctionType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_placement_retained_parent_v1 as retained
from tools import seal_curriculum_placement_retained_parent_v1 as producer
from tools import curriculum_yaku_canonical_v2 as canonical
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource

METHOD='qualified_canonical2472_yaku2500_placement_parent_evidence_v2'
PARENT=canonical.RUN/'fit-plan/registered.json'
QUEUE=canonical.QUEUE
ENTRY=canonical.ENTRY
CAP='yaku-balanced-terminal-v1/terminal-evidence-private.json'
PREFIX='curriculum-yaku-canonical-v2-exit-'
PUBLIC=(
    'tools/curriculum_placement_canonical_parent_v2.py',
    'tests/core/test_curriculum_placement_canonical_parent_v2.py',
    'docs/experiments/plans/curriculum-placement-canonical-parent-20261125.md',
)
PATHS={
    'yaku-parent-terminal-v1/terminal-evidence-private.json':CAP,
    'curriculum-yaku-parent-exit-':PREFIX,
    'curriculum-progression-parent-retention-v1/yaku-parent-queue-v1-private.json':
        'curriculum-ri-balanced-replay-fit-v2/canonical-yaku-queue-v2-private.json',
}


def scoped(function,**overrides):
    counts=dict.fromkeys(PATHS,0)
    def code_paths(code):
        values=[]
        references=Counter(instruction.arg for instruction in dis.get_instructions(code)
            if instruction.opname=='LOAD_CONST')
        for position,value in enumerate(code.co_consts):
            if isinstance(value,CodeType):value=code_paths(value)
            elif isinstance(value,str) and value in PATHS:
                # Python interns repeated constants: validate reference sites,
                # preserving the original two distinct receipt path uses.
                counts[value]+=references[position];value=PATHS[value]
            values.append(value)
        return code.replace(co_consts=tuple(values))
    code=code_paths(function.__code__)
    expected={
        producer.phase_proofs:{'curriculum-yaku-parent-exit-':2},
        producer.parent_artifacts:{'yaku-parent-terminal-v1/terminal-evidence-private.json':1,
            'curriculum-yaku-parent-exit-':1},
        producer.seal:{},
        retained.verify_parent:{'yaku-parent-terminal-v1/terminal-evidence-private.json':1,
            'curriculum-yaku-parent-exit-':2,
            'curriculum-progression-parent-retention-v1/yaku-parent-queue-v1-private.json':1},
    }
    if function not in expected or {key:n for key,n in counts.items() if n}!=expected[function]:
        raise ValueError('Review changed producer code before adapting its exact paths')
    return FunctionType(code,dict(function.__globals__,**overrides),
        function.__name__,function.__defaults__,function.__closure__)


def require_lineage(parent):
    c=parent['curriculum'];init=parent['training_config']['parent_initialization']
    if (c['run_id']!='jev-curriculum-yaku-canonical-parent-v2'
            or c['stage']!='stage3B-yaku-canonical-parent-v2' or c['stage_offset']!=3468
            or c['out']!='runs/curriculum-yaku-canonical-parent-v2/stage3B-fit'
            or init['method']!=canonical.METHOD or init['selected_step']!=2472):
        raise ValueError('Actual canonical2472 -> yaku2500 ancestry required')


def require_parent_evidence(parent,done,evidence,execution,cap):
    require_lineage(parent)
    retained.require_parent_evidence(parent,done,evidence,execution,cap)


def retained_namespace():
    return SimpleNamespace(ENTRY=ENTRY,require_parent_evidence=require_parent_evidence,
        PARENT_LOSS=retained.PARENT_LOSS,CHILD_LOSS=retained.CHILD_LOSS)


def phase_function():
    return scoped(producer.phase_proofs,retained=retained_namespace())


def parent_artifacts(registration):
    return scoped(producer.parent_artifacts,PARENT=PARENT,QUEUE=QUEUE,
        retained=retained_namespace(),phase_proofs=phase_function())(registration)


def seal(registration,data_path,original_path,out):
    def writer(path,value):
        if Path(path).name=='parent-seal-private.json':
            value=copy.deepcopy(value)
            value.update(placement_parent_lineage_method=METHOD,
                canonical_parent_selected_step=2472,placement_parent_selected_step=2500)
        producer.write(path,value)
    scoped(producer.seal,parent_artifacts=parent_artifacts,retained=retained_namespace(),
        PUBLIC=(*producer.PUBLIC,*PUBLIC,*canonical.PUBLIC),write=writer)(
            Path(registration).resolve(),Path(data_path).resolve(),Path(original_path).resolve(),Path(out).resolve())


def require_sealed_lineage(sealed):
    if (sealed.get('placement_parent_lineage_method')!=METHOD
            or sealed.get('canonical_parent_selected_step')!=2472
            or sealed.get('placement_parent_selected_step')!=2500
            or Path(sealed['parent_registration']).resolve()!=PARENT.resolve()):
        raise ValueError('Explicit actual canonical2472/yaku2500 placement seal required')


def verifier_function():
    return scoped(retained.verify_parent,ENTRY=ENTRY,require_parent_evidence=require_parent_evidence)


def verify_parent(config):
    require_sealed_lineage(read(config['parent_initialization']['manifest']))
    return verifier_function()(config)


@contextmanager
def parent_loader(initialization):
    previous=initialization.verified_parent;initialization.verified_parent=verify_parent
    try:yield
    finally:initialization.verified_parent=previous


def preflight():
    allocation=resource();out=ROOT/'runs/curriculum-placement-canonical-parent-v2/preflight-v1'
    if out.exists():raise ValueError('Never overwrite canonical placement parent evidence')
    tests=subprocess.run([sys.executable,'-m','unittest',
        'tests.core.test_curriculum_placement_canonical_parent_v2'],capture_output=True,text=True,timeout=90)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    old=ROOT/'runs/curriculum-placement-balanced-parent-v1/preflight-v1/completion-binding-private.json'
    binding=read(old)
    if any(sha(p)!=h for p,h in binding['files'].items()):raise ValueError('Original parent/data evidence functions changed')
    try:parent_artifacts(PARENT)
    except FileNotFoundError as error:
        if Path(error.filename)!=PARENT:raise
    else:raise ValueError('Expected the actual unavailable future parent to be refused')
    out.mkdir(parents=True)
    summary=dict(status='canonical_placement_parent_interface_checked_not_actual_parent_admitted',
        method=METHOD,actual_missing_future_parent_refused=True,
        unchanged_full57_data_and_replay_checks=True,actual_parent_admitted=False,
        student_updates=0,new_games=0,model_loaded=False,allocation=allocation,
        test_output=tests.stdout+tests.stderr)
    write(out/'summary-private.json',summary)
    files=dict(binding['files']);files[str(old)]=sha(old)
    for p in [ROOT/name for name in (*PUBLIC,*canonical.PUBLIC)]+[out/'summary-private.json']:
        files[str(p)]=sha(p)
    write(out/'completion-binding-private.json',dict(files=files))
    print({k:v for k,v in summary.items() if k!='allocation'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('preflight','seal'),required=True)
    parser.add_argument('--parent-registration',type=Path,default=PARENT)
    parser.add_argument('--data-ready',type=Path)
    parser.add_argument('--original-data-ready',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.phase=='preflight':preflight()
    else:
        if any(v is None for v in (args.data_ready,args.original_data_ready,args.out)):
            parser.error('--data-ready, --original-data-ready and --out are required for seal')
        seal(args.parent_registration,args.data_ready,args.original_data_ready,args.out)
