"""Unchanged placement learning recipe after the actual early-route3B parent.

The ancestry offset is 608+388+1800+2500=5296. The intermediate grade remains
an honest report; source/data/tensor/optimizer integrity is still required.
"""
import argparse
import copy
from pathlib import Path
import sys
from types import CodeType,FunctionType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_placement_early_parent_v1 as parent
from tools import curriculum_placement_balanced_course_v1 as balanced
from tools import integrate_curriculum_placement_balanced_v1 as control
from tools.curriculum_transfer import read,sha
from tools.capture_curriculum_paired_terminal_v2 import resource

registration=balanced.registration
training=balanced.training
auditing=balanced.auditing
runtime=balanced.runtime
CONFIG=ROOT/'configs/training/curriculum-placement-early-fit-v1.json'
RUN=ROOT/'runs/curriculum-placement-early-parent-v1'
BINDING=ROOT/'runs/curriculum-placement-early-course-v1/preflight-v1/completion-binding-private.json'
INTEGRATION_BINDING=BINDING
OFFSET=5296
IDENTITY=dict(run_id='jev-curriculum-placement-early-parent-v1',
    stage='stage4-placement-early-parent-v1',
    plan_root='runs/curriculum-placement-early-parent-v1/fit-plan',
    out='runs/curriculum-placement-early-parent-v1/stage4-fit')
REGISTERED=ROOT/IDENTITY['plan_root']/'registered.json'
INTEGRATION_OUT=RUN/'upstream-integration-v1'
PUBLIC=('tools/curriculum_placement_early_course_v1.py',
    'configs/training/curriculum-placement-early-fit-v1.json',
    'tests/core/test_curriculum_placement_early_course_v1.py',
    'docs/experiments/plans/curriculum-placement-early-parent-20261125.md')


def replace_offset(function,**overrides):
    """Change exactly the one historical ancestry literal, never the budget."""
    sites=0
    def rewrite(code):
        nonlocal sites
        values=[]
        for value in code.co_consts:
            if isinstance(value,CodeType):value=rewrite(value)
            elif type(value) is int and value==5968:sites+=1;value=OFFSET
            values.append(value)
        return code.replace(co_consts=tuple(values))
    code=rewrite(function.__code__)
    if sites!=1:raise ValueError('Review the exact original ancestry-offset site')
    return FunctionType(code,dict(function.__globals__,**overrides),function.__name__,
        function.__defaults__,function.__closure__)


def require_config(candidate):
    expected=dict(read(ROOT/'configs/training/curriculum-placement-parent-fit-v1.json'),**IDENTITY)
    if candidate!=expected:raise ValueError('Only the explicit early-route identity/output may change')


def require_recipe(config,data,sealed):
    require_config(config);parent.require_sealed_lineage(sealed)
    original=registration.original
    recipe=parent.report_grade_only(original.require_recipe,
        PARENT_METHOD=parent.PARENT_METHOD,STATUS=parent.STATUS)
    function=registration.require_recipe
    adapted=FunctionType(function.__code__,dict(function.__globals__,
        original=SimpleNamespace(LOSS=original.LOSS,require_recipe=recipe)),
        function.__name__,function.__defaults__,function.__closure__)
    return adapted(config,data,sealed)


def require_registration(reg):
    return replace_offset(balanced.require_registration,parent=parent,IDENTITY=IDENTITY)(reg)


def registration_function():
    function=registration.registration_function()
    def writer(path,value):
        if (Path(path).name=='registered.json'
                and value.get('status')=='placement_data_parent_and_budget_registered_not_fit_admitted'):
            value=copy.deepcopy(registration.retained_registration(value))
            value['placement_parent_lineage_method']=parent.METHOD
            require_registration(value)
        registration.original.write(path,value)
    return FunctionType(function.__code__,dict(function.__globals__,
        PARENT_METHOD=parent.PARENT_METHOD,STATUS=parent.STATUS,verify_parent=parent.verify_parent,
        require_recipe=require_recipe,write=writer,
        PUBLIC_FILES=(*function.__globals__['PUBLIC_FILES'],*parent.PUBLIC,*PUBLIC)),
        function.__name__,function.__defaults__,function.__closure__)


def train_function():
    f=training.train
    return FunctionType(f.__code__,dict(f.__globals__,verify_parent=parent.verify_parent,
        parent_loader=parent.parent_loader,require_registration=require_registration),
        f.__name__,f.__defaults__,f.__closure__)


def audit_function():
    f=auditing.audit
    return FunctionType(f.__code__,dict(f.__globals__,verify_parent=parent.verify_parent,
        require_registration=require_registration),f.__name__,f.__defaults__,f.__closure__)


def integration_function():
    return replace_offset(control.integration_function,course=sys.modules[__name__],
        BINDING=INTEGRATION_BINDING,PUBLIC=(*PUBLIC,*parent.PUBLIC))()


def require_entry():
    resource();binding=read(BINDING)
    if any(sha(path)!=digest for path,digest in binding['files'].items()):
        raise ValueError('Bound early-route placement source changed')


def register(config,seal_path):
    require_entry();require_config(read(config));parent.require_sealed_lineage(read(seal_path))
    registration_function()(Path(config).resolve(),Path(seal_path).resolve())


def train(registered):
    require_entry();train_function()(Path(registered).resolve())


def audit(registered,out):
    require_entry();audit_function()(Path(registered).resolve(),Path(out).resolve())


def integrate(registered=REGISTERED,out=INTEGRATION_OUT):
    require_entry()
    if Path(registered).resolve()!=REGISTERED.resolve() or Path(out).resolve()!=INTEGRATION_OUT.resolve():
        raise ValueError('Only the current early-route registration and isolated control are allowed')
    integration_function()(Path(registered).resolve(),Path(out).resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('register','integration','fit','audit'),required=True)
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--parent-seal',type=Path)
    p.add_argument('--registered',type=Path,default=REGISTERED);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.phase=='register':
        if a.parent_seal is None:p.error('--parent-seal is required')
        register(a.config,a.parent_seal)
    elif a.phase=='integration':integrate(a.registered,a.out or INTEGRATION_OUT)
    elif a.phase=='fit':train(a.registered)
    else:
        if a.out is None:p.error('--out is required')
        audit(a.registered,a.out)
