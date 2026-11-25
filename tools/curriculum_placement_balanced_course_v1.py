"""Registration/train/audit interfaces after an actual balanced-yaku parent.

Reuse the retained-placement function bodies and exact numerical contract.
Only the explicit parent verifier, isolated route identity and source bindings
differ. Actual integration/launch evidence remains mandatory before training.
"""
import argparse
import copy
from pathlib import Path
from types import FunctionType

from tools import curriculum_placement_balanced_parent_v1 as parent
from tools import register_curriculum_placement_parent_v1 as registration
from tools import train_curriculum_placement_parent_v1 as training
from tools import audit_curriculum_placement_parent_v1 as auditing
from tools import curriculum_placement_parent_runtime_v1 as runtime
from tools.curriculum_transfer import read, sha
from tools.capture_curriculum_paired_terminal_v2 import resource

ROOT = parent.ROOT
CONFIG = ROOT/'configs/training/curriculum-placement-balanced-fit-v1.json'
BINDING = ROOT/'runs/curriculum-placement-balanced-course-v1/preflight-v1/completion-binding-private.json'
IDENTITY = dict(run_id='jev-curriculum-placement-balanced-parent-v1',
    stage='stage4-placement-balanced-parent-v1',
    plan_root='runs/curriculum-placement-balanced-parent-v1/fit-plan',
    out='runs/curriculum-placement-balanced-parent-v1/stage4-fit')
PUBLIC = (
    'tools/curriculum_placement_balanced_course_v1.py',
    'configs/training/curriculum-placement-balanced-fit-v1.json',
    'tests/core/test_curriculum_placement_balanced_course_v1.py',
    'docs/experiments/plans/curriculum-placement-balanced-course-20261125.md',
)


def require_config(candidate):
    expected = dict(read(ROOT/'configs/training/curriculum-placement-parent-fit-v1.json'), **IDENTITY)
    if candidate != expected:
        raise ValueError('Only the alternate route identity/output may change the placement recipe')


def require_recipe(config, data, sealed):
    require_config(config)
    parent.require_sealed_lineage(sealed)
    return registration.require_recipe(config, data, sealed)


def require_registration(reg):
    runtime.require_registration(reg)
    if (reg.get('placement_parent_lineage_method') != parent.METHOD
            or reg['curriculum'].get('stage_offset') != 5968
            or any(reg['curriculum'].get(k) != value for k, value in IDENTITY.items())
            or reg['training_config'].get('model_id') != IDENTITY['run_id']):
        raise ValueError('Explicit actual balanced2472/yaku2500 placement registration required')


def registration_function():
    function = registration.registration_function()
    def writer(path, value):
        if (Path(path).name == 'registered.json'
                and value.get('status') == 'placement_data_parent_and_budget_registered_not_fit_admitted'):
            value = registration.retained_registration(value)
            value = copy.deepcopy(value)
            value['placement_parent_lineage_method'] = parent.METHOD
            require_registration(value)
        registration.original.write(path, value)
    scope = dict(function.__globals__, verify_parent=parent.verify_parent,
        require_recipe=require_recipe, write=writer,
        PUBLIC_FILES=(*function.__globals__['PUBLIC_FILES'], *parent.PUBLIC, *PUBLIC))
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)


def train_function():
    function = training.train
    scope = dict(function.__globals__, verify_parent=parent.verify_parent,
        parent_loader=parent.parent_loader, require_registration=require_registration)
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)


def audit_function():
    function = auditing.audit
    scope = dict(function.__globals__, verify_parent=parent.verify_parent,
                 require_registration=require_registration)
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)


def require_entry():
    resource()
    binding = read(BINDING)
    if any(sha(path) != digest for path, digest in binding['files'].items()):
        raise ValueError('The prospective alternate placement interface changed after preflight')


def register(config, seal_path):
    require_entry()
    require_config(read(config))
    parent.require_sealed_lineage(read(seal_path))
    registration_function()(Path(config).resolve(), Path(seal_path).resolve())


def train(registered):
    require_entry()
    train_function()(Path(registered).resolve())


def audit(registered, out):
    require_entry()
    audit_function()(Path(registered).resolve(), Path(out).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('register', 'fit', 'audit'), required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--parent-seal', type=Path)
    parser.add_argument('--registered', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.phase == 'register':
        if args.parent_seal is None: parser.error('--parent-seal is required')
        register(args.config, args.parent_seal)
    elif args.phase == 'fit':
        if args.registered is None: parser.error('--registered is required')
        train(args.registered)
    else:
        if args.registered is None or args.out is None: parser.error('--registered and --out are required')
        audit(args.registered, args.out)
