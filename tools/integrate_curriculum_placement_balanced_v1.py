"""Prospective real two-update integration for retained-loss balanced placement.

The inherited control is isolated, TRAIN-only and never a candidate. This
module does not manufacture its unavailable parent, data or optimizer proof.
"""
import argparse
import copy
from pathlib import Path
from types import FunctionType

from tools import preflight_curriculum_placement_v1 as original
from tools import curriculum_placement_balanced_course_v1 as course
from tools import curriculum_placement_parent_loss_v2 as loss
from tools.curriculum_transfer import read, sha
from tools.capture_curriculum_paired_terminal_v2 import resource

ROOT = course.ROOT
ENTRY = ROOT/'cluster/curriculum-placement-balanced-integration-v1.slurm'
BINDING = ROOT/'runs/curriculum-placement-balanced-integration-v1/entry-preflight-v1/completion-binding-private.json'
REGISTERED = ROOT/course.IDENTITY['plan_root']/'registered.json'
OUT = ROOT/Path(course.IDENTITY['out']).parent/'upstream-integration-v1'
PUBLIC = (
    'tools/integrate_curriculum_placement_balanced_v1.py',
    'cluster/curriculum-placement-balanced-integration-v1.slurm',
    'tests/core/test_curriculum_placement_balanced_integration_v1.py',
    'docs/experiments/plans/curriculum-placement-balanced-integration-20261125.md',
)


def integration_function():
    def writer(path, value):
        value = copy.deepcopy(value)
        if Path(path).name == 'integration-proof-private.json':
            value['loss_method'] = loss.METHOD
            value['placement_parent_lineage_method'] = course.parent.METHOD
            value['global_student_stage_offset'] = 5968
        elif Path(path).name == 'completion-binding-private.json':
            value['files'][str(BINDING.resolve())] = sha(BINDING)
        original.write(path, value)
    function = original.preflight
    scope = dict(function.__globals__, require_registration=course.require_registration,
        verify_parent=course.parent.verify_parent, annotate=loss.annotate,
        weighted_loss=loss.weighted_loss, summarize=loss.summarize, write=writer,
        PUBLIC_FILES=(*original.PUBLIC_FILES, *course.parent.PUBLIC, *course.PUBLIC, *PUBLIC))
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)


def require_paths(registered, out):
    if Path(registered).resolve() != REGISTERED.resolve() or Path(out).resolve() != OUT.resolve():
        raise ValueError('The exact alternate registration and isolated integration output are required')


def integrate(registered=REGISTERED, out=OUT):
    resource()
    require_paths(registered, out)
    binding = read(BINDING)
    if any(sha(path) != digest for path, digest in binding['files'].items()):
        raise ValueError('Before-parent integration entry changed')
    course.require_entry()
    integration_function()(Path(registered).resolve(), Path(out).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered', type=Path, default=REGISTERED)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    integrate(args.registered, args.out)
