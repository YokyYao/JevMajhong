"""Admit a future qualified balanced2472 -> yaku2500 parent to placement.

Only producer identities and paths differ from the retained-parent seal. The
original qualification, replay, loss, partition and model checks are reused.
This entry neither registers a placement fit nor launches optimizer updates.
"""
import argparse
import ast
from contextlib import contextmanager
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

from tools import curriculum_placement_retained_parent_v1 as retained
from tools import seal_curriculum_placement_retained_parent_v1 as producer
from tools.curriculum_transfer import read
from tools.curriculum_yaku_balanced_parent_v1 import METHOD as INITIALIZER

ROOT = producer.ROOT
METHOD = 'qualified_balanced2472_yaku2500_placement_parent_evidence_v1'
PARENT = ROOT/'runs/curriculum-yaku-balanced-parent-v1/fit-plan/registered.json'
QUEUE = ROOT/'runs/curriculum-ri-balanced-replay-fit-v1/balanced-yaku-queue-v1-private.json'
ENTRY = ROOT/'cluster/curriculum-yaku-balanced-v1.slurm'
CAP = 'yaku-balanced-terminal-v1/terminal-evidence-private.json'
PREFIX = 'curriculum-yaku-balanced-exit-'
PUBLIC = (
    'tools/curriculum_placement_balanced_parent_v1.py',
    'tests/core/test_curriculum_placement_balanced_parent_v1.py',
    'docs/experiments/plans/curriculum-placement-balanced-parent-20261125.md',
)


def adapted_tree(function, replacements):
    """Replace exact declared path constants, refusing missing or extra edits."""
    tree = ast.parse(inspect.getsource(function))
    counts = dict.fromkeys(replacements, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replacements:
            old = node.value
            node.value = replacements[old][0]
            counts[old] += 1
    if counts != {old: spec[1] for old, spec in replacements.items()}:
        raise ValueError('Original evidence paths changed; review the explicit adapter')
    return ast.fix_missing_locations(tree)


def specialize(function, replacements, namespace):
    scope = dict(function.__globals__, **namespace)
    exec(compile(adapted_tree(function, replacements), inspect.getsourcefile(function), 'exec'), scope)
    return scope[function.__name__]


def require_lineage(parent):
    c = parent['curriculum']
    init = parent['training_config']['parent_initialization']
    if (c['run_id'] != 'jev-curriculum-yaku-balanced-parent-v1'
            or c['stage'] != 'stage3B-yaku-balanced-parent-v1' or c['stage_offset'] != 3468
            or init['method'] != INITIALIZER or init['selected_step'] != 2472
            or Path(c['out']).as_posix() != 'runs/curriculum-yaku-balanced-parent-v1/stage3B-fit'):
        raise ValueError('Actual balanced2472 -> yaku2500 lineage is required')


def require_parent_evidence(parent, done, evidence, execution, cap):
    require_lineage(parent)
    # The original strategy grade and complete numerical evidence are unchanged.
    retained.require_parent_evidence(parent, done, evidence, execution, cap)


def retained_namespace():
    return SimpleNamespace(ENTRY=ENTRY, require_parent_evidence=require_parent_evidence,
                           PARENT_LOSS=retained.PARENT_LOSS, CHILD_LOSS=retained.CHILD_LOSS)


def phase_function():
    return specialize(producer.phase_proofs,
        {'curriculum-yaku-parent-exit-': (PREFIX, 2)},
        dict(retained=retained_namespace()))


def parent_artifacts(registration):
    return specialize(producer.parent_artifacts, {
        'yaku-parent-terminal-v1/terminal-evidence-private.json': (CAP, 1),
        'curriculum-yaku-parent-exit-': (PREFIX, 1),
    }, dict(PARENT=PARENT, QUEUE=QUEUE, retained=retained_namespace(),
            phase_proofs=phase_function()))(registration)


def seal(registration, data_path, original_path, out):
    # The original producer checks the actual GPU allocation and requires all57
    # sources, full original replay, exact final2500 and four successful phases.
    def writer(path, value):
        if Path(path).name == 'parent-seal-private.json':
            value = copy.deepcopy(value)
            value['placement_parent_lineage_method'] = METHOD
            value['balanced_parent_selected_step'] = 2472
            value['placement_parent_selected_step'] = 2500
        producer.write(path, value)
    namespace = dict(parent_artifacts=parent_artifacts, retained=retained_namespace(),
                     PUBLIC=(*producer.PUBLIC, *PUBLIC), write=writer)
    specialize(producer.seal, {}, namespace)(
        Path(registration).resolve(), Path(data_path).resolve(),
        Path(original_path).resolve(), Path(out).resolve())


def require_sealed_lineage(sealed):
    if (sealed.get('placement_parent_lineage_method') != METHOD
            or sealed.get('balanced_parent_selected_step') != 2472
            or sealed.get('placement_parent_selected_step') != 2500
            or Path(sealed['parent_registration']).resolve() != PARENT.resolve()):
        raise ValueError('Explicit actual balanced2472 -> yaku2500 placement seal required')


def verifier_function():
    return specialize(retained.verify_parent, {
        'yaku-parent-terminal-v1/terminal-evidence-private.json': (CAP, 1),
        'curriculum-progression-parent-retention-v1/yaku-parent-queue-v1-private.json':
            ('curriculum-ri-balanced-replay-fit-v1/balanced-yaku-queue-v1-private.json', 1),
        'curriculum-yaku-parent-exit-': (PREFIX, 2),
    }, dict(ENTRY=ENTRY, require_parent_evidence=require_parent_evidence))


def verify_parent(config):
    sealed = read(config['parent_initialization']['manifest'])
    require_sealed_lineage(sealed)
    return verifier_function()(config)


@contextmanager
def parent_loader(initialization):
    original = initialization.verified_parent
    initialization.verified_parent = verify_parent
    try:
        yield
    finally:
        initialization.verified_parent = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-registration', type=Path, default=PARENT)
    parser.add_argument('--data-ready', type=Path, required=True)
    parser.add_argument('--original-data-ready', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    seal(args.parent_registration, args.data_ready, args.original_data_ready, args.out)
