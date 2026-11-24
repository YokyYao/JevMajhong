"""Register retained-parent placement without changing the original budget."""
import argparse
import copy
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import register_curriculum_placement_v1 as original
from tools import curriculum_placement_parent_runtime_v1 as runtime
from tools.curriculum_placement_retained_parent_v1 import verify_parent
from tools.curriculum_transfer import read, sha
from tools.capture_curriculum_paired_terminal_v2 import resource

BINDING = ROOT/'runs/curriculum-placement-parent-registration-v1/entry-preflight-v1/completion-binding-private.json'
PUBLIC = ('tools/register_curriculum_placement_parent_v1.py',
    'tools/preflight_curriculum_placement_parent_registration_v1.py',
    'tests/core/test_curriculum_placement_parent_registration_v1.py',
    'configs/training/curriculum-placement-parent-fit-v1.json',
    'docs/experiments/plans/curriculum-placement-parent-registration-20261124.md')


def require_recipe(config, data, sealed):
    if (config['loss_method'] != runtime.LOSS or config.get('parent_loss_method') != runtime.PARENT_LOSS
            or config.get('retained_parent_replay_objective_explicit') is not True
            or config.get('upstream_normalized_terminal_alignment_required') is not True
            or sealed.get('parent_loss_method') != runtime.PARENT_LOSS
            or sealed.get('child_loss_method') != runtime.LOSS
            or sealed.get('exact_retained_3B_replay_objective') is not True):
        raise ValueError('Explicit retained-parent and normalized-placement recipe/seal required')
    derived = copy.deepcopy(config); derived['loss_method'] = original.LOSS
    return original.require_recipe(derived, data, sealed)


def retained_registration(result):
    if (result['curriculum']['loss_method'] != runtime.LOSS
            or result['training_config']['loss_method'] != runtime.LOSS):
        raise ValueError('Both actual training configuration and curriculum must retain the new loss')
    updated = copy.deepcopy(result)
    updated.update(retained_parent_replay_objective_explicit=True,
        upstream_normalized_terminal_alignment_required=True, parent_loss_method=runtime.PARENT_LOSS)
    runtime.require_registration(updated)
    return updated


def registration_function():
    def writer(path, value):
        if (Path(path).name == 'registered.json'
                and value.get('status') == 'placement_data_parent_and_budget_registered_not_fit_admitted'):
            value = retained_registration(value)
        original.write(path, value)
    scope = dict(vars(original), LOSS=runtime.LOSS, verify_parent=verify_parent,
        require_recipe=require_recipe, PUBLIC_FILES=(*original.PUBLIC_FILES, *PUBLIC), write=writer)
    function = original.register
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)


def register(config, seal_path):
    resource(); binding = read(BINDING)
    if any(sha(p) != h for p, h in binding['files'].items()):
        raise ValueError('The before-completion registration entry changed')
    registration_function()(config.resolve(), seal_path.resolve())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--parent-seal', type=Path, required=True); a = p.parse_args()
    register(a.config, a.parent_seal)
