"""Retained-parent placement runtime with the unchanged numeric/policy contract."""
import copy

from tools import curriculum_placement_runtime_v1 as original
from tools.curriculum_placement_parent_loss_v2 import METHOD as LOSS
from tools.curriculum_yaku_parent_loss_v1 import METHOD as PARENT_LOSS

SCHEMA = original.SCHEMA
ROUTE = original.ROUTE
PlacementEpochRandom = original.PlacementEpochRandom
random_proxy = original.random_proxy
prediction_parity = original.prediction_parity
require_step0 = original.require_step0
policy_contract = original.policy_contract


def legacy_numeric_view(reg):
    if (reg['curriculum']['loss_method'] != LOSS or reg['training_config']['loss_method'] != LOSS
            or reg.get('retained_parent_replay_objective_explicit') is not True
            or reg.get('upstream_normalized_terminal_alignment_required') is not True
            or reg.get('parent_loss_method') != PARENT_LOSS):
        raise ValueError('Explicit retained3B parent and normalized-alignment placement loss required')
    derived = copy.deepcopy(reg)
    derived['curriculum']['loss_method'] = original.LOSS
    return derived


def require_registration(reg):
    original.require_registration(legacy_numeric_view(reg))


def evaluation_groups(reg, step):
    return original.evaluation_groups(legacy_numeric_view(reg), step)


def matched_curve(games, reg, step, aggregate):
    return original.matched_curve(games, legacy_numeric_view(reg), step, aggregate)
