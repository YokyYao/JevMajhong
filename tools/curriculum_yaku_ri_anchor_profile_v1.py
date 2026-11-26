"""Unchanged canonical25 losses with explicit augmented validation cohorts."""
import dis
import math
from pathlib import Path
from types import FunctionType

from tools import curriculum_yaku_parent_fit_v1 as original
from tools import curriculum_ri_anchor_runtime_v1 as inherited
from tools.curriculum_transfer import read
from tools.curriculum_parent import require_replay
from tools.curriculum_complete_legal_class_policy_v1 import select

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'runs/curriculum-yaku-augmented-data-v1/data-v1'


def augmented_function():
    function = original.profile
    wanted = {24516: 25826, 23688: 24998}
    counts = {old: sum(instruction.opname == 'LOAD_CONST' and instruction.argval == old
        for instruction in dis.get_instructions(function)) for old in wanted}
    if counts != {24516: 1, 23688: 1}:
        raise ValueError('Review original25-source cardinality guards before augmentation')
    code = function.__code__.replace(co_consts=tuple(
        wanted[value] if type(value) is int and value in wanted else value
        for value in function.__code__.co_consts))
    return FunctionType(code, dict(function.__globals__, replay_profile=inherited.profile),
        function.__name__, function.__defaults__, function.__closure__)


def partition(records, predictions, cohorts, expected):
    if set(cohorts) != set(expected):
        raise ValueError('Every declared disjoint validation cohort required')
    groups = {name: set(ids) for name, ids in cohorts.items()}
    if any(len(cohorts[name]) != len(groups[name]) or len(groups[name]) != expected[name] for name in groups):
        raise ValueError('Exact unique cohort counts required')
    seen = set()
    for ids in groups.values():
        if seen & ids:
            raise ValueError('Validation cohort identities must be disjoint')
        seen.update(ids)
    rows = {row['_meta']['id']: row for row in records}; saved = {row['id']: row for row in predictions}
    if (len(rows) != len(records) or len(saved) != len(predictions)
            or set(rows) != seen or set(saved) != seen):
        raise ValueError('Complete distinct full validation records and predictions required')
    return groups, rows, saved


def profile(records, predictions, base, ce):
    cohorts = read(DATA/'cohort-ids-private.json')['validation']
    partition(records, predictions, cohorts,
        dict(original19=23688, basic_supplement=1310, yaku_routes=828))
    result = augmented_function()(records, predictions, base, ce)
    replay = result['replay_profile']
    result.update(original_validation=replay['original_validation'],
        new_teacher_validation=replay['new_teacher_validation'],
        retention_quality_cohort='original23688', canonical_validation_has_TRAIN_anchor=False,
        full_parent_replay_records=24998, full_validation_records=25826,
        original_and_new_basic_and_route_cohorts_reported=True)
    return result


def prediction_parity(parent_rows, parent_predictions, child_rows, child_predictions):
    if len(parent_rows) != 24998 or len(child_rows) != 25826:
        raise ValueError('Full augmented parent and complete25-source child required')
    require_replay(parent_rows, child_rows)
    old = {row['id']: row for row in parent_predictions}; current = {row['id']: row for row in child_predictions}
    if (len(old) != len(parent_predictions) or len(current) != len(child_predictions)
            or set(old) != {row['_meta']['id'] for row in parent_rows}
            or set(current) != {row['_meta']['id'] for row in child_rows}):
        raise ValueError('Every distinct actual parent and child prediction required')
    error = 0.
    for row in parent_rows:
        identifier = row['_meta']['id']; before = old[identifier]; after = current[identifier]
        if before['keys'] != after['keys'] or len(before['p']) != len(before['keys']) or len(after['p']) != len(after['keys']):
            raise ValueError('Complete parent menu order and probability dimensions must match')
        if set(before['keys']) != set(row['questions']['action']['criteria']):
            raise ValueError('Actual native menu identity differs')
        for values in (before['p'], after['p']):
            if any(not math.isfinite(value) or value < 0 for value in values) or abs(sum(values)-1) > 1e-5:
                raise ValueError('Finite normalized full parent distributions required')
        error = max(error, max(abs(a-b) for a, b in zip(before['p'], after['p'])))
        kinds = [row['questions']['action']['criteria'][key]['type'] for key in before['keys']]
        if select(kinds, before['p']) != select(kinds, after['p']):
            raise ValueError('Inherited complete-legal fixed-RI choice changed')
    if not math.isfinite(error) or error >= 5e-5:
        raise ValueError('Every inherited parent prediction must retain serving parity')
    return dict(parent_menus=24998, child_menus=25826, max_probability_error=error,
        choices_equal=True, tolerance=5e-5)


def require_step0(identity, reg, proof):
    tensors = proof['exact_tensors']; parity = proof['all_parent_prediction_parity']
    if (proof['checked_before_first_optimizer_update'] is not True
            or proof['step0_bundle_identity'] != identity['identity_sha256']
            or proof['parent_bundle_identity'] != reg['curriculum']['parent_identity']
            or tensors['parent_adapter_exact'] is not True or tensors['parent_head_exact'] is not True
            or tensors['adapter_tensors'] != 372 or tensors['head_tensors'] != 4
            or parity['parent_menus'] != 24998 or parity['child_menus'] != 25826
            or parity['choices_equal'] is not True or not 0 <= parity['max_probability_error'] < 5e-5):
        raise ValueError('Actual exact tensors and full24998-parent serving parity before first update required')
