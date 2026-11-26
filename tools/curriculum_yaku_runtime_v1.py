"""Shared actual yaku-course policy, warm-start and finite evidence contracts."""
import hashlib
import json
import math
from pathlib import Path

from tools.curriculum_transfer import read, sha
from tools.curriculum_complete_legal_class_policy_v1 import select, METHOD as ROUTE
from tools.curriculum_yaku_objective_v1 import METHOD, MASS

ROOT = Path(__file__).resolve().parents[1]


def require_registration(reg):
    c = reg['curriculum']
    if (reg['status'] != 'yaku2500_registered_not_started' or reg['registered_final_step'] != 2500
            or reg['train_records'] != 91247 or reg['validation_records'] != 24516
            or c['epochs'] != 2 or c['accumulation'] != 73 or c['seed'] != 2026112305
            or c['loss_method'] != METHOD or reg['course_loss_mass'] != MASS or c['selection_method'] != ROUTE
            or c['expected_record_exposures'] != 182494 or c['validation_and_game_steps'] != [0, 625, 1250, 1875, 2500]
            or c['optimizer_resume'] is not False or c['new_optimizer'] is not True):
        raise ValueError('Frozen actual yaku2500 registration required')


def policy_contract(identity, reg):
    seal = read(reg['curriculum']['parent_seal']); spec = dict(seal['policy'], parameter_identity=identity['identity_sha256'])
    if (spec['method'] != ROUTE or spec['route_sha256'] != sha(ROOT/'tools/curriculum_complete_legal_class_policy_v1.py')
            or spec['RI_class_rule_sha256'] != sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py')
            or spec['runtime_source_sha256'] != reg['source_sha256']): raise ValueError('Frozen full-legal RI execution differs')
    return spec, hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def prediction_parity(parent_rows, parent_predictions, child_rows, child_predictions):
    from tools.curriculum_parent import require_replay
    require_replay(parent_rows, child_rows)
    old = {p['id']: p for p in parent_predictions}; current = {p['id']: p for p in child_predictions}
    if (len(old) != 23688 or len(current) != 24516
            or len(old) != len(parent_predictions) or len(current) != len(child_predictions)
            or len(parent_rows) != 23688 or len(child_rows) != 24516
            or set(old) != {r['_meta']['id'] for r in parent_rows}
            or set(current) != {r['_meta']['id'] for r in child_rows}):
        raise ValueError('Full distinct parent and child predictions required')
    error = 0.
    for row in parent_rows:
        rid = row['_meta']['id']; a = old[rid]; b = current[rid]
        if a['keys'] != b['keys']: raise ValueError('Actual parent menu order differs')
        if len(a['p']) != len(b['p']): raise ValueError('Actual parent probability shape differs')
        if any(not math.isfinite(v) or v < 0 for v in a['p']+b['p']):
            raise ValueError('Nonfinite parent or child probability')
        error = max(error, max(abs(x-y) for x, y in zip(a['p'], b['p'])))
        kinds = [row['questions']['action']['criteria'][k]['type'] for k in a['keys']]
        if select(kinds, a['p']) != select(kinds, b['p']): raise ValueError('Warm-start complete-legal fixed-RI choice changed')
    if not math.isfinite(error) or error >= 5e-5: raise ValueError('Full parent probability parity failed')
    return dict(parent_menus=23688, child_menus=24516, max_probability_error=error, choices_equal=True, tolerance=5e-5)


def require_step0(identity, reg, proof):
    tensors = proof['exact_tensors']; parity = proof['all_parent_prediction_parity']
    if (proof['checked_before_first_optimizer_update'] is not True or proof['step0_bundle_identity'] != identity['identity_sha256']
            or proof['parent_bundle_identity'] != reg['curriculum']['parent_identity']
            or not tensors['parent_adapter_exact'] or not tensors['parent_head_exact']
            or tensors['adapter_tensors'] != 372 or tensors['head_tensors'] != 4
            or parity['parent_menus'] != 23688 or parity['child_menus'] != 24516
            or parity['choices_equal'] is not True or not 0 <= parity['max_probability_error'] < 5e-5):
        raise ValueError('Actual exact tensors/full prediction proof required before step0 games')
