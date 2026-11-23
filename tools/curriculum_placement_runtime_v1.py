"""Exact prospective phase4 epoch execution and honest early-input inheritance.

No model loading or job submission occurs on import. A new child-policy
baseline is mandatory because late input instructions and win execution change.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from tools.curriculum_placement_exposures_v1 import epoch_order
from tools.curriculum_placement_objective_v1 import METHOD as LOSS
from tools.curriculum_placement_input_v1 import SCHEMA
from tools.curriculum_placement_full_win_policy_v1 import METHOD as ROUTE
from tools.curriculum_complete_legal_class_policy_v1 import select as ordinary_select
from tools.curriculum_endgame import in_endgame_scope
from tools.curriculum_transfer import read, sha
from tools.curriculum_placement_mix_v1 import masses
from tools.prepare_curriculum_placement_exposures_v1 import proposed_schedule

ROOT = Path(__file__).resolve().parents[1]


def require_registration(reg):
    c = reg['curriculum']; schedule = proposed_schedule(reg['train_records'])
    if (reg['status'] != 'placement_data_parent_and_budget_registered_not_fit_admitted'
            or reg['fit_launch_admitted'] or reg['optimizer_updates'] != 0
            or not reg['prelaunch_integration_and_code_binding_required']
            or c['loss_method'] != LOSS or c['selection_method'] != ROUTE
            or c['placement_input_schema'] != SCHEMA or c['seed'] != 2026112306
            or c['epochs'] != 2 or c['batch'] != 1 or not c['new_optimizer'] or c['optimizer_resume']
            or c['learning_rate'] != 3e-6 or c['head_learning_rate'] != 1.5e-5
            or not c['final_candidate_only'] or c['automatic_release'] or reg['automatic_release']
            or reg['final_reserved_grid_opened'] or reg['same_execution_policy_as_parent']
            or c['stage_zero_seed_groups'] != 8 or c['stage_end_seed_groups'] != 8
            or c['curve_seed_groups'] != 2 or not c['zero_curve_use_matching16_subset']
            or not c['parent_games_not_step0_cache'] or not c['new_policy_zero64_actual_games_required']
            or reg['course_loss_mass'] != {s: float(v) for s, v in masses().items()}
            or reg['registered_final_step'] != schedule['proposed_final_step']
            or c['accumulation'] != schedule['accumulation']
            or reg['expected_record_exposures'] != schedule['expected_record_exposures']
            or c['expected_record_exposures'] != schedule['expected_record_exposures']
            or c['validation_and_game_steps'] != schedule['validation_and_game_steps']
            or c['eval_every_steps'] != schedule['eval_every_steps']
            or c['checkpoint_every_steps'] != schedule['checkpoint_every_steps']):
        raise ValueError('Exact full two-epoch placement registration and fresh baseline required')


def evaluation_groups(reg, step):
    require_registration(reg)
    if step not in reg['curriculum']['validation_and_game_steps']:
        raise ValueError('Only registered saved evaluation points are eligible')
    return 8 if step in (0, reg['registered_final_step']) else 2


def matched_curve(games, reg, step, aggregate):
    """Keep fresh64 baseline/final summaries; journal the identical16 cells."""
    groups = evaluation_groups(reg, step); c = reg['curriculum']
    expected = {(seed, seat, profile)
        for seed in range(c['curve_seed_start'], c['curve_seed_start']+groups)
        for profile in c['opponent_profiles'] for seat in range(4)}
    cells = [(r['seed'], r['seat'], r['profile']) for r in games]
    if (len(cells) != len(expected) or set(cells) != expected
            or any(not r['completed'] or not r['ledger_passed'] for r in games)):
        raise ValueError('Every distinct actual registered full-match cell and ledger required')
    summary = aggregate(games)
    summary.update(grid='policy_zero64' if step == 0 else 'stage_end64' if groups == 8 else 'curve16',
        actual_new_policy_matches=len(games), parent_cache_reused_as_step0=False)
    subset = [r for r in games if r['seed'] < c['curve_seed_start']+2]
    curve = aggregate(subset)
    if curve['matches'] != 16 or curve['seed_groups'] != 2:
        raise ValueError('The common16 actual development cells are required at every point')
    curve.update(grid='curve16', full_development_matches=len(games),
        full_development_grid=summary['grid'], parent_cache_reused_as_step0=False)
    if groups == 8: summary['curve_grid_subset'] = aggregate(subset)
    return summary, curve


class PlacementEpochRandom(random.Random):
    def __init__(self, seed, categories, accumulation, directory, expected):
        super().__init__(seed)
        self.categories = list(categories); self.accumulation = accumulation
        self.directory = Path(directory); self.expected = expected; self.epochs = 0
        if len(expected) != 2:
            raise ValueError('Exactly two prepared epoch receipts required')

    def shuffle(self, order):
        if sorted(order) != list(range(len(self.categories))):
            raise ValueError('Complete actual expanded TRAIN index set required')
        if self.epochs >= 2:
            raise ValueError('Unregistered third epoch is not admitted')
        rng = random.Random(); rng.setstate(self.getstate())
        selected, receipt = epoch_order(self.categories, rng, self.accumulation)
        expected = self.expected[self.epochs]
        # Validate the entire source-count receipt, not just its permutation.
        if dict(epoch=self.epochs+1, **receipt) != expected:
            raise ValueError('Actual order or per-update exposures differ from prepared epoch')
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/f'epoch-{self.epochs+1:02d}.json').open('x') as stream:
            json.dump(expected, stream, indent=2); stream.write('\n')
        self.setstate(rng.getstate()); self.epochs += 1; order[:] = selected


def random_proxy(original, seed, categories, accumulation, directory, expected):
    created = []
    def factory(value=None):
        if type(value) is int and value == seed:
            if created:
                raise ValueError('Unexpected duplicate main placement training RNG')
            result = PlacementEpochRandom(seed, categories, accumulation, directory, expected)
            created.append(result); return result
        return original.Random(value)
    return SimpleNamespace(seed=original.seed, Random=factory), created


def visible_input(row):
    return dict(state=row['state'], questions={qid: {k: copy.deepcopy(v) for k, v in q.items()
        if k not in ('src', 'label', 'target')} for qid, q in row['questions'].items()})


def prediction_parity(parent_rows, parent_predictions, child_rows, child_predictions):
    """Check all inherited early inputs; explicitly exclude changed late inputs.

    Prediction keys/order and every probability are verified even on changed
    late rows, but probability equality is required only on identical early
    model input. Final tensors are checked separately by the training wrapper.
    """
    def index(rows, key):
        result = {key(r): r for r in rows}
        if not rows or len(result) != len(rows):
            raise ValueError('Complete distinct rows/predictions required')
        return result
    parents = index(parent_rows, lambda r: r['_meta']['id'])
    children = index(child_rows, lambda r: r['_meta']['id'])
    old = index(parent_predictions, lambda r: r['id'])
    current = index(child_predictions, lambda r: r['id'])
    if set(parents) != set(old) or set(children) != set(current) or not set(parents) <= set(children):
        raise ValueError('Every original parent and child validation prediction required')
    error = 0.; early = 0; late = 0; changed_late = 0
    for rid, row in parents.items():
        child = children[rid]; a = old[rid]; b = current[rid]
        if (a['keys'] != b['keys'] or len(a['keys']) != len(set(a['keys']))
                or set(a['keys']) != set(row['questions']['action']['criteria'])
                or set(b['keys']) != set(child['questions']['action']['criteria'])):
            raise ValueError('Inherited actual menu identity/order differs')
        for p in (a['p'], b['p']):
            if (len(p) != len(a['keys']) or any(not math.isfinite(v) or v < 0 for v in p)
                    or abs(sum(p)-1) > 1e-5):
                raise ValueError('Aligned finite normalized actual probabilities required')
        old_visible = visible_input(row); new_visible = visible_input(child)
        if in_endgame_scope(row['state']):
            late += 1; changed_late += old_visible != new_visible
            continue
        if old_visible != new_visible:
            raise ValueError('An early inherited input changed')
        early += 1
        error = max(error, max(abs(x-y) for x, y in zip(a['p'], b['p'])))
        kinds = [row['questions']['action']['criteria'][k]['type'] for k in a['keys']]
        if 'hora' not in kinds and ordinary_select(kinds, a['p']) != ordinary_select(kinds, b['p']):
            raise ValueError('Early complete-legal fixed-RI choice changed')
    if not early or not late or not changed_late or not 0 <= error < 5e-5:
        raise ValueError('Full early parity and explicit changed late instruction scope required')
    return dict(parent_menus=len(parents), child_menus=len(children), early_unchanged_menus=early,
        late_parent_menus=late, changed_late_menus=changed_late,
        max_early_probability_error=error, early_choices_equal=True, tolerance=5e-5,
        full_parent_prediction_parity_claimed=False)


def policy_contract(identity, reg):
    seal = read(reg['curriculum']['parent_seal']); spec = copy.deepcopy(seal['child_policy'])
    if (spec['method'] != ROUTE or spec['input_schema'] != SCHEMA
            or not spec['explicit_public_event_required'] or spec['runtime_source_sha256'] != reg['source_sha256']):
        raise ValueError('Exact full-win/public-event placement policy required')
    for name, expected in spec['route_files'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Placement policy source changed: '+name)
    spec['parameter_identity'] = identity['identity_sha256']
    return spec, hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def require_step0(identity, reg, proof):
    tensors = proof['exact_tensors']; parity = proof['early_prediction_parity']
    if (proof['checked_before_first_optimizer_update'] is not True
            or proof['step0_bundle_identity'] != identity['identity_sha256']
            or proof['parent_bundle_identity'] != reg['curriculum']['parent_identity']
            or not tensors['parent_adapter_exact'] or not tensors['parent_head_exact']
            or tensors['adapter_tensors'] != 372 or tensors['head_tensors'] != 4
            or parity['parent_menus'] != 24516 or parity['child_menus'] != reg['validation_records']
            or parity['early_unchanged_menus']+parity['late_parent_menus'] != 24516
            or parity['early_unchanged_menus'] <= 0 or not 0 < parity['changed_late_menus'] <= parity['late_parent_menus']
            or parity['full_parent_prediction_parity_claimed'] or not parity['early_choices_equal']
            or not 0 <= parity['max_early_probability_error'] < 5e-5):
        raise ValueError('Exact tensors and complete early parity before fresh child-policy baseline required')
