import copy
import random
from tempfile import TemporaryDirectory
import unittest

from tools.curriculum_placement_runtime_v1 import (PlacementEpochRandom, random_proxy, prediction_parity,
    require_step0, require_registration, matched_curve, LOSS, ROUTE, SCHEMA)
from tools.curriculum_placement_exposures_v1 import epoch_order, choose_accumulation
from tools.curriculum_placement_mix_v1 import REPLAY_MASS, NEW_SOURCES, RECORDS, masses
from tools.prepare_curriculum_placement_exposures_v1 import proposed_schedule


def categories_and_orders():
    counts = {s: max(1, int(v*RECORDS)) for s, v in REPLAY_MASS.items()}
    counts[max(counts, key=counts.get)] += RECORDS-sum(counts.values())
    counts.update({s: 16 for s in NEW_SOURCES})
    categories = [s for s, n in counts.items() for _ in range(n)]
    accum = choose_accumulation(len(categories)); rng = random.Random(2026112306)
    orders = []; receipts = []
    for epoch in (1, 2):
        order, receipt = epoch_order(categories, rng, accum)
        orders.append(order); receipts.append(dict(epoch=epoch, **receipt))
    return categories, accum, orders, receipts


def parity_fixture():
    def row(rid, hand, instructions):
        return dict(_meta=dict(id=rid), state=dict(round_wind=0, hand_number=hand),
            questions=dict(action=dict(type='choice', instructions=instructions, src='old', label='a',
                criteria=dict(a=dict(type='dahai'), b=dict(type='reach')))))
    parent = [row('early', 1, 'old'), row('late', 4, 'old')]
    child = copy.deepcopy(parent); child[-1]['questions']['action']['instructions'] = 'placement'
    child.append(row('new', 4, 'placement'))
    old = [dict(id=r['_meta']['id'], keys=['a','b'], p=[.8,.2]) for r in parent]
    current = copy.deepcopy(old)+[dict(id='new', keys=['a','b'], p=[.9,.1])]
    current[1]['p'] = [.1,.9]
    return parent, old, child, current


def registration_fixture():
    total = RECORDS+32*16; schedule = proposed_schedule(total)
    c = dict(loss_method=LOSS, selection_method=ROUTE, placement_input_schema=SCHEMA, seed=2026112306,
        epochs=2, batch=1, new_optimizer=True, optimizer_resume=False,
        learning_rate=3e-6, head_learning_rate=1.5e-5, final_candidate_only=True, automatic_release=False,
        stage_zero_seed_groups=8, stage_end_seed_groups=8, curve_seed_groups=2,
        zero_curve_use_matching16_subset=True, parent_games_not_step0_cache=True,
        new_policy_zero64_actual_games_required=True, accumulation=schedule['accumulation'],
        expected_record_exposures=schedule['expected_record_exposures'],
        validation_and_game_steps=schedule['validation_and_game_steps'],
        eval_every_steps=schedule['eval_every_steps'], checkpoint_every_steps=100,
        curve_seed_start=2026200000, opponent_profiles=['efficiency', 'calling'])
    return dict(status='placement_data_parent_and_budget_registered_not_fit_admitted', curriculum=c,
        fit_launch_admitted=False, optimizer_updates=0, prelaunch_integration_and_code_binding_required=True,
        automatic_release=False, final_reserved_grid_opened=False, same_execution_policy_as_parent=False,
        course_loss_mass={s: float(v) for s, v in masses().items()}, train_records=total,
        registered_final_step=schedule['proposed_final_step'],
        expected_record_exposures=schedule['expected_record_exposures'])


class PlacementRuntimeTests(unittest.TestCase):
    def test_fresh64_baseline_and_final_are_separate_from_same16_curves(self):
        reg = registration_fixture(); require_registration(reg); start = reg['curriculum']['curve_seed_start']
        games = [dict(seed=seed, seat=seat, profile=profile, completed=True, ledger_passed=True)
            for seed in range(start, start+8) for seat in range(4)
            for profile in reg['curriculum']['opponent_profiles']]
        aggregate = lambda rows: dict(matches=len(rows), seed_groups=len({r['seed'] for r in rows}))
        for step, grid in ((0, 'policy_zero64'), (reg['registered_final_step'], 'stage_end64')):
            summary, curve = matched_curve(games, reg, step, aggregate)
            self.assertEqual(summary['matches'], 64); self.assertEqual(summary['grid'], grid)
            self.assertEqual(curve['matches'], 16); self.assertEqual(curve['full_development_matches'], 64)
            self.assertFalse(curve['parent_cache_reused_as_step0'])
        intermediate = reg['curriculum']['validation_and_game_steps'][1]
        summary, _ = matched_curve(games[:16], reg, intermediate, aggregate)
        self.assertEqual(summary['matches'], 16)
        for bad in (games[:16], games[:-1]+[games[0]], [dict(r, ledger_passed=False) for r in games]):
            with self.assertRaises(ValueError): matched_curve(bad, reg, 0, aggregate)

    def test_changed_budget_resume_or_old_zero_cache_cannot_be_admitted(self):
        for field, value in (('stage_zero_seed_groups', 2), ('parent_games_not_step0_cache', False),
                             ('optimizer_resume', True), ('loss_method', 'old'), ('epochs', 1)):
            reg = registration_fixture(); reg['curriculum'][field] = value
            with self.assertRaises(ValueError): require_registration(reg)

    def test_upstream_rng_adapter_executes_exact_two_prepared_permutations(self):
        cats, accum, orders, receipts = categories_and_orders()
        with TemporaryDirectory() as directory:
            proxy, created = random_proxy(random, 2026112306, cats, accum, directory, receipts)
            rng = proxy.Random(2026112306)
            for expected in orders:
                actual = list(range(len(cats))); rng.shuffle(actual)
                self.assertEqual(actual, expected)
            self.assertEqual(created, [rng]); self.assertEqual(rng.epochs, 2)
            with self.assertRaises(ValueError): rng.shuffle(actual)
            with self.assertRaises(ValueError): proxy.Random(2026112306)
            self.assertEqual(proxy.Random(42).random(), random.Random(42).random())

    def test_source_receipt_tampering_fails_without_advancing_rng_or_indices(self):
        cats, accum, _, receipts = categories_and_orders(); bad = copy.deepcopy(receipts)
        bad[0]['source_exposure_counts_by_update'][0][cats[0]] = 999
        with TemporaryDirectory() as directory:
            rng = PlacementEpochRandom(2026112306, cats, accum, directory, bad)
            before = rng.getstate(); indices = list(range(len(cats)))
            with self.assertRaises(ValueError): rng.shuffle(indices)
            self.assertEqual(rng.getstate(), before); self.assertEqual(rng.epochs, 0)
            self.assertEqual(indices, list(range(len(cats))))

    def test_changed_late_probabilities_are_not_claimed_as_parent_parity(self):
        data = parity_fixture(); result = prediction_parity(*data)
        self.assertEqual(result['early_unchanged_menus'], 1)
        self.assertEqual(result['changed_late_menus'], 1)
        self.assertEqual(result['max_early_probability_error'], 0.)
        self.assertFalse(result['full_parent_prediction_parity_claimed'])
        for kind in ('early_input', 'early_probability', 'late_nonfinite', 'missing_prediction', 'late_unchanged'):
            a,b,c,d = parity_fixture()
            if kind == 'early_input': c[0]['questions']['action']['instructions'] = 'changed'
            elif kind == 'early_probability': d[0]['p'] = [.7,.3]
            elif kind == 'late_nonfinite': d[1]['p'] = [float('nan'), .5]
            elif kind == 'missing_prediction': d.pop()
            else: c[1]['questions']['action']['instructions'] = 'old'
            with self.assertRaises(ValueError): prediction_parity(a,b,c,d)

    def test_only_complete_exact_preupdate_inheritance_admits_new_baseline(self):
        proof = dict(checked_before_first_optimizer_update=True, step0_bundle_identity='child',
            parent_bundle_identity='parent', exact_tensors=dict(parent_adapter_exact=True,
            parent_head_exact=True, adapter_tensors=372, head_tensors=4),
            early_prediction_parity=dict(parent_menus=24516, child_menus=25000,
                early_unchanged_menus=20000, late_parent_menus=4516, changed_late_menus=4516,
                full_parent_prediction_parity_claimed=False, early_choices_equal=True,
                max_early_probability_error=0.))
        reg = dict(curriculum=dict(parent_identity='parent'), validation_records=25000)
        require_step0(dict(identity_sha256='child'), reg, proof)
        for field, value in (('parent_menus', 100), ('changed_late_menus', 0),
                             ('max_early_probability_error', float('nan')),
                             ('full_parent_prediction_parity_claimed', True)):
            bad = copy.deepcopy(proof); bad['early_prediction_parity'][field] = value
            with self.assertRaises(ValueError): require_step0(dict(identity_sha256='child'), reg, bad)
