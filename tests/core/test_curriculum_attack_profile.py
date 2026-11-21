import copy
import math
from pathlib import Path
import json
import unittest

from tools.curriculum_attack_losses import mixture
from tools.curriculum_attack_profile import profile, target_ce
from tools.audit_curriculum_attack import quality, paired_comparison, reference_operations, operation_rates
from tools.curriculum_operation_coverage import OPERATIONS
from tools.register_curriculum_attack import validate_recipe

ROOT = Path(__file__).resolve().parents[2]


def row(kind, phase, ident):
    criteria = dict(a0=dict(type='dahai'), a1=dict(type='dahai'))
    q = dict(type='choice', instructions='choose', criteria=criteria, label='a0', src='curriculum3:'+kind)
    if kind == 'expert:riichi_yes':
        criteria['a0']['type'] = 'reach'
    elif kind == 'expert:riichi_no':
        criteria['a1']['type'] = 'reach'
    if kind.startswith('mc:'):
        criteria['a2'] = dict(type='none')
        q['target'] = dict(a0=.6, a1=.4, a2=0.)
    return dict(_meta=dict(id=ident, phase=phase, label_type='paired_future_masked' if kind.startswith('mc:') else 'expert_replay'),
        state={}, questions=dict(action=q))


class AttackProfileTests(unittest.TestCase):
    def setUp(self):
        kinds = [('expert:'+k, p) for k, p in [('plain', 'closed_discard'), ('riichi_yes', 'closed_discard'),
            ('riichi_no', 'closed_discard'), ('open', 'open_discard'), ('call', 'call_window'), ('kan', 'kan_window')]]
        kinds += [('mc:'+p, p) for p in ('closed_discard', 'open_discard', 'call_window', 'kan_window')]
        self.rows = [row(k, p, str(i)) for i, (k, p) in enumerate(kinds)]
        self.mass = {k: .1 for k, _ in kinds}
        self.phases = {p: .25 for _, p in kinds}
        self.pred = [dict(id=r['_meta']['id'], keys=list(r['questions']['action']['criteria']),
            p=[.6, .4, 0.] if r['_meta']['label_type'] == 'paired_future_masked' else [.8, .2]) for r in self.rows]

    def test_unknown_predicted_mass_does_not_turn_into_masked_loss(self):
        a = profile(self.rows, self.pred, self.phases, self.mass)
        changed = copy.deepcopy(self.pred)
        for p in changed[-4:]:
            p['p'] = [.006, .004, .99]
        b = profile(self.rows, changed, self.phases, self.mass)
        self.assertAlmostEqual(a['weighted_target_ce'], b['weighted_target_ce'])
        self.assertEqual(a['expert_accuracy'], b['expert_accuracy'])
        self.assertEqual(sum(a['MC_unlabelled_actions_selected'].values()), 0)
        self.assertEqual(sum(b['MC_unlabelled_actions_selected'].values()), 4)
        ce, entropy = target_ce(self.rows[-1]['questions']['action'], ['a0', 'a1', 'a2'], [.006, .004, .99])
        self.assertAlmostEqual(ce, -.6*math.log(.6)-.4*math.log(.4))
        self.assertAlmostEqual(ce, entropy)

    def test_invalid_predictions_cannot_drop_rows_or_sneak_in_nan(self):
        for predictions in (self.pred[:-1], [self.pred[0]]+self.pred[:-1]):
            with self.assertRaises(ValueError):
                profile(self.rows, predictions, self.phases, self.mass)
        broken = copy.deepcopy(self.pred)
        broken[0]['p'] = [float('nan'), .2]
        with self.assertRaises(ValueError):
            profile(self.rows, broken, self.phases, self.mass)

    def test_recipe_uses_complete_real_sized_population_and_fixed_final(self):
        counts = [8826, 194, 272, 5590, 3569, 1786, 717, 254, 1503, 248]
        pool = [r for r, n in zip(self.rows, counts) for _ in range(n)]
        recipe = mixture(pool)
        data = dict(loss_mass=recipe['loss_mass'], example_weights=recipe['example_weights'],
            count_only_accumulation_proposal=dict(accumulation=127, final_fraction=99/127))
        c = json.loads((ROOT/'configs/training/curriculum-stage3-attack-v1.json').read_text())
        self.assertEqual(validate_recipe(c, data, pool), 362)
        for updates in (dict(epochs=1), dict(candidate_rule='minimum_validation_CE'), dict(validation_every_steps=90),
                        dict(optimizer_resume=True), dict(all10_strata_each_update=False)):
            with self.assertRaises(ValueError):
                validate_recipe(dict(c, **updates), data, pool)


class AttackQualityTests(unittest.TestCase):
    def test_cached_operation_predictions_are_recomputed_and_identity_checked(self):
        r = dict(_meta=dict(id='expert', group_id='g', label_type='expert_replay'),
            questions=dict(action=dict(criteria=dict(a=dict(type='daiminkan'),b=dict(type='none')),label='b')))
        counts = {op: {k:0 for k in ('tp','fn','tn','fp')} for op in OPERATIONS}
        counts['daiminkan']['tn'] = 1
        d = dict(status='calls_data_token_and_reference_diagnostic_complete', reference_identity='v8',
            reference_menus=1, reference_operation_counts=counts)
        predictions = [dict(id='expert',group_id='g',probabilities=dict(a=.1,b=.9))]
        self.assertEqual(reference_operations([r],predictions,d,'v8'),counts)
        for identity, values in (('another',predictions),('v8',[]),
                                 ('v8',[dict(predictions[0],probabilities=dict(a=.9,b=.1))])):
            with self.assertRaises(ValueError):reference_operations([r],values,d,identity)

    def test_operation_report_exposes_asymmetric_acceptance_and_decline(self):
        value = operation_rates(dict(daiminkan=dict(tp=88,fn=0,tn=3,fp=105)))['daiminkan']
        self.assertEqual(value['observed_acceptance_recall'],1.)
        self.assertAlmostEqual(value['observed_decline_recall'],3/108)
        self.assertAlmostEqual(value['balanced_agreement'],.5*(1+3/108))

    def test_better_simulated_ce_does_not_hide_dealin_or_net_regression(self):
        c = json.loads((ROOT/'configs/training/curriculum-stage3-attack-v1.json').read_text())
        initial = dict(stratum_accuracy={'expert:plain': .5}, riichi_type_counts=dict(tp=1, fn=1, tn=1, fp=1),
            operation_type_counts={op: dict(tp=1, fn=1, tn=1, fp=1) for op in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan')},
            stratum_ce={'mc:'+p: 1. for p in ('closed_discard', 'open_discard', 'call_window', 'kan_window')})
        final = copy.deepcopy(initial)
        final['stratum_ce'] = {k: .9 for k in initial['stratum_ce']}
        comparison = dict(net_points={'gain': 100.}, dealin={'gain': -.01})
        self.assertTrue(quality(c, initial, final, comparison)['passed'])
        for key, bad_value in (('net_points', -1.), ('dealin', .01)):
            bad = copy.deepcopy(comparison)
            bad[key]['gain'] = bad_value
            self.assertFalse(quality(c, initial, final, bad)['passed'])
        self.assertFalse(quality(c, initial, initial, comparison)['passed'])

    def test_paired_intervals_keep_whole_seeds_and_require_matching_games(self):
        reference = [dict(seed=s, seat=p, profile=kind, rank=2, net_points=0., wins=1, hands=4, dealin_hands=0)
            for s in range(8) for p in range(4) for kind in ('ordinary', 'calling')]
        final = [dict(r, net_points=100.) for r in reference]
        result = paired_comparison(final, reference)
        self.assertEqual(result['net_points'], dict(gain=100., ci95=[100., 100.], seed_groups=8))
        for bad in (reference[:-1], [reference[0]]+reference[:-1]):
            with self.assertRaises(ValueError):
                paired_comparison(final, bad)


if __name__ == '__main__':
    unittest.main()
