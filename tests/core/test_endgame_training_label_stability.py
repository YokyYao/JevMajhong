import copy
import json
from pathlib import Path
import unittest

from tools.audit_endgame_training_label_stability import (validate_config,outcome_arrays,
    target_for_worlds,leave_one_out,aggregate)


class TrainingLabelStabilityTests(unittest.TestCase):
    def test_registration_forbids_new_worlds_validation_and_training(self):
        path=Path(__file__).resolve().parents[2]/'configs/training/endgame-training-label-stability-v1.json'
        config=json.loads(path.read_text());validate_config(config)
        for change in ({'validation_data_used':True},{'fit_registered':True},
                       {'new_simulations':1},{'legacy_positions':9},{'native_worlds':32}):
            with self.assertRaises(ValueError):validate_config(dict(config,**change))

    def test_one_lucky_world_does_not_become_heldout_improvement(self):
        reference={'a':.8,'b':.2};utilities={'a':[0.]*4,'b':[1.1,0.,0.,0.]}
        escape={'a':[1.]*4,'b':[1.]*4}
        result=leave_one_out(reference,utilities,escape,['a','b'])
        self.assertTrue(result['full']['strict'])
        self.assertAlmostEqual(result['full_choice_mean_utility_difference'],.275)
        self.assertEqual(result['strict_remaining_fraction'],.75)
        self.assertEqual(result['choice_matches_full_fraction'],.75)
        self.assertEqual(result['heldout_mean_utility_difference'],0.)
        self.assertEqual(result['worlds'][0]['choice'],'a')

    def test_consistent_paired_advantage_survives_every_heldout_world(self):
        result=leave_one_out({'a':.8,'b':.2},{'a':[0.]*4,'b':[1.]*4},
            {'a':[1.]*4,'b':[1.]*4},['a','b'])
        self.assertEqual(result['strict_remaining_fraction'],1.)
        self.assertEqual(result['support_matches_full_fraction'],1.)
        self.assertEqual(result['heldout_mean_utility_difference'],1.)

    def test_expected_gain_cannot_override_the_original_escape_floor(self):
        result=leave_one_out({'a':.8,'b':.2},{'a':[0.]*4,'b':[1.,-.05,-.05,-.05]},
            {'a':[1.]*4,'b':[1.,0.,0.,0.]},['a','b'])
        self.assertFalse(result['full']['strict']);self.assertEqual(result['full']['choice'],'a')
        self.assertEqual(result['heldout_mean_utility_difference'],0.)

    def test_paired_array_and_index_contracts_are_not_silently_repaired(self):
        ref={'a':.8,'b':.2};utilities={'a':[0.]*4,'b':[1.]*4};escape={'a':[1.]*4,'b':[1.]*4}
        with self.assertRaises(ValueError):target_for_worlds(ref,utilities,escape,['a','b'],[0,0])
        with self.assertRaises(ValueError):target_for_worlds(ref,utilities,dict(escape,b=[1.]),['a','b'],[0,1])
        with self.assertRaises(ValueError):target_for_worlds(ref,dict(utilities,b=[float('nan')]*4),escape,['a','b'],[0,1])

    def test_cohort_aggregation_weights_positions_not_number_of_worlds(self):
        four=leave_one_out({'a':.8,'b':.2},{'a':[0.]*4,'b':[1.1]*4},
            {'a':[1.]*4,'b':[1.]*4},['a','b'])
        sixteen=leave_one_out({'a':.8,'b':.2},{'a':[0.]*16,'b':[0.]*16},
            {'a':[1.]*16,'b':[1.]*16},['a','b'])
        summary=aggregate([{'audit':four},{'audit':sixteen}])
        self.assertEqual(summary['paired_worlds'],20)
        self.assertAlmostEqual(summary['heldout_mean_utility_difference'],.55)

    def test_raw_scores_rank_history_and_split_are_verified(self):
        from jevmahjong.endgame.context import terminal_utility
        def outcome(scores,ranks):
            return {'scores':scores,'ranks':ranks,'ledger_passed':True,'utility':terminal_utility(ranks[3]),
                    'hand_history':{'final_scores':scores[:]}}
        a=outcome([32000,25000,22000,21000],[1,2,3,4])
        b=outcome([26000,24000,22000,28000],[2,3,4,1])
        label={'status':'complete_terminal_menu','source':{'split':'train','seat':3},
            'baseline_identity_sha256':'formal','worlds_per_action':4,'continued_until_native_match_done':True,
            'record':{'questions':{'action':{'criteria':{'a':{},'b':{}}}}},'actual_actions':2,
            'outcomes':[[copy.deepcopy(a) for _ in range(4)],[copy.deepcopy(b) for _ in range(4)]]}
        utilities,escape=outcome_arrays(label,['a','b'],3,4,'formal')
        self.assertEqual(utilities['a'],[-.05]*4);self.assertEqual(utilities['b'],[1.1]*4)
        self.assertEqual(escape['a'],[0.]*4);self.assertEqual(escape['b'],[1.]*4)
        bad=copy.deepcopy(label);bad['source']['split']='validation'
        with self.assertRaises(ValueError):outcome_arrays(bad,['a','b'],3,4,'formal')
        bad=copy.deepcopy(label);bad['outcomes'][0][0]['ranks']=[4,3,2,1]
        with self.assertRaises(ValueError):outcome_arrays(bad,['a','b'],3,4,'formal')
        bad=copy.deepcopy(label);bad['outcomes'][0][0]['hand_history']['final_scores'][0]+=100
        with self.assertRaises(ValueError):outcome_arrays(bad,['a','b'],3,4,'formal')


if __name__=='__main__':unittest.main()
