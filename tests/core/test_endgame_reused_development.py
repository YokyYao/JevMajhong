import copy
import unittest
from tools.evaluate_endgame_reused_development import combined_eligible, standard_summary


class ReusedDevelopmentTests(unittest.TestCase):
    def test_top2_and_escape_remain_distinct_and_group_weighted(self):
        def row(seed, initial, before, after):
            def outcome(rank):
                return {'rank': rank, 'utility': [1.1, 1., 0., -.05][rank - 1],
                    'net_points_from_window': 1000, 'hands_ended_after_window': 1,
                    'own_wins_after_window': 1, 'dealins_after_window': 0}
            return {'seed': seed, 'initial_rank': initial, 'baseline': outcome(before), 'candidate': outcome(after)}
        result = standard_summary([row(1,3,3,2), row(1,4,4,3), row(2,2,2,2)])
        self.assertEqual(result['placement_rates']['fourth_escape']['candidate'], 1.)
        self.assertAlmostEqual(result['terminal_utility_gain'], (1.05/2)/2)
        self.assertEqual(result['baseline_rollouts_reused'], 3)
        self.assertTrue(result['consumed_development_pool'])

    def test_target_gain_cannot_hide_standard_escape_regression(self):
        general = {'terminal_utility_gain': .1, 'placement_rates': {name: {'candidate_minus_baseline': 0.}
            for name in ('third_to_top2','fourth_escape','hold_top2')}}
        target = {'paired_utility_gain': .1, 'by_initial_rank': {
            '3': {'candidate': {'top2_count': 2}, 'baseline': {'top2_count': 1}},
            '4': {'candidate': {'top2_count': 5}, 'baseline': {'top2_count': 5}}}}
        self.assertTrue(combined_eligible(general,target))
        bad = copy.deepcopy(general); bad['placement_rates']['fourth_escape']['candidate_minus_baseline']=-.01
        self.assertFalse(combined_eligible(bad,target))
        bad = copy.deepcopy(target); bad['by_initial_rank']['4']['candidate']['top2_count']=4
        self.assertFalse(combined_eligible(general,bad))
        bad = copy.deepcopy(target); bad['paired_utility_gain']=0.
        self.assertFalse(combined_eligible(general,bad))


if __name__ == '__main__':
    unittest.main()
