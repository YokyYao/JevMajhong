import unittest
from tools.evaluate_endgame_pursuit import select_targets, summarize


class TargetedSourceEvaluationTests(unittest.TestCase):
    def test_source_selection_deduplicates_and_does_not_use_final_results(self):
        def row(seed, key, cell, target=True):
            return {'seed': seed, 'selection_key': key, 'coverage_stratum': {'cell': cell, 'target': target}}
        sources = [{'selected': [row(1, 'b', 'near3'), row(1, 'a', 'near4')], 'final_scores': [1,2,3,4]},
                   {'selected': [row(2, 'c', 'near3')]}, {'selected': [row(3, 'd', 'far', False)]}]
        selected = select_targets(sources, 2)
        self.assertEqual({r['seed'] for r in selected}, {1,2})
        sources[0]['final_scores'] = [100000,0,0,0]
        self.assertEqual(select_targets(sources, 2), selected)

    def test_top2_and_escape_are_reported_separately_under_risk(self):
        def outcome(rank, utility):
            return {'rank': rank, 'utility': utility, 'net_points_from_window': 0, 'declined_wins': 0}
        rows = [{'seed': 1, 'initial_rank': 3, 'baseline': outcome(3,0), 'candidate': outcome(2,1)},
                {'seed': 2, 'initial_rank': 4, 'baseline': outcome(3,0), 'candidate': outcome(2,1)},
                {'seed': 3, 'initial_rank': 4, 'baseline': outcome(3,0), 'candidate': outcome(4,-.05)}]
        report = summarize(rows)
        fourth = report['by_initial_rank']['4']
        self.assertEqual(fourth['candidate']['top2_rate'], .5)
        self.assertEqual(fourth['baseline']['top2_rate'], 0)
        self.assertEqual(fourth['candidate']['escape_fourth_rate'], .5)
        self.assertEqual(fourth['baseline']['escape_fourth_rate'], 1)
        self.assertGreater(report['paired_utility_gain'], 0)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]])


if __name__ == '__main__':
    unittest.main()
