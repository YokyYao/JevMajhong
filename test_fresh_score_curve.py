"""Known-return score curves, complete wall grouping and shared parent credit."""
import unittest
from fresh_score_curve import summarize


def rows():
    cfg = {'seed_start': 100, 'seeds': 4, 'stage': 'development'}
    result = []
    for seed in range(100, 104):
        for suite in ('closed', 'mixed', 'strong'):
            for seat in range(4):
                values = {'parent': 0, 'step-000050': (seed - 99) * 100,
                    'step-000100': 200, 'step-000200': -100}
                for name, gain in values.items():
                    result.append(dict(policy=name, seed=seed, suite=suite, seat=seat,
                        score=25000 + gain, rank=seat + 1, hands=1, hand_wins=0,
                        dealins=0, calls=0, point_components={'test_credit': gain},
                        settlement_ledger_passed=True))
    return result, cfg


class FreshScoreCurveTest(unittest.TestCase):
    def test_net_points_choose_winner_and_parent_is_counted_once(self):
        data, cfg = rows(); actual = summarize(data, cfg, [50, 100, 200])
        self.assertEqual(actual['best_step'], 50); self.assertEqual(actual['confirmation_step'], 50)
        self.assertEqual(actual['games'], 192); self.assertEqual(actual['parent_games_computed_once'], 48)
        comparison = actual['comparisons']['50']['overall']['candidate_minus_reference']
        self.assertEqual(comparison['gain'], 250.); self.assertEqual(comparison['independent_wall_streams'], 4)
        for step in ('50', '100', '200'):
            self.assertEqual(actual['comparisons'][step]['overall']['policies']['reference']['games'], 48)

    def test_nonpositive_candidates_do_not_trigger_confirmation(self):
        data, cfg = rows()
        for r in data:
            if r['policy'] != 'parent':
                r['score'] = 25000; r['point_components'] = {'test_credit': 0}
        actual = summarize(data, cfg, [50, 100, 200])
        self.assertEqual(actual['best_step'], 50); self.assertIsNone(actual['confirmation_step'])

    def test_incomplete_or_duplicate_grid_is_rejected(self):
        data, cfg = rows()
        with self.assertRaises(ValueError): summarize(data[:-1], cfg, [50, 100, 200])
        with self.assertRaises(ValueError): summarize(data + [data[0]], cfg, [50, 100, 200])
        with self.assertRaises(ValueError): summarize(data, cfg, [50, 50, 200])

    def test_confirmation_keeps_one_winner_without_reranking(self):
        data, cfg = rows(); cfg['stage'] = 'independent_confirmation'
        data = [r for r in data if r['policy'] in ('parent', 'step-000050')]
        actual = summarize(data, cfg, [50])
        self.assertEqual(actual['evaluated_confirmation_step'], 50)
        self.assertIsNone(actual['confirmation_step'])
        self.assertFalse(actual['requires_independent_confirmation'])
        with self.assertRaises(ValueError): summarize(data, cfg, [50, 100])


if __name__ == '__main__': unittest.main()
