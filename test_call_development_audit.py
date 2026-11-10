import copy
import unittest
from scripts.audit_call_development import audit_suite


def fixture():
    cfg = {'development_seed_start': 101, 'development_seeds': 2, 'previous_call_checkpoint': 'previous'}
    policies = {}; rows = []
    for name, rank, unit, mean in [('reference', 3, 0, 0), ('candidate', 1, 100, 150), ('previous', 2, 50, 75)]:
        for seed in (101, 102):
            for seat in range(4):
                rows.append({'suite': 'closed', 'policy': name, 'seed': seed, 'seat': seat,
                    'score': 25000 + unit * (seed - 100), 'rank': rank, 'hands': 10,
                    'hand_wins': 2, 'dealins': 1})
        policies[name] = {'games': 8, 'independent_seeds': 2, 'firsts': 8 if rank == 1 else 0,
            'top2s': 8 if rank <= 2 else 0, 'first_rate': float(rank == 1),
            'top2_rate': float(rank <= 2), 'fourth_rate': 0., 'mean_rank': rank,
            'mean_final_points': 25000 + mean, 'mean_net_points': mean,
            'hand_win_rate': .2, 'dealin_rate': .1}
    # Two independent wall means, not eight independent seat outcomes.
    half = 12.706204736432095 * 50
    primary = {'gain': 150., 'ci95': [150 - half, 150 + half]}
    secondary = {'gain': 75., 'ci95': [75 - half / 2, 75 + half / 2]}
    summary = {'policies': policies, 'comparisons': {'candidate_minus_reference': primary,
        'candidate_minus_previous': copy.deepcopy(secondary), 'previous_minus_reference': copy.deepcopy(secondary)},
        'paired_net_points_gain': 150., 'paired_seed_cluster_ci95': primary['ci95'].copy(),
        'independent_wall_streams': 2, 'is_final_confirmation': False}
    return rows, summary, cfg


class DevelopmentAuditTests(unittest.TestCase):
    def test_reconciles_three_policies_and_wall_cluster_interval(self):
        rows, summary, cfg = fixture()
        result = audit_suite(rows, summary, cfg, 'closed')
        self.assertEqual(result['games'], 24)
        self.assertEqual(result['independent_wall_clusters'], 2)
        self.assertEqual(result['comparisons']['candidate_minus_reference']['gain'], 150)

    def test_missing_or_duplicated_game_is_rejected(self):
        rows, summary, cfg = fixture()
        for changed in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError): audit_suite(changed, summary, cfg, 'closed')

    def test_wrongly_narrow_interval_cannot_pass(self):
        rows, summary, cfg = fixture()
        summary['comparisons']['candidate_minus_reference']['ci95'] = [140, 160]
        with self.assertRaisesRegex(ValueError, 'ci95'): audit_suite(rows, summary, cfg, 'closed')

    def test_changed_strength_or_phase_label_is_rejected(self):
        rows, summary, cfg = fixture()
        for changed in ('mean_net_points', 'is_final_confirmation'):
            bad = copy.deepcopy(summary)
            if changed == 'mean_net_points': bad['policies']['candidate'][changed] += 1
            else: bad[changed] = True
            with self.assertRaises(ValueError): audit_suite(rows, bad, cfg, 'closed')


if __name__ == '__main__': unittest.main()
