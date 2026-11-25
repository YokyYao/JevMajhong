"""Projection arithmetic fixtures only; these are not simulated game outcomes."""
import copy
import unittest

from jevmahjong.training.metric_journal import aggregate_complete_matches
from tools.curriculum_development_projection_v1 import project_step, source_groups


class DevelopmentProjectionTests(unittest.TestCase):
    def fixture(self, step=2500):
        c = dict(curve_seed_groups=2, stage_end_seed_groups=8,
            validation_and_game_steps=[0, 625, 1250, 1875, 2500],
            final_optimizer_steps=2500, curve_seed_start=2026200000,
            opponent_profiles=['efficiency', 'calling'])
        n = source_groups(c, step); start = c['curve_seed_start']
        rows = [dict(seed=seed, profile=profile, seat=seat, completed=True, ledger_passed=True,
            hands=10, rank=1 if seed < start+2 else 4,
            wins=3 if seed < start+2 else 1, dealin_hands=1 if seed < start+2 else 2,
            net_points=6000 if seed < start+2 else -2000)
            for seed in range(start, start+n) for profile in c['opponent_profiles'] for seat in range(4)]
        full = aggregate_complete_matches(rows)
        p = dict(seeds=list(range(start, start+n)), profiles=c['opponent_profiles'], seats=4,
            groups=n, checkpoint=dict(identity_sha256='weights'), policy_identity='policy')
        s = dict(full, checkpoint_sha256='weights', policy_identity='policy')
        if n == 8:s['curve_grid_subset'] = aggregate_complete_matches(rows[:16])
        j = dict(full, checkpoint_sha256='weights', policy_identity='policy',
            grid='stage_end64' if n == 8 else 'curve16')
        return c, step, p, s, rows, j

    def test_endpoint_uses_fixed_seeds_and_keeps_full_result_separate(self):
        args = self.fixture(); before = copy.deepcopy(args)
        curve, full = project_step(*args)
        self.assertEqual(curve['matches'], 16); self.assertEqual(full['matches'], 64)
        self.assertEqual(curve['top2_rate'], 1); self.assertEqual(full['top2_rate'], .25)
        self.assertEqual(curve['net_points'], 6000); self.assertEqual(full['net_points'], 0)
        self.assertEqual(curve['original_grid'], 'stage_end64')
        self.assertTrue(curve['curve16_projected_from_completed64'])
        self.assertEqual(args, before)

    def test_incomplete_endpoint_is_not_a_completed_curve_point(self):
        args = list(self.fixture()); args[4] = args[4][:16]
        with self.assertRaises(ValueError):project_step(*args)

    def test_duplicate_endpoint_cell_is_rejected(self):
        args = self.fixture(); args[4][-1] = dict(args[4][-2])
        with self.assertRaises(ValueError):project_step(*args)

    def test_incorrect_saved_subset_cannot_replace_actual_subset(self):
        args = self.fixture(); args[3]['curve_grid_subset']['net_points'] += 1
        with self.assertRaises(ValueError):project_step(*args)

    def test_full_journal_cannot_be_relabelled_as_curve16(self):
        args = self.fixture(); args[5]['grid'] = 'curve16'
        with self.assertRaises(ValueError):project_step(*args)

    def test_wrong_parameter_identity_is_rejected(self):
        args = self.fixture(); args[5]['checkpoint_sha256'] = 'other'
        with self.assertRaises(ValueError):project_step(*args)

    def test_intermediate16_has_no_extra_endpoint(self):
        args = self.fixture(1250); curve, full = project_step(*args)
        self.assertIsNone(full); self.assertEqual(curve['matches'], 16)
        self.assertFalse(curve['curve16_projected_from_completed64'])
        self.assertEqual(curve['original_grid'], 'curve16')

    def test_reserved_interval_or_unregistered_step_is_rejected(self):
        c = self.fixture()[0]; c['curve_seed_start'] = 2026299995
        with self.assertRaises(ValueError):source_groups(c, 2500)
        with self.assertRaises(ValueError):source_groups(c, 123)


if __name__ == '__main__':unittest.main()
