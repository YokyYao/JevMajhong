import unittest
import numpy as np

from jevmahjong.data.bootstrap_targets import (bootstrap_probabilities,
    revised_reward_bootstrap_probabilities, validate_reward_revision)


class PairedBootstrapTargetsTests(unittest.TestCase):
    def test_constant_futures_preserve_distribution_and_safe_tradeoff(self):
        # The safer action can win less: preserve the declared reward tradeoff.
        values = np.tile([[0., 0., 0., 0.], [.15, -.05, 0., -.1]], (16, 1, 1))
        p = bootstrap_probabilities(values, seed=7)
        np.testing.assert_allclose(p, np.tile([1/(1+np.exp(3)), 1/(1+np.exp(-3))], (256, 1)))

    def test_common_futures_keep_identical_actions_tied(self):
        # Independent action resampling would spuriously break this tie.
        rng = np.random.default_rng(12)
        values = rng.normal(size=(16, 1, 4)).repeat(3, axis=1)
        p = bootstrap_probabilities(values, seed=9)
        np.testing.assert_allclose(p, np.full((256, 3), 1/3))

    def test_uncertain_winner_is_less_certain_than_mean_only_target(self):
        values = np.zeros((4, 2, 4))
        values[:, 1, 0] = [1, 1, 1, -1]
        values[:, 1, 1] = [.5, .5, .5, -.5]
        p = bootstrap_probabilities(values, seed=11, replicates=4096).mean(axis=0)
        self.assertGreater(p[1], .65)
        self.assertLess(p[1], .95)
        self.assertAlmostEqual(p.sum(), 1.)

    def test_action_permutation_and_fixed_seed(self):
        values = np.random.default_rng(3).normal(size=(32, 4, 4))
        original = bootstrap_probabilities(values, seed=51)
        reordered = bootstrap_probabilities(values[:, [3, 1, 0, 2]], seed=51)
        np.testing.assert_allclose(original[:, [3, 1, 0, 2]], reordered, atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(original, bootstrap_probabilities(values, seed=51))

    def test_rejects_incomplete_or_nonfinite_future_arrays(self):
        for values in [np.zeros((1, 2, 4)), np.zeros((16, 2, 3)), np.full((2, 2, 4), np.nan)]:
            with self.assertRaises(ValueError):
                bootstrap_probabilities(values, seed=1)
        for n in [0, True, 1.5]:
            with self.assertRaises(ValueError):
                bootstrap_probabilities(np.zeros((2, 2, 4)), seed=1, replicates=n)


class RewardRevisionTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(mode='own_win_first_v1', win_weight=1., fast_weight=.25,
                        dealin_penalty=2., net_weight=0., point_scale=8000., speed_gamma=.98)
        self.new = dict(self.old, dealin_penalty=1.)

    def test_lower_risk_cost_changes_win_risk_choice_without_mutating_outcomes(self):
        values = np.tile([[0., 0., 0., 0.], [-.1, .2, 0., .15]], (16, 1, 1))
        saved = values.copy()
        before = bootstrap_probabilities(values, seed=1).mean(axis=0)
        after = revised_reward_bootstrap_probabilities(values, parent_reward=self.old,
                                                      reward=self.new, seed=1).mean(axis=0)
        self.assertLess(before[1], .2)
        self.assertGreater(after[1], .7)
        np.testing.assert_array_equal(values, saved)

    def test_rejects_changed_speed_gamma_with_unrecomputed_speed_credits(self):
        with self.assertRaises(ValueError):
            validate_reward_revision(self.old, dict(self.new, speed_gamma=.95))

    def test_rejects_changed_win_weight_and_reward_mode(self):
        for reward in [dict(self.new, win_weight=2.), dict(self.new, mode='risk_adjusted_fast_win_v1')]:
            with self.assertRaises(ValueError):
                validate_reward_revision(self.old, reward)

    def test_rejects_missing_or_undisclosed_reward_fields(self):
        for reward in [self.old, dict(self.new, extra_weight=1.)]:
            with self.assertRaises(ValueError):
                validate_reward_revision(self.old, reward)

    def test_rejects_inconsistent_parent_utility_before_resampling(self):
        values = np.tile([[0., 0., 0., 0.], [1., .2, 0., .15]], (16, 1, 1))
        with self.assertRaises(ValueError):
            revised_reward_bootstrap_probabilities(values, parent_reward=self.old, reward=self.new, seed=1)

    def test_rejects_nonfinite_or_negative_new_risk_cost(self):
        for penalty in [float('nan'), -1.]:
            with self.assertRaises(ValueError):
                validate_reward_revision(self.old, dict(self.old, dealin_penalty=penalty))


if __name__ == '__main__':
    unittest.main()
