import unittest
import numpy as np

from jevmahjong.data.bootstrap_targets import bootstrap_probabilities


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


if __name__ == '__main__':
    unittest.main()
