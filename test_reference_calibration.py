import copy
import unittest

from reference_calibration import calibrated_baselines


class ReferenceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.controls = [{'seed': seed, 'seat': seat, 'reward': 3 * seed + seat}
                         for seed in range(5) for seat in range(4)]
        self.games = [{**g, 'replica': replica, 'reward': -4 + .25 * g['reward'],
                       'trainable_decisions': 2}
                      for g in self.controls for replica in range(2)]

    def test_recovers_known_affine_relationship_outside_each_stream(self):
        baselines, stats = calibrated_baselines(self.games, self.controls)
        for game in self.games:
            self.assertAlmostEqual(baselines[game['seed'], game['seat'], game['replica']], game['reward'])
        for fit in stats['fits']:
            self.assertAlmostEqual(fit['alpha'], .25)
            self.assertAlmostEqual(fit['intercept'], -4)
            self.assertNotIn(fit['excluded_seed'], fit['fit_seeds'])
        self.assertLess(stats['calibrated_residual']['rms'], 1e-10)

    def test_own_stream_returns_never_influence_its_fitted_baseline(self):
        before, _ = calibrated_baselines(self.games, self.controls)
        changed = copy.deepcopy(self.games)
        for game in changed:
            if game['seed'] == 2:
                game['reward'] += 1000 + game['seat']
        after, _ = calibrated_baselines(changed, self.controls)
        for key in before:
            if key[0] == 2:
                self.assertEqual(before[key], after[key])

    def test_constant_control_falls_back_to_other_stream_mean(self):
        controls = [{**g, 'reward': 0} for g in self.controls]
        _, stats = calibrated_baselines(self.games, controls)
        self.assertTrue(all(f['alpha'] == 0 for f in stats['fits']))

    def test_duplicates_and_nonfinite_returns_are_rejected(self):
        for games, controls in [(self.games + [self.games[0]], self.controls),
                                (self.games, self.controls + [self.controls[0]]),
                                ([{**self.games[0], 'reward': float('nan')}] + self.games[1:], self.controls)]:
            with self.assertRaises(ValueError):
                calibrated_baselines(games, controls)


if __name__ == '__main__':
    unittest.main()
