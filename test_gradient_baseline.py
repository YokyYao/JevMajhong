import copy
import unittest

import torch
from gradient_baseline import score_weighted_baselines, trajectory_score_norm


class BernoulliHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, decide, options):
        return torch.stack([self.theta * 0, self.theta])


class GradientBaselineTests(unittest.TestCase):
    def row(self, action):
        return {'decide': torch.zeros(1), 'options': torch.zeros(2, 1),
                'eligible': [0, 1], 'action': action}

    def test_trajectory_score_sums_before_squaring(self):
        head = BernoulliHead()
        self.assertAlmostEqual(trajectory_score_norm(head, [self.row(1)], 0), .25)
        self.assertAlmostEqual(trajectory_score_norm(head, [self.row(1), self.row(1)], 0), 1)
        self.assertAlmostEqual(trajectory_score_norm(head, [self.row(0), self.row(1)], 0), 0)
        self.assertEqual(float(head.theta.detach()), 0)
        self.assertIsNone(head.theta.grad)

    def test_uniform_mixture_derivative_is_included(self):
        self.assertAlmostEqual(trajectory_score_norm(BernoulliHead(), [self.row(1)], .2), .16, places=6)

    def fixture(self):
        controls = [{'seed': s, 'seat': p, 'reward': 3*s+p} for s in range(5) for p in range(4)]
        games = [{**c, 'replica': 0, 'reward': -4 + .25*c['reward']} for c in controls]
        weights = {(g['seed'], g['seat'], 0): 1. for g in games}
        return games, controls, weights

    def test_zero_weight_outlier_is_ignored_and_current_seed_is_excluded(self):
        games, controls, weights = self.fixture()
        for g in games:
            if g['seed'] == 1:
                g['reward'] = 1000
                weights[g['seed'], g['seat'], 0] = 0
        baseline, _ = score_weighted_baselines(games, controls, weights)
        for g in games:
            if g['seed'] == 2:
                self.assertAlmostEqual(baseline[2, g['seat'], 0], -4 + .25*(6+g['seat']))
        altered = copy.deepcopy(games)
        for g in altered:
            if g['seed'] == 2:
                g['reward'] += 10000
        again, _ = score_weighted_baselines(altered, controls, weights)
        for k in baseline:
            if k[0] == 2:
                self.assertEqual(baseline[k], again[k])


if __name__ == '__main__':
    unittest.main()
