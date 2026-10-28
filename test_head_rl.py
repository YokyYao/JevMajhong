import json
import math
from pathlib import Path
import unittest
import torch

from head_rl import deployment_greedy_index, development_gate, mixed_probabilities, ppo_surrogate, terminal_reward


class HeadRLTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('head-rl-pilot.json').read_text())

    def test_reward_is_final_first_place_only(self):
        self.assertEqual([terminal_reward(i) for i in range(1, 5)], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            terminal_reward(0)

    def test_mask_excludes_highest_illegal_logit_and_exploration_has_support(self):
        logits = torch.tensor([2., 1000., -1000.], requires_grad=True)
        probability = mixed_probabilities(logits, [0, 2], .05)
        self.assertEqual(len(probability), 2)
        self.assertAlmostEqual(float(probability.sum().detach()), 1, places=6)
        self.assertGreaterEqual(float(probability[1].detach()), .025)
        (-probability[0].log()).backward()
        self.assertEqual(float(logits.grad[1]), 0)

    def test_ppo_clips_positive_and_negative_advantages_correctly(self):
        a = ppo_surrogate(torch.tensor(math.log(1.5)), 0., torch.tensor(1.), .1)
        b = ppo_surrogate(torch.tensor(math.log(.5)), 0., torch.tensor(-1.), .1)
        self.assertAlmostEqual(float(a), -1.1, places=5)
        self.assertAlmostEqual(float(b), .9, places=5)

    def test_deployment_preserves_underflow_ties_after_guarding(self):
        logits = torch.tensor([-1., 1000., 0.])
        self.assertEqual(deployment_greedy_index(logits, [0, 2]), 0)
        self.assertEqual(int(mixed_probabilities(logits, [0, 2], 0).argmax()), 1)

    def test_futility_and_regression_gates_are_independent_of_nll(self):
        base = {'first_rate': .4, 'mean_net_points': 3000, 'fourth_rate': .15}
        self.assertEqual(development_gate(1, base, base, self.config)[0], 'continue')
        self.assertEqual(development_gate(2, base, base, self.config)[0], 'stop_no_early_progress')
        candidate = dict(base, first_rate=.46875)
        self.assertEqual(development_gate(1, candidate, base, self.config)[0], 'target_reached')
        candidate = dict(base, first_rate=.5, mean_net_points=900)
        self.assertEqual(development_gate(1, candidate, base, self.config)[0], 'stop_regression')

    def test_seed_sets_are_disjoint_and_update_budget_is_short(self):
        c = self.config
        train = set(range(c['training_seed_start'], c['training_seed_start']+c['max_stages']*c['training_seeds_per_stage']))
        dev = set(range(c['development_seed_start'], c['development_seed_start']+c['development_seeds']))
        confirm = set(range(c['confirmation_seed_start'], c['confirmation_seed_start']+c['confirmation_seeds']))
        self.assertFalse(train & dev or train & confirm or dev & confirm)
        self.assertEqual(c['max_stages']*c['updates_per_stage'], 64)


if __name__ == '__main__':
    unittest.main()
