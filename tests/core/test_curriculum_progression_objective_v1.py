import math
import unittest

import torch

from tools.curriculum_progression_objective_v1 import FIELD, MARKER, METHOD, parts


class ProgressionObjectiveTests(unittest.TestCase):
    def question(self, label=0):
        return dict(src='curriculum2:basic:teacher:riichi_yes' if label == 0 else 'curriculum2:basic:teacher:riichi_no',
            label=label, **{FIELD: [0], MARKER: METHOD})

    def test_lambda1_is_exact_ordinary_CE_but_lambda2_adds_real_class_gradient(self):
        for label in (0, 1, 2):
            z = torch.tensor([.2, -.3, .8], dtype=torch.float64, requires_grad=True)
            q = self.question(label); p = parts(z, q, 'cpu', class_weight=1.)
            self.assertAlmostEqual(float(p['objective'].detach()), float(p['standard_target_ce'].detach()), places=12)
            actual = parts(z, q, 'cpu')
            self.assertAlmostEqual(float((actual['objective']-p['objective']).detach()), float(p['RI_group_ce'].detach()), places=12)
            grad1 = torch.autograd.grad(p['objective'], z, retain_graph=True)[0]
            grad2 = torch.autograd.grad(actual['objective'], z)[0]
            self.assertGreater(float((grad2-grad1).abs().max()), .05)

    def test_balanced_binary_class_has_zero_RI_gradient_at_full_probability_half(self):
        z = torch.tensor([math.log(2), 0., 0.], dtype=torch.float64, requires_grad=True)
        yes = parts(z, self.question(0), 'cpu')['RI_group_ce']
        no = parts(z, self.question(1), 'cpu')['RI_group_ce']
        gradient = torch.autograd.grad((yes+no)/2, z)[0]
        self.assertAlmostEqual(float(gradient[0]), 0., places=12)
        self.assertAlmostEqual(float(z.softmax(0)[0].detach()), .5, places=12)

    def test_MC_keeps_unobserved_actions_masked_with_exact_zero_gradient(self):
        z = torch.tensor([0., 1., 90.], dtype=torch.float64, requires_grad=True)
        q = dict(src='curriculum3:mc:kan_window', target=[.4, .6, 0.], label=1, **{FIELD: [], MARKER: METHOD})
        result = parts(z, q, 'cpu'); result['objective'].backward()
        self.assertTrue(torch.isfinite(result['objective']))
        self.assertEqual(float(z.grad[2]), 0.)
        self.assertEqual(float(result['extra_RI_group_ce'].detach()), 0.)


if __name__ == '__main__': unittest.main()
