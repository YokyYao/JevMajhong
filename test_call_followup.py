import unittest
import torch
from scripts.refit_call_query import class_weights, teacher_kl
from scripts.evaluate_call_repair import paired_point_deltas


class CallFollowupTests(unittest.TestCase):
    def test_importance_weighting_restores_natural_priors(self):
        priors = {'chi': .05, 'pon': .10, 'none': .85}
        quotas = {'chi': 1024, 'pon': 1024, 'none': 2048}
        weights = class_weights(priors, quotas)
        for kind in priors:
            self.assertAlmostEqual(weights[kind] * quotas[kind] / sum(quotas.values()), priors[kind])
        self.assertLess(weights['chi'], 1)
        self.assertGreater(weights['none'], 1)
        with self.assertRaises(ValueError): class_weights({'chi': 0, 'pon': .1, 'none': .9}, quotas)
        half = class_weights(priors, quotas, .5)
        for kind in priors:
            self.assertAlmostEqual(half[kind] * quotas[kind] / sum(quotas.values()),
                                   .5 * priors[kind] + .5 * quotas[kind] / sum(quotas.values()))

    def test_teacher_kl_masks_padding_and_does_not_train_teacher(self):
        teacher = torch.tensor([[1., 2., float('-inf')], [0., 2., 1.]], requires_grad=True)
        student = torch.tensor([[2., 1., float('-inf')], [1., 0., 2.]], requires_grad=True)
        kl = teacher_kl(student, teacher)
        self.assertTrue(torch.isfinite(kl).all())
        self.assertTrue((kl >= 0).all())
        kl.sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        torch.testing.assert_close(teacher_kl(teacher.detach(), teacher), torch.zeros(2))

    def test_third_policy_cannot_contaminate_paired_point_difference(self):
        rows = []
        for seed in (1, 2):
            for seat in range(4):
                for policy, score in [('reference', 25000), ('candidate', 25100), ('previous', 99999)]:
                    rows.append({'seed': seed, 'seat': seat, 'policy': policy, 'score': score})
        self.assertEqual(paired_point_deltas(rows, 'candidate', 'reference'), [100, 100])
        self.assertEqual(paired_point_deltas(rows, 'previous', 'reference'), [74999, 74999])
        with self.assertRaises(ValueError): paired_point_deltas(rows + [rows[0]], 'candidate', 'reference')
        with self.assertRaises(ValueError): paired_point_deltas(rows[1:], 'candidate', 'reference')


if __name__ == '__main__': unittest.main()
