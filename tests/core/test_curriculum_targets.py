import unittest

import torch

from tools.curriculum_losses import question_loss, weighted_loss


class MaskedCurriculumTargetTests(unittest.TestCase):
    def test_single_label_support_has_zero_loss_and_zero_gradient(self):
        logits = torch.tensor([4., -10., 999.], requires_grad=True)
        loss = question_loss(logits, dict(target=[0., 1., 0.], src='paired_future_masked'), 'cpu')
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_phase_weighted_masked_loss_ignores_unobserved_action(self):
        derivatives = []
        losses = []
        for unknown_logit in (-1000., 1000.):
            logits = torch.tensor([.3, -.5, unknown_logit], requires_grad=True)
            q = dict(target=[.6, .4, 0.], src='paired_future_masked:kan_window')
            loss = weighted_loss(logits, q, 'cpu',
                dict(expert_replay=.75, paired_future_masked=.25), dict(kan_window=2.))
            loss.backward()
            self.assertEqual(logits.grad[-1].item(), 0.)
            self.assertGreater(logits.grad[:2].abs().sum().item(), 0.)
            losses.append(loss.item())
            derivatives.append(logits.grad)
        self.assertEqual(losses[0], losses[1])
        self.assertTrue(torch.equal(derivatives[0], derivatives[1]))
