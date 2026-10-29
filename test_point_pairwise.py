import unittest
import copy
from types import SimpleNamespace

import torch

from point_pairwise import PointPairwiseTrainer, point_pairwise_loss
from head_rl import mixed_probabilities


class PointPairwiseTests(unittest.TestCase):
    def test_optimizer_updates_only_the_actor(self):
        class Head(torch.nn.Linear):
            def forward(self, decide, options):
                return super().forward(options).squeeze(-1)
        head = Head(2, 1, bias=False)
        reference = copy.deepcopy(head).requires_grad_(False)
        critic = torch.nn.Linear(2, 1).requires_grad_(False)
        torch.nn.init.zeros_(critic.weight); torch.nn.init.zeros_(critic.bias)
        lm = torch.nn.Linear(2, 2).requires_grad_(False)
        config = dict(advantage_estimator='greedy_reference', actor_lr=5e-5,
                      exploration_uniform=.15, reference_kl_beta=1., pairwise_beta=1.)
        policy = SimpleNamespace(head=head, reference=reference, critic=critic,
                                 model=SimpleNamespace(lm=lm), config=config)
        options, decide = torch.eye(2), torch.ones(2)
        p = mixed_probabilities(reference(decide, options), [0, 1], .15).detach()
        row = dict(decide=decide, options=options, eligible=[0, 1], action=1, teacher_action=0,
                   advantage=1., reference=p)
        original = head.weight.detach().clone()
        trainer = PointPairwiseTrainer(policy)
        result = trainer.update([row])
        self.assertFalse(torch.equal(head.weight, original))
        self.assertEqual(len(trainer.optimizer.param_groups), 1)
        self.assertFalse(result['ppo_surrogate_used'])
        self.assertTrue(all(p.grad is None for p in lm.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_extremely_unlikely_beneficial_action_has_nonzero_gradient(self):
        # Uniform exploration can sample this action, while its mixed-logp gradient underflows.
        logits = torch.tensor([0., -200.], requires_grad=True)
        p = mixed_probabilities(logits, [0, 1], .15)
        old_gradient = torch.autograd.grad(-p[1].log(), logits, retain_graph=True)[0]
        self.assertEqual(float(old_gradient[1]), 0.)
        reference = logits.detach().clone().requires_grad_(True)
        loss = point_pairwise_loss(logits, reference, 1, 0, 2.)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[1]), -1.)
        self.assertAlmostEqual(float(logits.grad[0]), 1.)
        self.assertIsNone(reference.grad)

    def test_negative_points_reverse_the_preference_and_keep_score_weight(self):
        for advantage in [-1., -3.]:
            logits = torch.tensor([3., -2.], requires_grad=True)
            loss = point_pairwise_loss(logits, logits.detach(), 1, 0, advantage)
            loss.backward()
            self.assertAlmostEqual(float(logits.grad[1]), abs(advantage)/2)
            self.assertAlmostEqual(float(logits.grad[0]), -abs(advantage)/2)

    def test_zero_delta_has_zero_actor_gradient(self):
        logits = torch.tensor([5., -3.], requires_grad=True)
        loss = point_pairwise_loss(logits, torch.zeros(2), 1, 0, 0.)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros(2)))
        with self.assertRaisesRegex(ValueError, 'unchanged action'):
            point_pairwise_loss(logits, logits.detach(), 0, 0, 1.)


if __name__ == '__main__':
    unittest.main()
