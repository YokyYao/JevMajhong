"""Point-delta-weighted reference-relative action ranking (not PPO)."""
import math

import torch
from torch.nn import functional as F

from head_rl import PPO, mixed_probabilities


def point_pairwise_loss(logits, reference_logits, sampled, teacher, advantage, beta=1.):
    if not math.isfinite(advantage) or not math.isfinite(beta) or beta <= 0:
        raise ValueError('Pairwise score weight and margin scale must be finite')
    if advantage != 0 and sampled == teacher:
        raise ValueError('An unchanged action cannot have a nonzero paired-point label')
    if advantage == 0:
        return logits.sum()*0
    margin = logits[sampled] - logits[teacher]
    reference_margin = (reference_logits[sampled] - reference_logits[teacher]).detach()
    signed_change = (1 if advantage > 0 else -1)*beta*(margin-reference_margin)
    return abs(advantage)*F.softplus(-signed_change)


class PointPairwiseTrainer(PPO):
    """Reuse the optimizer/trust-region helpers, replace the actor objective only."""
    def update(self, batch):
        head, policy, config = self.policy.head, self.policy, self.config
        device = next(head.parameters()).device
        losses, actors, kls = [], [], []
        for row in batch:
            decide, options = row['decide'].to(device), row['options'].to(device)
            logits = head(decide, options)
            with torch.no_grad():
                reference_logits = policy.reference(decide, options)
            eligible = row['eligible']
            actor = point_pairwise_loss(logits, reference_logits,
                                        eligible[row['action']], eligible[row['teacher_action']],
                                        row['advantage'], config['pairwise_beta'])
            p = mixed_probabilities(logits, eligible, config['exploration_uniform'])
            reference = row['reference'].to(device)
            kl = (p*(p.log()-reference.log())).sum()
            losses.append(actor + config['reference_kl_beta']*kl)
            actors.append(actor.detach()); kls.append(kl.detach())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite point-pairwise objective')
        self.optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(head.parameters(), .5)
        if not torch.isfinite(grad):
            raise FloatingPointError('Nonfinite point-pairwise gradient')
        self.optimizer.step()
        if any(p.grad is not None for p in policy.model.lm.parameters()) or any(p.grad is not None for p in policy.critic.parameters()):
            raise ValueError('Point-pairwise training must keep backbone, LoRA and critic frozen')
        return {'loss': float(loss.detach()), 'policy_loss': float(torch.stack(actors).mean()),
                'value_loss': 0., 'reference_kl_batch': float(torch.stack(kls).mean()),
                'head_grad_norm': float(grad), 'critic_grad_norm': 0.,
                'positive_labels': sum(r['advantage'] > 0 for r in batch),
                'negative_labels': sum(r['advantage'] < 0 for r in batch),
                'zero_labels': sum(r['advantage'] == 0 for r in batch),
                'objective': 'point_weighted_reference_relative_pairwise', 'ppo_surrogate_used': False}
