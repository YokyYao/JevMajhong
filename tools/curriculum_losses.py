"""Portable curriculum loss arithmetic, compatible with the pinned research code."""
import torch

PHASE_PREFIX = 'paired_future_masked:'


def question_loss(logits, question, device):
    if question.get('target') is not None:
        target = torch.tensor(question['target'], dtype=logits.dtype, device=device)
        if target.shape != logits.shape or not torch.isfinite(target).all() or (target < 0).any() or target.sum() <= 0:
            raise ValueError('Valid complete target vector required')
        if question.get('src') == 'paired_future_masked':
            mask = target > 0
            target, logits = target[mask], logits[mask]
        target = target/target.sum()
        return -(target*logits.log_softmax(-1)).sum()
    label = torch.tensor([question['label']], device=device)
    return torch.nn.functional.cross_entropy(logits[None], label)


def weighted_loss(logits, question, device, weights, phase_weights=None):
    src = question.get('src', '')
    if isinstance(src, str) and src.startswith(PHASE_PREFIX):
        phase = src[len(PHASE_PREFIX):]
        if phase_weights is None or phase not in phase_weights:
            raise ValueError('Unexpected loss-only source phase')
        return phase_weights[phase]*question_loss(logits, dict(question, src='paired_future_masked'), device)
    label = 'paired_future_masked' if src == 'paired_future_masked' else 'expert_replay'
    if phase_weights is not None and label == 'paired_future_masked':
        raise ValueError('Missing MC phase loss tag')
    return weights[label]*question_loss(logits, question, device)
