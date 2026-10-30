"""Contextual option-value projection exported as a native query-bias update."""
import copy
import math

import torch


def value_features(head,decide,options):
    if head.residual is not None or head.temperature!=1.:
        raise ValueError('Plain temperature1 pointer required')
    return head.k(options)*head.scale


def folded_bias(head,coefficient,strength):
    if strength<0 or not math.isfinite(strength):raise ValueError('Invalid projection strength')
    if head.residual is not None or head.temperature!=1.:raise ValueError('Unsupported pointer format')
    result=copy.deepcopy(head)
    shift=coefficient.to(device=result.q.bias.device,dtype=result.q.bias.dtype)
    if shift.shape!=result.q.bias.shape or not torch.isfinite(shift).all():raise ValueError('Invalid value projection')
    with torch.no_grad():result.q.bias.add_(shift,alpha=strength)
    return result.eval()
