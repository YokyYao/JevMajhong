"""Prospective 3A objective retaining the exact qualified calls13 losses.

No registration, parent admission or model fit is performed here. The
non-replay progression targets and the nineteen source masses are unchanged.
"""
from collections import defaultdict
import math

from tools import curriculum_calls_classroute_v1 as parent
from tools import curriculum_progression_objective_v1 as progression

METHOD = 'progression19_exact_calls13_replay_objective_retained_v1'
MARKER = '_progression_parent_loss_v1'
MASS = progression.MASS
COMPONENTS = ('objective', 'standard_target_ce', 'parent_RI_max_ce',
              'RI_group_ce_diagnostic', 'conditional_discard_ce',
              'ordinary_hard_ce', 'masked_MC_ce', 'signed_objective_minus_standard_ce')


def annotate(encoded, request):
    if any(MARKER in q for q in encoded['questions']):
        raise ValueError('Never annotate the same question twice')
    progression.annotate(encoded, request)
    for q in encoded['questions']:
        if q['src'].startswith('curriculum2:'):
            parent.annotate({'questions': [q]}, request)
        q[MARKER] = METHOD
    return encoded


def parts(logits, question, device):
    if question.get(MARKER) != METHOD or question['src'] not in MASS:
        raise ValueError('Explicit nineteen-source parent-retention annotation required')
    if question['src'].startswith('curriculum2:'):
        old = parent.parts(logits, question, device)
        objective, ce = old['objective'], old['standard_action_ce']
        zero = objective*0
        result = dict(objective=objective, standard_target_ce=ce,
            parent_RI_max_ce=old['riichi_max_class_ce'],
            RI_group_ce_diagnostic=old['riichi_group_class_ce'],
            conditional_discard_ce=old['conditional_discard_ce'],
            ordinary_hard_ce=old['plain_action_ce']+old['advanced_action_ce']+old['post_reach_action_ce'],
            masked_MC_ce=zero, signed_objective_minus_standard_ce=objective-ce)
    else:
        old = progression.parts(logits, question, device)
        objective, ce = old['objective'], old['standard_target_ce']
        zero = objective*0
        result = dict(objective=objective, standard_target_ce=ce,
            parent_RI_max_ce=zero, RI_group_ce_diagnostic=old['RI_group_ce'],
            conditional_discard_ce=old['conditional_discard_ce'],
            ordinary_hard_ce=old['ordinary_hard_ce'], masked_MC_ce=old['masked_MC_ce'],
            signed_objective_minus_standard_ce=objective-ce)
    if set(result) != set(COMPONENTS):
        raise ValueError('Complete explicit actual objective decomposition required')
    return result


def weighted_loss(logits, question, device, weights, observer=None):
    import torch
    if set(weights) != set(MASS) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('All nineteen positive registered source weights required')
    result = parts(logits, question, device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'], **{k: float(v.detach()) for k, v in result.items()}))
    return result['objective']*weights[question['src']]


def summarize(pending, weights):
    if (not pending or set(weights) != set(MASS)
            or any(not math.isfinite(v) or v <= 0 for v in weights.values())):
        raise ValueError('Actual optimizer examples and all positive source weights required')
    groups = defaultdict(list)
    for row in pending:
        if row['source'] not in MASS or any(not math.isfinite(row[k]) for k in COMPONENTS):
            raise ValueError('Actual complete finite example components required')
        if abs(row['objective']-row['standard_target_ce']-row['signed_objective_minus_standard_ce']) > 1e-5:
            raise ValueError('Actual objective minus CE diagnostic differs')
        groups[row['source']].append(row)
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in groups.items()}
    counts = {s: len(rows) for s, rows in groups.items()}; n = sum(counts.values())
    def weighted(k):
        return sum(counts[s]*weights[s]*means[s][k] for s in means)/n
    return dict(source_loss_components=means, source_counts=counts,
        source_objective={s: r['objective'] for s, r in means.items()},
        source_ce={s: r['standard_target_ce'] for s, r in means.items()},
        loss=weighted('objective'), ce=weighted('standard_target_ce'),
        signed_objective_adjustment=weighted('signed_objective_minus_standard_ce'), loss_method=METHOD)
