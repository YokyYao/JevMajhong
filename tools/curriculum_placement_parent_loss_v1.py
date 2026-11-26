"""Retain exact qualified3B replay losses in prospective placement training.

The32 new terminal objectives, source masses and repeat recipe are unchanged.
This module cannot admit a parent, complete a dataset or start a model fit.
"""
from collections import defaultdict
import copy
import math

from tools import curriculum_yaku_parent_loss_v1 as replay
from tools import curriculum_placement_objective_v1 as terminal
from tools.curriculum_placement_mix_v1 import masses, NEW_SOURCES

METHOD = 'placement57_exact_retained25_replay_and_unchanged32_terminal_losses_v1'
MARKER = '_placement_parent_loss_v1'
COMPONENTS = (*replay.COMPONENTS, 'terminal_soft_objective')
ANNOTATIONS = replay.ANNOTATIONS | {MARKER, terminal.MARKER, terminal.FIELD}


def annotate(encoded, request):
    for q in encoded['questions']:
        if MARKER in q or q['src'] not in masses():
            raise ValueError('Pristine exact57-source placement question required')
        module = replay if q['src'] in replay.MASS else terminal
        raw = request['questions'][q['qid']]
        module.annotate(dict(encoded, questions=[q]), dict(request, questions={q['qid']: raw}))
        q[MARKER] = METHOD
    return encoded


def lift_terminal(values):
    objective = values['objective']; zero = objective*0
    return dict(objective=objective, standard_target_ce=values['standard_target_ce'],
        parent_RI_max_ce=zero, RI_group_ce_diagnostic=values['RI_group_ce'],
        conditional_discard_ce=values['conditional_discard_ce'],
        ordinary_hard_ce=values['ordinary_hard_ce'], masked_MC_ce=values['masked_MC_ce'],
        signed_objective_minus_standard_ce=objective-values['standard_target_ce'],
        new_route_objective=zero, terminal_soft_objective=objective)


def parts(logits, question, device):
    if question.get(MARKER) != METHOD or question['src'] not in masses():
        raise ValueError('Explicit exact57-source retained-parent annotation required')
    if question['src'] in replay.MASS:
        old = replay.parts(logits, question, device)
        return dict(old, terminal_soft_objective=old['objective']*0)
    return lift_terminal(terminal.parts(logits, question, device))


def weighted_loss(logits, question, device, weights, observer=None):
    import torch
    if set(weights) != set(masses()) or any(not math.isfinite(v) or not 0 < v <= 16 for v in weights.values()):
        raise ValueError('All57 positive source weights within the unchanged scalar cap16 required')
    values = parts(logits, question, device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'], **{k: float(v.detach()) for k, v in values.items()}))
    return values['objective']*weights[question['src']]


def summarize(pending, weights):
    if (not pending or set(weights) != set(masses())
            or any(not math.isfinite(v) or not 0 < v <= 16 for v in weights.values())):
        raise ValueError('Actual observations and complete bounded57-source weights required')
    grouped = defaultdict(list)
    for row in pending:
        if row['source'] not in weights or any(not math.isfinite(row[k]) for k in COMPONENTS):
            raise ValueError('Complete finite actual loss observations required')
        if abs(row['objective']-row['standard_target_ce']-row['signed_objective_minus_standard_ce']) > 1e-5:
            raise ValueError('Signed objective accounting differs')
        expected = row['objective'] if row['source'] in NEW_SOURCES else 0.
        if abs(row['terminal_soft_objective']-expected) > 1e-5:
            raise ValueError('Terminal objective diagnostic differs')
        grouped[row['source']].append(row)
    counts = {s: len(rows) for s, rows in grouped.items()}
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in grouped.items()}
    average = lambda k: sum(counts[s]*weights[s]*means[s][k] for s in means)/len(pending)
    return dict(source_loss_components=means, source_counts=counts,
        source_objective={s: v['objective'] for s, v in means.items()},
        source_ce={s: v['standard_target_ce'] for s, v in means.items()},
        loss=average('objective'), ce=average('standard_target_ce'),
        signed_objective_adjustment=average('signed_objective_minus_standard_ce'),
        new_route_objective=average('new_route_objective'),
        terminal_soft_objective=average('terminal_soft_objective'), loss_method=METHOD,
        sparse_sources_per_update_expected=True)


def materialization_unchanged(original, annotated):
    restored = copy.deepcopy(annotated)
    if len(original['questions']) != len(restored['questions']): raise ValueError('Question membership changed')
    for before, after in zip(original['questions'], restored['questions']):
        added = set(after)-set(before)
        if not added <= ANNOTATIONS: raise ValueError('Unknown annotation field')
        for key in added: after.pop(key)
    if restored != original: raise ValueError('Visible input, source, labels, target mask or menu order changed')
