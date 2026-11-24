"""Prospective 3B objective retaining the actual parent-loss 3A replay formula.

This module cannot admit a parent, register a student or start training.
The six new route losses and the original25-source masses remain unchanged.
"""
from collections import defaultdict
import copy
import math

from tools import curriculum_progression_parent_loss_v1 as parent
from tools import curriculum_yaku_objective_v1 as previous
from tools import curriculum_calls_riichi_retention_v1 as calls_retention
from tools.curriculum_riichi_class_loss import FIELD as BASIC_FIELD

METHOD = 'yaku25_exact_retained19_replay_and_unchanged_six_route_losses_v1'
MARKER = '_yaku_parent_loss_v1'
MASS = previous.MASS
NEW_SOURCES = previous.NEW_SOURCES
ROUTES = previous.ROUTES
RECORDS = previous.RECORDS
STEPS_PER_EPOCH = previous.STEPS_PER_EPOCH
FINAL_STEP = previous.FINAL_STEP
weights = previous.weights
epoch_order = previous.epoch_order
COMPONENTS = (*parent.COMPONENTS, 'new_route_objective')
ANNOTATIONS = {MARKER, parent.MARKER, parent.progression.MARKER,
               parent.progression.FIELD, parent.parent.MARKER,
               calls_retention.MARKER, BASIC_FIELD}


def old_source(source):
    if source in NEW_SOURCES: return 'curriculum3:teacher:ordinary_completion'
    if source not in parent.MASS: raise ValueError('Unknown retained-replay/route source')
    return source


def annotate(encoded, request):
    originals = {}; translated = dict(request, questions={})
    for qid, raw in request['questions'].items():
        source = raw['src']; originals[qid] = source
        if source not in MASS or (source in NEW_SOURCES and raw.get('target') is not None):
            raise ValueError('Original replay and six unchanged observed hard-route sources required')
        translated['questions'][qid] = dict(raw, src=old_source(source))
    for q in encoded['questions']:
        if (q.get(MARKER) is not None or parent.MARKER in q
                or q['src'] != originals[q['qid']]):
            raise ValueError('Pristine original source-aligned materialization required')
        q['src'] = old_source(q['src'])
    result = parent.annotate(encoded, translated)
    for q in result['questions']:
        q['src'] = originals[q['qid']]; q[MARKER] = METHOD
    return result


def parts(logits, question, device):
    if question.get(MARKER) != METHOD or question['src'] not in MASS:
        raise ValueError('Explicit retained-parent25-source annotation required')
    result = parent.parts(logits, dict(question, src=old_source(question['src'])), device)
    return dict(result, new_route_objective=(result['objective']
        if question['src'] in NEW_SOURCES else result['objective']*0))


def weighted_loss(logits, question, device, source_weights, observer=None):
    import torch
    if set(source_weights) != set(MASS) or any(not math.isfinite(v) or v <= 0 for v in source_weights.values()):
        raise ValueError('All25 positive source weights required')
    actual = parts(logits, question, device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'], **{k: float(v.detach()) for k, v in actual.items()}))
    return actual['objective']*source_weights[question['src']]


def summarize(pending, source_weights):
    if (not pending or set(source_weights) != set(MASS)
            or any(not math.isfinite(v) or v <= 0 for v in source_weights.values())):
        raise ValueError('Actual optimizer observations and all25 positive source weights required')
    groups = defaultdict(list)
    for row in pending:
        if row['source'] not in MASS or any(not math.isfinite(row[k]) for k in COMPONENTS):
            raise ValueError('Complete finite actual source components required')
        if abs(row['objective']-row['standard_target_ce']-row['signed_objective_minus_standard_ce']) > 1e-5:
            raise ValueError('Actual signed objective accounting differs')
        groups[row['source']].append(row)
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS}
             for s, rows in groups.items()}
    counts = {s: len(rows) for s, rows in groups.items()}; n = sum(counts.values())
    average = lambda k: sum(counts[s]*source_weights[s]*means[s][k] for s in means)/n
    return dict(source_loss_components=means, source_counts=counts,
        source_objective={s: m['objective'] for s, m in means.items()},
        source_ce={s: m['standard_target_ce'] for s, m in means.items()},
        loss=average('objective'), ce=average('standard_target_ce'),
        signed_objective_adjustment=average('signed_objective_minus_standard_ce'),
        new_route_objective=average('new_route_objective'), loss_method=METHOD)


def materialization_unchanged(original, annotated):
    restored = copy.deepcopy(annotated)
    if len(original['questions']) != len(restored['questions']):
        raise ValueError('Question membership changed')
    for before, after in zip(original['questions'], restored['questions']):
        added = set(after)-set(before)
        if not added <= ANNOTATIONS: raise ValueError('Unknown annotation field')
        for name in added: after.pop(name)
    if restored != original:
        raise ValueError('Visible input, labels, target mask, source or candidate order changed')
