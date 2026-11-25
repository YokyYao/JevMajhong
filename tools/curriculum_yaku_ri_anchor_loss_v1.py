"""Retain actual19-source anchor objectives and unchanged six route losses.

The same student logits supply all terms. No model is loaded here, no parent
is admitted, and no runtime teacher/head is introduced.
"""
from collections import defaultdict
import math

from tools import curriculum_progression_parent_loss_v1 as retained
from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_ri_inverse_repeat_loss_v1 as inverse
from tools import curriculum_yaku_parent_loss_v1 as yaku

METHOD = 'inverse_repeat_retained19_fixed4_RI_anchor_plus_six_yaku_routes_v1'
WEIGHT_METHOD = 'per_original_inverse_repeat_weights_exact_yaku25_objectives_v1'
MASS = yaku.MASS
COMPONENTS = (*retained.COMPONENTS, 'new_route_objective', 'parent_RI_KL',
              'parent_RI_anchor_adjustment', 'base25_objective_without_anchor')


def make_targets(full_train, predictions):
    if (not full_train or {row['questions']['action']['src'] for row in full_train} != set(MASS)
            or len({row['_meta']['id'] for row in full_train}) != len(full_train)):
        raise ValueError('Complete distinct25-source TRAIN corpus required')
    replay = [row for row in full_train if row['questions']['action']['src'] in retained.MASS]
    # Reuse the exact original4244-target numerical cache admission on the
    # complete retained subset. Actual source/checkpoint binding is external.
    return anchor.make_targets(replay, predictions)


def attach_weight(encoded, weight, identifier):
    if (type(weight) not in (int, float) or not math.isfinite(weight) or not 0 < weight <= 16
            or not isinstance(identifier, str) or not identifier):
        raise ValueError('Positive bounded inverse-repeat weight and original ID required')
    for question in encoded['questions']:
        if (question.get(yaku.MARKER) != yaku.METHOD or question['src'] not in MASS
                or question.get(retained.MARKER) != retained.METHOD
                or any(key in question for key in (inverse.FIELD, inverse.IDENTIFIER, inverse.MARKER))):
            raise ValueError('Pristine retained25-source loss annotation required')
        question.update({inverse.FIELD: float(weight), inverse.IDENTIFIER: identifier,
                         inverse.MARKER: WEIGHT_METHOD})
    return encoded


def parts(logits, question, device):
    if (question.get(yaku.MARKER) != yaku.METHOD or question['src'] not in MASS
            or question.get(anchor.MARKER) != anchor.METHOD or anchor.FIELD not in question):
        raise ValueError('Explicit yaku25 and post-encoding parent-anchor annotation required')
    if question['src'] in retained.MASS:
        components = anchor.parts(logits, question, device)
        components['new_route_objective'] = components['objective'] * 0
    else:
        if question['src'] not in yaku.NEW_SOURCES or question[anchor.FIELD] is not None:
            raise ValueError('Six ordinary hard-route losses must have no RI penalty')
        components = yaku.parts(logits, question, device)
        components.update(parent_RI_KL=components['objective'] * 0,
            parent_RI_anchor_adjustment=components['objective'] * 0)
    components['base25_objective_without_anchor'] = (
        components['objective'] - components['parent_RI_anchor_adjustment'])
    if set(components) != set(COMPONENTS):
        raise ValueError('Complete actual25-source objective accounting required')
    return components


def weighted_loss(logits, question, device, observer=None):
    import torch
    if (question.get(inverse.MARKER) != WEIGHT_METHOD
            or type(question.get(inverse.FIELD)) is not float
            or not math.isfinite(question[inverse.FIELD]) or not 0 < question[inverse.FIELD] <= 16
            or not isinstance(question.get(inverse.IDENTIFIER), str) or not question[inverse.IDENTIFIER]):
        raise ValueError('Exact25-source inverse-repeat annotation required')
    components = parts(logits, question, device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'], record_id=question[inverse.IDENTIFIER],
            example_weight=question[inverse.FIELD],
            **{key: float(value.detach()) for key, value in components.items()}))
    return components['objective'] * question[inverse.FIELD]


def summarize(pending):
    if not pending:
        raise ValueError('Actual weighted optimized observations required')
    groups = defaultdict(list)
    for row in pending:
        if (row['source'] not in MASS or not row['record_id']
                or not math.isfinite(row['example_weight']) or not 0 < row['example_weight'] <= 16
                or any(not math.isfinite(row[key]) for key in COMPONENTS)
                or abs(row['objective'] - row['standard_target_ce'] - row['signed_objective_minus_standard_ce']) > 1e-5
                or abs(row['objective'] - row['base25_objective_without_anchor'] - row['parent_RI_anchor_adjustment']) > 1e-5
                or abs(row['parent_RI_anchor_adjustment'] - anchor.STRENGTH * row['parent_RI_KL']) > 1e-5
                or row['parent_RI_KL'] < -1e-5):
            raise ValueError('Complete finite reconciled actual25-source objective components required')
        if row['source'] not in anchor.RI_SOURCES and (row['parent_RI_KL'] != 0
                or row['parent_RI_anchor_adjustment'] != 0):
            raise ValueError('RI penalty must not enter another source')
        if row['source'] not in yaku.NEW_SOURCES and row['new_route_objective'] != 0:
            raise ValueError('Route objective diagnostic must not include retained-source losses')
        groups[row['source']].append(row)
    n = len(pending)
    weighted = lambda key: sum(row['example_weight'] * row[key] for row in pending) / n
    return dict(loss=weighted('objective'), ce=weighted('standard_target_ce'),
        signed_objective_adjustment=weighted('signed_objective_minus_standard_ce'),
        base25_objective_without_anchor=weighted('base25_objective_without_anchor'),
        parent_RI_KL=weighted('parent_RI_KL'),
        parent_RI_anchor_adjustment=weighted('parent_RI_anchor_adjustment'),
        new_route_objective=weighted('new_route_objective'),
        source_counts={source: len(rows) for source, rows in groups.items()},
        source_loss_components={source: {key: sum(row[key] for row in rows)/len(rows)
            for key in COMPONENTS} for source, rows in groups.items()},
        source_weighted_component_contributions={source: {
            key: sum(row['example_weight']*row[key] for row in rows)/n for key in COMPONENTS}
            for source, rows in groups.items()},
        effective_source_mass={source: sum(row['example_weight'] for row in rows)/n
            for source, rows in groups.items()},
        source_example_weight_range={source: [min(row['example_weight'] for row in rows),
            max(row['example_weight'] for row in rows)] for source, rows in groups.items()},
        loss_method=METHOD, parent_anchor_method=anchor.METHOD,
        route_objective_method=yaku.METHOD)
