"""Keep raw labels immutable while matching upstream's exact normalization.

The failed v1 full-data preflight is preserved. v2 changes only annotation
alignment for terminal targets; all objective formulas remain identical.
"""
import copy
import math

from tools import curriculum_placement_parent_loss_v1 as original

METHOD = 'placement57_retained25_replay_upstream_normalized_terminal_alignment_v2'
MARKER = '_placement_parent_loss_v2'
COMPONENTS = original.COMPONENTS
NEW_SOURCES = original.NEW_SOURCES
replay = original.replay
terminal = original.terminal
masses = original.masses
lift_terminal = original.lift_terminal


def annotate(encoded, request):
    if any(MARKER in q for q in encoded['questions']): raise ValueError('Never annotate a question twice')
    derived = dict(request, questions=dict(request['questions']))
    for q in encoded['questions']:
        if q['src'] not in NEW_SOURCES: continue
        raw = request['questions'][q['qid']]; target = raw.get('target'); keys = q['keys']
        if (not isinstance(target, dict) or set(target) != set(keys)
                or any(not isinstance(target[k], (int, float)) or not math.isfinite(target[k]) or target[k] < 0 for k in keys)):
            raise ValueError('Original finite aligned terminal soft labels required')
        # Exactly the pinned materializer's conversion, sum order and division.
        weights = [float(target[k]) for k in keys]; total = sum(weights)
        if total <= 0 or abs(total-1.) > 1e-10: raise ValueError('Original normalized terminal target required')
        normalized = [v/total for v in weights]
        if q.get('target') != normalized:
            raise ValueError('Materialized target differs from exact upstream normalization')
        derived['questions'][q['qid']] = dict(raw, target=dict(zip(keys, normalized)))
    original.annotate(encoded, derived)
    for q in encoded['questions']: q[MARKER] = METHOD
    return encoded


def parts(logits, question, device):
    if question.get(MARKER) != METHOD: raise ValueError('Explicit normalized-alignment v2 annotation required')
    return original.parts(logits, question, device)


def weighted_loss(logits, question, device, weights, observer=None):
    if question.get(MARKER) != METHOD: raise ValueError('Explicit normalized-alignment v2 annotation required')
    return original.weighted_loss(logits, question, device, weights, observer)


def summarize(pending, weights):
    return dict(original.summarize(pending, weights), loss_method=METHOD)


def materialization_unchanged(before, after):
    restored = copy.deepcopy(after)
    for q in restored['questions']:
        if q.pop(MARKER, None) != METHOD: raise ValueError('Explicit v2 annotation required')
    original.materialization_unchanged(before, restored)
