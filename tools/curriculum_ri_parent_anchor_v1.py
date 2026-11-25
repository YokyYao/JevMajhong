"""Prospective TRAIN-parent RI anchor; does not register or admit a new fit."""
import math

from tools import curriculum_progression_parent_loss_v1 as retained
from tools.curriculum_calls_repair import prediction_index
from tools.audit_curriculum_basic_teacher import fingerprint

METHOD = 'retained19_plus_fixed4_parent_Bernoulli_RI_KL_v1'
FIELD = '_parent_RI_anchor_v1'
MARKER = '_parent_RI_anchor_method_v1'
STRENGTH = 4.
RI_SOURCES = frozenset(s for s in retained.MASS if s.startswith('curriculum2:basic:')
                      and s.endswith((':riichi_yes', ':riichi_no')))


def make_targets(full_train, predictions):
    """Require complete original TRAIN membership; caller binds its provenance.

    Predictions must be all and only the four RI populations, from the frozen
    initialization. Native source admission/checkpoint identity is external;
    this numerical helper cannot establish those facts from probabilities.
    """
    if (not full_train or {r['questions']['action']['src'] for r in full_train} != set(retained.MASS)
            or len({r['_meta']['id'] for r in full_train}) != len(full_train)):
        raise ValueError('Full distinct nineteen-source TRAIN population required')
    selected = [r for r in full_train if r['questions']['action']['src'] in RI_SOURCES]
    if {r['questions']['action']['src'] for r in selected} != RI_SOURCES:
        raise ValueError('All four original RI classes required')
    saved = prediction_index(selected, predictions); targets = {}
    for row in selected:
        q = row['questions']['action']; p = saved[row['_meta']['id']]
        reach = [k for k in p['keys'] if q['criteria'][k]['type'] == 'reach']
        if (len(reach) != 1 or {v['type'] for v in q['criteria'].values()} != {'reach', 'dahai'}
                or (q['label'] == reach[0]) != q['src'].endswith(':riichi_yes')):
            raise ValueError('Exact one-reach menu and source-label class required')
        probability = p['p'][p['keys'].index(reach[0])]/sum(p['p'])
        if not 0 < probability < 1:
            raise ValueError('Finite interior parent class probability required')
        targets[row['_meta']['id']] = dict(source=q['src'], label=q['label'],
            keys=frozenset(p['keys']), reach_key=reach[0], probability=probability,
            visible_sha256=fingerprint(row))
    return targets


def annotate(encoded, request, targets):
    if any(MARKER in q or FIELD in q for q in encoded['questions']):
        raise ValueError('Never attach an anchor twice')
    retained.annotate(encoded, request)
    for q in encoded['questions']:
        if q['src'] in RI_SOURCES:
            raw = request['questions'][q['qid']]; identifier = request['_meta']['id']
            target = targets.get(identifier)
            if (target is None or target['source'] != q['src'] or target['label'] != raw['label']
                    or target['keys'] != frozenset(q['keys']) or target['visible_sha256'] != fingerprint(request)):
                raise ValueError('Exact TRAIN identity, input, label and complete parent menu required')
            q[FIELD] = dict(probability=target['probability'], reach=q['keys'].index(target['reach_key']))
        else:
            q[FIELD] = None
        q[MARKER] = METHOD
    return encoded


def parts(logits, question, device):
    import torch
    from torch.nn import functional as F
    if question.get(MARKER) != METHOD or FIELD not in question:
        raise ValueError('Explicit prospective parent-anchor annotation required')
    base = retained.parts(logits, question, device)
    anchor = question[FIELD]; zero = base['objective']*0
    if question['src'] not in RI_SOURCES:
        if anchor is not None:
            raise ValueError('Do not anchor ordinary/call/MC decisions')
        return dict(base, parent_RI_KL=zero, parent_RI_anchor_adjustment=zero)
    if not isinstance(anchor, dict) or set(anchor) != {'probability', 'reach'}:
        raise ValueError('Exact one-class anchor required')
    probability = anchor['probability']; reach = anchor['reach']
    if (not isinstance(probability, (int, float)) or not math.isfinite(probability)
            or not 0 < probability < 1 or type(reach) is not int or not 0 <= reach < len(logits)):
        raise ValueError('Finite interior parent probability and valid reach index required')
    work = logits if logits.dtype == torch.float64 else logits.float()
    discard = [i for i in range(len(work)) if i != reach]
    if not discard:
        raise ValueError('At least one discard alternative required')
    margin = work[reach] - torch.logsumexp(work[discard], 0)
    target = torch.tensor(probability, dtype=work.dtype, device=device)
    kl = (F.binary_cross_entropy_with_logits(margin, target)
          + probability*math.log(probability)+(1-probability)*math.log1p(-probability))
    adjustment = STRENGTH*kl
    return dict(base, objective=base['objective']+adjustment,
        signed_objective_minus_standard_ce=base['signed_objective_minus_standard_ce']+adjustment,
        parent_RI_KL=kl, parent_RI_anchor_adjustment=adjustment)
