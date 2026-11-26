"""Prospective25-source yaku course using the verified progression loss.

Alias only private source metadata for six new hard-label pools. Model-visible
records, action identities, masks and the class2 RI formula stay unchanged.
"""
from collections import Counter
import copy
import math

from tools import curriculum_progression_objective_v1 as previous
from tools.curriculum_course_order import course_order

ROUTES = ('kokushi', 'chuuren', 'suuankou', 'daisangen', 'chiitoitsu', 'unselected_random_initial_hand_control')
NEW_SOURCES = tuple('curriculum3B:teacher:'+r for r in ROUTES)
MASS = {s: .8*v for s, v in previous.MASS.items()}
MASS.update({s: .2/len(NEW_SOURCES) for s in NEW_SOURCES})
METHOD = 'yaku25_replay_full_RI_group_class2_and_masked_MC_v1'
MARKER = '_yaku25_annotation_v1'
COMPONENTS = previous.COMPONENTS
RECORDS = 91247
STEPS_PER_EPOCH = 1250
FINAL_STEP = 2500


def weights(counts):
    if (set(counts) != set(MASS) or any(type(v) is not int or v <= 0 for v in counts.values())
            or sum(counts.values()) != RECORDS or abs(sum(MASS.values())-1) > 1e-12):
        raise ValueError('Exact25-source TRAIN counts and normalized prospective masses required')
    result = {s: MASS[s]*RECORDS/n for s, n in counts.items()}
    if max(result.values()) > 16:
        raise ValueError('Scalar example weight cap16 exceeded')
    return result


def old_source(source):
    if source in NEW_SOURCES:
        return 'curriculum3:teacher:ordinary_completion'
    if source not in previous.MASS:
        raise ValueError('Unknown yaku/replay loss source')
    return source


def annotate(encoded, request):
    originals = {}
    translated = dict(request, questions={})
    for qid, raw in request['questions'].items():
        src = raw['src']; originals[qid] = src
        if src not in MASS or (src in NEW_SOURCES and raw.get('target') is not None):
            raise ValueError('Six new sources retain observed hard supervision only')
        translated['questions'][qid] = dict(raw, src=old_source(src))
    for q in encoded['questions']:
        if q.get(MARKER) is not None or q['src'] != originals[q['qid']]:
            raise ValueError('Pristine aligned materialized questions required')
        q['src'] = old_source(q['src'])
    result = previous.annotate(encoded, translated)
    for q in result['questions']:
        q['src'] = originals[q['qid']]
        q[MARKER] = METHOD
    return result


def parts(logits, question, device):
    if question.get(MARKER) != METHOD or question['src'] not in MASS:
        raise ValueError('Actual yaku-course annotation required')
    # Previous.parts validates preserved old annotation/masks/RI indices.
    # This local copy never mutates the source question or module globals.
    return previous.parts(logits, dict(question, src=old_source(question['src'])), device)


def weighted_loss(logits, question, device, source_weights, observer=None):
    import torch
    if set(source_weights) != set(MASS) or any(not math.isfinite(v) or v <= 0 for v in source_weights.values()):
        raise ValueError('All25 positive registered source weights required')
    result = parts(logits, question, device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'], **{k: float(v.detach()) for k, v in result.items()}))
    return result['objective']*source_weights[question['src']]


def summarize(pending, source_weights):
    from collections import defaultdict
    if not pending or set(source_weights) != set(MASS):
        raise ValueError('Actual optimizer examples and full source weights required')
    grouped = defaultdict(list)
    for row in pending:
        if row['source'] not in MASS or any(not math.isfinite(row[k]) for k in COMPONENTS):
            raise ValueError('Actual finite source loss components required')
        grouped[row['source']].append(row)
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in grouped.items()}
    counts = {s: len(rows) for s, rows in grouped.items()}; n = sum(counts.values())
    average = lambda k: sum(counts[s]*source_weights[s]*m[k] for s, m in means.items())/n
    return dict(source_loss_components=means, source_counts=counts,
        source_objective={s: m['objective'] for s, m in means.items()},
        source_ce={s: m['standard_target_ce'] for s, m in means.items()},
        loss=average('objective'), ce=average('standard_target_ce'), extra_RI_class_loss=average('extra_RI_group_ce'))


def epoch_order(categories, rng):
    if len(categories) != RECORDS or set(categories) != set(MASS):
        raise ValueError('Complete actual25-source TRAIN corpus required')
    order = course_order(categories, rng, 511, min_tail_fraction=0.)
    updates = [order[i:i+73] for i in range(0, len(order), 73)]
    blocks = [order[i:i+511] for i in range(0, len(order), 511)]
    if (sorted(order) != list(range(RECORDS)) or len(updates) != STEPS_PER_EPOCH
            or len(updates[-1]) != 70 or len(blocks) != 179 or len(blocks[-1]) != 289
            or any({categories[i] for i in block} != set(MASS) for block in blocks)):
        raise ValueError('Full no-replacement25-source block/microbatch layout failed')
    return order, dict(records=RECORDS, optimizer_updates=STEPS_PER_EPOCH,
        mixing_blocks=179, mixing_block_records=511, final_mixing_block_records=289,
        final_optimizer_update_records=70, each_record_exactly_once=True, copies_added=0,
        all25_sources_each_mixing_block=True, all25_sources_each_optimizer_update=False,
        source_counts_by_update=[dict(Counter(categories[i] for i in batch)) for batch in updates])


def materialization_unchanged(original, annotated):
    restored = copy.deepcopy(annotated)
    for q in restored['questions']:
        for name in (MARKER, previous.MARKER, previous.FIELD):
            q.pop(name, None)
    if restored != original:
        raise ValueError('Source alias/annotation changed actual visible materialization')
