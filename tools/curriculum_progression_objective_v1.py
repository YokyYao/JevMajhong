"""Prospective full-RI-class objective and exact nineteen-source mixing.

This does not admit a parent or launch training. Lambda1 decomposes ordinary
hard-action CE exactly; lambda2 is a genuinely different binary-class weight.
"""
from collections import Counter
import math

from tools.curriculum_calls_post_reach_data_v1 import MASS as CALLS_MASS
from tools.curriculum_course_order import course_order

METHOD = 'progression19_full_RI_group_class2_conditional_discard_and_masked_MC_v1'
FIELD = '_progression_RI_group_indices_v1'
MARKER = '_progression_objective_v1'
MASS = {'curriculum2:'+s: .55*v for s, v in CALLS_MASS.items()}
MASS.update({'curriculum3:teacher:closed_discard_under_riichi': .15,
             'curriculum3:teacher:ordinary_completion': .20})
MASS.update({'curriculum3:mc:'+p: .025 for p in ('closed_discard', 'open_discard', 'call_window', 'kan_window')})
COMPONENTS = ('objective', 'standard_target_ce', 'RI_group_ce', 'extra_RI_group_ce',
              'conditional_discard_ce', 'ordinary_hard_ce', 'masked_MC_ce')


def weights(counts):
    if (set(counts) != set(MASS) or any(type(n) is not int or n <= 0 for n in counts.values())
            or abs(sum(MASS.values())-1) > 1e-12):
        raise ValueError('Complete nineteen-source counts and normalized fixed loss masses required')
    result = {s: MASS[s]*sum(counts.values())/n for s, n in counts.items()}
    if max(result.values()) > 16: raise ValueError('Registered scalar example-weight cap16 exceeded')
    return result


def annotate(encoded, request):
    for q in encoded['questions']:
        raw = request['questions'][q['qid']]; keys = q['keys']; source = raw['src']
        if (source not in MASS or MARKER in q or q['src'] != source or len(set(keys)) != len(keys)
                or set(keys) != set(raw['criteria']) or keys[q['label']] != raw['label']):
            raise ValueError('Pristine aligned source, key order and observed label required')
        kinds = [raw['criteria'][k]['type'] for k in keys]
        if source.startswith('curriculum3:mc:'):
            target = q.get('target')
            if raw.get('target') is None or target is None or sum(v > 0 for v in target) < 2:
                raise ValueError('Informative unchanged masked target required')
            indices = []
        else:
            if q.get('target') is not None or raw.get('target') is not None:
                raise ValueError('Observed teacher/human hard labels must stay hard')
            indices = [i for i, k in enumerate(kinds) if k == 'reach'] if set(kinds) == {'reach', 'dahai'} else []
            if indices and len(indices) != 1: raise ValueError('Exactly one reach candidate required')
        q[FIELD] = indices; q[MARKER] = METHOD
    return encoded


def parts(logits, q, device, class_weight=2.):
    import torch
    from torch.nn import functional as F
    from tools.curriculum_losses import question_loss
    source = q['src']
    if (source not in MASS or q.get(MARKER) != METHOD or FIELD not in q
            or logits.ndim != 1 or not torch.isfinite(logits).all()
            or not math.isfinite(class_weight) or class_weight < 1):
        raise ValueError('Annotated finite full-group objective required')
    work = logits if logits.dtype == torch.float64 else logits.float()
    if source.startswith('curriculum3:mc:'):
        if q[FIELD] or q.get('target') is None: raise ValueError('MC mask cannot become RI hard supervision')
        ce = question_loss(work, dict(q, src='paired_future_masked'), device); zero = ce*0
        return {k: ce if k in ('objective', 'standard_target_ce', 'masked_MC_ce') else zero for k in COMPONENTS}
    label = q['label']
    if type(label) is not int or not 0 <= label < len(work) or q.get('target') is not None:
        raise ValueError('Aligned observed hard label required')
    ce = F.cross_entropy(work[None], torch.tensor([label], device=device)); zero = ce*0
    if not q[FIELD]:
        return {k: ce if k in ('objective', 'standard_target_ce', 'ordinary_hard_ce') else zero for k in COMPONENTS}
    if len(q[FIELD]) != 1 or type(q[FIELD][0]) is not int or not 0 <= q[FIELD][0] < len(work) or len(work) < 2:
        raise ValueError('One RI and at least one discard required')
    reach = q[FIELD][0]; discards = [i for i in range(len(work)) if i != reach]
    margin = work[reach]-torch.logsumexp(work[discards], 0)
    group = F.softplus(-margin if label == reach else margin)
    conditional = zero if label == reach else F.cross_entropy(work[discards][None], torch.tensor([discards.index(label)], device=device))
    return dict(objective=class_weight*group+conditional, standard_target_ce=ce,
        RI_group_ce=group, extra_RI_group_ce=(class_weight-1)*group,
        conditional_discard_ce=conditional, ordinary_hard_ce=zero, masked_MC_ce=zero)


def epoch_order(categories, rng):
    # The511-record mixing block is seven73-record optimizer updates.
    # Its214-record final block still covers all19 sources, and is split into
    #73/73/68 updates. Do not claim all19 sources in each optimizer update.
    if len(categories) != 87595 or set(categories) != set(MASS):
        raise ValueError('Exact audited union TRAIN categories required')
    order = course_order(categories, rng, 511, min_tail_fraction=0.)
    micro = [order[i:i+73] for i in range(0, len(order), 73)]
    macro = [order[i:i+511] for i in range(0, len(order), 511)]
    if (len(micro) != 1200 or len(macro) != 172 or len(micro[-1]) != 68
            or any({categories[i] for i in batch} != set(MASS) for batch in macro)
            or sorted(order) != list(range(len(categories)))):
        raise ValueError('Exact unique macro coverage/micro update cadence failed')
    return order, dict(records=87595, optimizer_updates=1200, final_optimizer_update_records=68,
        mixing_blocks=172, mixing_block_records=511, final_mixing_block_records=214,
        each_record_exactly_once=True, all19_sources_each_mixing_block=True,
        all19_sources_each_optimizer_update=False, copies_added=0,
        source_counts_by_update=[dict(Counter(categories[i] for i in batch)) for batch in micro])
