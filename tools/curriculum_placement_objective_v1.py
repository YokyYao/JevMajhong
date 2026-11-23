"""Prospective terminal soft-target loss; no source mix or fit is admitted.

Replay keeps the exact3B loss. Partial action-value targets normalize only
over actually tested actions. Complete RI windows additionally supervise
the full reach-versus-all-discards class used by RI0.5 execution.
"""
import math
from tools import curriculum_yaku_objective_v1 as replay

METHOD='placement_soft_full_RI_class2_tested_action_mask_v1'
MARKER='_placement_soft_objective_v1'
FIELD='_placement_soft_action_types_v1'
COMPONENTS=replay.COMPONENTS


def new_source(source):
    return isinstance(source,str) and source.startswith(('curriculum4:rank_win:','curriculum4:rank_nonwin:'))


def annotate(encoded,request):
    for q in encoded['questions']:
        raw=request['questions'][q['qid']]
        if not new_source(raw['src']):
            replay.annotate(dict(encoded,questions=[q]),dict(request,questions={q['qid']:raw}))
            continue
        keys=q['keys'];target=q.get('target');expected=raw.get('target')
        if (q['src']!=raw['src'] or MARKER in q or len(set(keys))!=len(keys)
            or set(keys)!=set(raw['criteria']) or not isinstance(expected,dict)
            or set(expected)!=set(keys) or not isinstance(target,list)
            or len(target)!=len(keys) or keys[q['label']]!=raw['label']
            or any(target[i]!=expected[k] for i,k in enumerate(keys))):
            raise ValueError('Pristine aligned original soft labels, keys and source required')
        if (any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in target)
            or abs(sum(target)-1)>1e-10 or sum(v>0 for v in target)<2):
            raise ValueError('Finite normalized target on at least two actually tested actions required')
        kinds=[raw['criteria'][k]['type'] for k in keys]
        if set(kinds)=={'reach','dahai'} and (kinds.count('reach')!=1 or any(v<=0 for v in target)):
            raise ValueError('RI class target must cover the unique reach and ALL actual discards')
        q[FIELD]=kinds;q[MARKER]=METHOD
    return encoded


def parts(logits,q,device,class_weight=2.):
    if not new_source(q['src']):
        return replay.parts(logits,q,device)
    import torch
    from torch.nn import functional as F
    if (q.get(MARKER)!=METHOD or FIELD not in q or logits.ndim!=1
        or not torch.isfinite(logits).all() or not math.isfinite(class_weight) or class_weight<1):
        raise ValueError('Annotated finite placement loss and class weight at least1 required')
    work=logits if logits.dtype==torch.float64 else logits.float()
    target=torch.tensor(q.get('target'),dtype=work.dtype,device=device)
    kinds=q[FIELD]
    if (target.shape!=work.shape or len(kinds)!=len(work) or not torch.isfinite(target).all()
        or (target<0).any() or abs(float(target.sum())-1)>1e-6 or int((target>0).sum())<2):
        raise ValueError('Finite aligned normalized tested-action target required')
    target=target/target.sum();support=target>0
    ce=-(target[support]*work[support].log_softmax(0)).sum();zero=ce*0
    if set(kinds)!={'reach','dahai'}:
        return {k:ce if k in ('objective','standard_target_ce','masked_MC_ce') else zero for k in COMPONENTS}
    if kinds.count('reach')!=1 or not bool(support.all()):
        raise ValueError('Partial RI labels cannot identify the full legal-menu RI class')
    reach=kinds.index('reach');discards=[i for i in range(len(work)) if i!=reach]
    normalizer=torch.logsumexp(work,0)
    log_yes=work[reach]-normalizer
    log_no=torch.logsumexp(work[discards],0)-normalizer
    mass=target[reach]
    group=-mass*log_yes-(1-mass)*log_no
    conditional=-(target[discards]*F.log_softmax(work[discards],0)).sum()
    return dict(objective=ce+(class_weight-1)*group,standard_target_ce=ce,
        RI_group_ce=group,extra_RI_group_ce=(class_weight-1)*group,
        conditional_discard_ce=conditional,ordinary_hard_ce=zero,masked_MC_ce=zero)


def weighted_loss(logits,q,device,source_weights,observer=None):
    import torch
    source=q['src']
    if source not in source_weights or not math.isfinite(source_weights[source]) or source_weights[source]<=0:
        raise ValueError('Positive registered source weight required')
    result=parts(logits,q,device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=source,**{k:float(v.detach()) for k,v in result.items()}))
    return result['objective']*source_weights[source]

