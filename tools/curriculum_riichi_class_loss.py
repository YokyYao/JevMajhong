"""Prospective one-forward RI decision-aligned loss; not an admitted fit."""
from tools.curriculum_basic_repair import MASS

FIELD='_riichi_loss_indices_v1'


def annotate(encoded,request):
    """Attach loss-only indices after the pinned materializer chooses key order."""
    for question in encoded['questions']:
        raw=request['questions'][question['qid']];source=raw['src'].removeprefix('curriculum1:')
        if source not in MASS or raw.get('target') is not None:raise ValueError('Only six-source expert hard labels supported')
        keys=question['keys'];criteria=raw['criteria']
        if (len(set(keys))!=len(keys) or set(keys)!=set(criteria) or keys[question['label']]!=raw['label']
            or question['src']!=raw['src']):raise ValueError('Pinned materialized candidate order/label differs')
        reach=[i for i,k in enumerate(keys) if criteria[k]['type']=='reach']
        types={v['type'] for v in criteria.values()};kind=source.split(':')[1]
        if kind=='plain':
            if reach:raise ValueError('Plain row cannot contain a reach choice')
        elif (len(reach)!=1 or types!={'reach','dahai'}
            or (question['label'] in reach)!=(kind=='riichi_yes')):
            raise ValueError('RI supervision requires one reach, ordinary discards and consistent class')
        question[FIELD]=reach
    return encoded


def parts(logits,question,device):
    import torch
    from torch.nn import functional as F
    key=question['src'].removeprefix('curriculum1:')
    if key not in MASS or question.get('target') is not None or FIELD not in question:
        raise ValueError('Annotated six-source hard supervision required')
    if logits.ndim!=1 or not torch.isfinite(logits).all():raise ValueError('Finite one-dimensional candidate logits required')
    label=question['label'];indices=question[FIELD];kind=key.split(':')[1]
    if type(label) is not int or not 0<=label<len(logits):raise ValueError('Invalid materialized hard label')
    work=logits if logits.dtype==torch.float64 else logits.float()
    target=torch.tensor([label],device=device);standard=F.cross_entropy(work[None],target);zero=standard*0
    if kind=='plain':
        if indices:raise ValueError('Plain loss must remain ordinary hard CE')
        return dict(objective=standard,standard_action_ce=standard,riichi_max_class_ce=zero,
            riichi_group_class_ce=zero,conditional_discard_ce=zero,plain_action_ce=standard)
    if (len(indices)!=1 or type(indices[0]) is not int or not 0<=indices[0]<len(work)
        or len(work)<2 or (label in indices)!=(kind=='riichi_yes')):
        raise ValueError('Exactly one consistent reach candidate and discard alternatives required')
    reach=indices[0];discards=[i for i in range(len(work)) if i!=reach];d=work[discards];r=work[reach]
    yes=kind=='riichi_yes';margin=r-d.amax()
    # amax distributes subgradients over tied maxima, preserving candidate
    # permutation equivariance. This replaces only the class component.
    maximum=F.softplus(-margin if yes else margin)
    group_margin=r-torch.logsumexp(d,0)
    group=F.softplus(-group_margin if yes else group_margin)
    conditional=zero if yes else F.cross_entropy(d[None],torch.tensor([discards.index(label)],device=device))
    return dict(objective=maximum+conditional,standard_action_ce=standard,
        riichi_max_class_ce=maximum,riichi_group_class_ce=group,conditional_discard_ce=conditional,plain_action_ce=zero)


def weighted_loss(logits,question,device,weights,observer=None):
    import torch
    key=question['src'].removeprefix('curriculum1:')
    if key not in weights:raise ValueError('Registered source weight required')
    result=parts(logits,question,device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=key,**{k:float(v.detach()) for k,v in result.items()}))
    return result['objective']*weights[key]
