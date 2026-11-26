"""Prospective twelve-source calls loss retaining the isolated basic RI objective.

No trainer, parent or fit is admitted by this module. No serving code changes.
"""
import copy
import math
from tools.build_curriculum_calls_repair import MASS
from tools.curriculum_riichi_class_loss import FIELD, annotate as basic_annotate, parts as basic_parts

MARKER='_calls_riichi_retention_v1'
METHOD='basic_riichi_max_class_retained_other_calls_ce_unchanged_v1'


def source_key(source):
    if not source.startswith('curriculum2:') or source.removeprefix('curriculum2:') not in MASS:
        raise ValueError('Only the existing twelve-source expert calls dataset is supported')
    return source.removeprefix('curriculum2:')


def annotate(encoded,request):
    """Preserve original source/labels/keys; add private loss-only indices."""
    for question in encoded['questions']:
        raw=request['questions'][question['qid']];key=source_key(raw['src'])
        keys=question['keys'];criteria=raw['criteria']
        if (question.get('target') is not None or raw.get('target') is not None
            or question['src']!=raw['src'] or len(keys)!=len(set(keys)) or set(keys)!=set(criteria)
            or keys[question['label']]!=raw['label'] or MARKER in question or FIELD in question):
            raise ValueError('Pristine aligned expert materialization required')
        indices=[]
        if key.startswith('basic:'):
            src='curriculum1:'+key.removeprefix('basic:')
            temporary=copy.deepcopy(question);temporary['src']=src
            raw_copy=copy.deepcopy(raw);raw_copy['src']=src
            basic_annotate({'questions':[temporary]},{'questions':{question['qid']:raw_copy}})
            indices=temporary[FIELD]
        question[FIELD]=indices;question[MARKER]=METHOD
    return encoded


def parts(logits,question,device):
    import torch
    from torch.nn import functional as F
    key=source_key(question['src'])
    if question.get('target') is not None or question.get(MARKER)!=METHOD or FIELD not in question:
        raise ValueError('Annotated twelve-source hard supervision required')
    if logits.ndim!=1 or not torch.isfinite(logits).all():raise ValueError('Finite candidate logits required')
    label=question['label']
    if type(label) is not int or not 0<=label<len(logits):raise ValueError('Invalid materialized hard label')
    if key.startswith('basic:') and key.rsplit(':',1)[1] in ('riichi_yes','riichi_no'):
        mapped=dict(question,src='curriculum1:'+key.removeprefix('basic:'))
        result=basic_parts(logits,mapped,device)
        return dict(result,advanced_action_ce=result['objective']*0)
    if question[FIELD]:raise ValueError('Non-basic-RI loss must not acquire RI indices')
    # Match the original calls trainer exactly, including CE dtype/autocast.
    ce=F.cross_entropy(logits[None],torch.tensor([label],device=device));zero=ce*0
    return dict(objective=ce,standard_action_ce=ce,riichi_max_class_ce=zero,
        riichi_group_class_ce=zero,conditional_discard_ce=zero,
        plain_action_ce=ce if key.startswith('basic:') else zero,
        advanced_action_ce=ce if key.startswith('advanced:') else zero)


def weighted_loss(logits,question,device,weights,observer=None):
    import torch
    key=source_key(question['src'])
    if set(weights)!=set(MASS) or not math.isfinite(weights[key]) or weights[key]<=0:
        raise ValueError('Complete positive registered twelve-source weights required')
    result=parts(logits,question,device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=key,**{name:float(value.detach()) for name,value in result.items()}))
    return result['objective']*weights[key]
