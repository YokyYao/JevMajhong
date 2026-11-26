"""Offline public-context learnability probe, never a Mahjong policy or parent."""
from collections import Counter
import math

FEATURES=('self_is_dealer','self_score','score_rank','gap_to_second','lead_over_fourth',
    'turn','opponent_riichi_count','honba','riichi_sticks','visible_dora_indicators',
    'tenpai_discard_options','tenpai_ukeire_min','tenpai_ukeire_max',
    'safe_tenpai_discard_options','safe_tenpai_ukeire_max',
    'value_honor_roles_min','value_honor_roles_max','dora_after_discard_min',
    'dora_after_discard_max','red_after_discard_min','red_after_discard_max','complete_tenpai_summary')
RI_MASS={'legacy:riichi_yes':.175,'legacy:riichi_no':.175,
         'teacher:riichi_yes':.325,'teacher:riichi_no':.325}


def feature_vector(context):
    """Explicit public-summary allowlist; each value has a missingness flag."""
    result=[]
    for name in FEATURES:
        value=context[name]
        if value is not None and (type(value) not in (int,float,bool) or not math.isfinite(value)):
            raise ValueError('Finite public scalar or explicit missing value required')
        result.extend((0. if value is None else float(value),float(value is None)))
    return result


def example_weights(sources):
    counts=Counter(sources)
    if set(counts)!=set(RI_MASS):raise ValueError('All four source/RI classes required')
    return [len(sources)*RI_MASS[s]/counts[s] for s in sources]


def standardize(training):
    """Fit normalization on TRAIN only; constant features keep unit scale."""
    import torch
    if training.ndim!=2 or training.shape[1]!=2*len(FEATURES) or not torch.isfinite(training).all():
        raise ValueError('Complete finite training feature matrix required')
    mean=training.mean(0);std=training.std(0,unbiased=False)
    std=torch.where(std<1e-6,torch.ones_like(std),std)
    return mean,std


def confusion(sources,truth,selected):
    if len(sources)!=len(truth) or len(truth)!=len(selected):raise ValueError('Aligned cases required')
    result={name:Counter() for name in ('legacy','teacher')}
    for source,yes,actual in zip(sources,truth,selected):
        if source not in RI_MASS or type(yes) is not bool or type(actual) is not bool:
            raise ValueError('Exact binary source classes required')
        if yes!=source.endswith(':riichi_yes'):raise ValueError('RI source/target disagree')
        result[source.split(':')[0]]['tp' if yes and actual else 'fn' if yes else 'fp' if actual else 'tn']+=1
    def metrics(counts):
        values={key:counts[key] for key in ('tp','fn','tn','fp')};yes=values['tp']+values['fn'];no=values['tn']+values['fp']
        return dict(counts=values,accept_recall=values['tp']/yes if yes else None,
            decline_recall=values['tn']/no if no else None,
            balanced_recall=.5*(values['tp']/yes+values['tn']/no) if yes and no else None)
    pooled=sum(result.values(),Counter())
    return dict(by_source={key:metrics(value) for key,value in result.items()},pooled=metrics(pooled))


def fit_probe(training,truth,sources):
    """One predeclared zero-init linear model,500 full-batch steps; no VAL arg."""
    import torch
    if not training.is_cuda or not truth.is_cuda:raise ValueError('Allocated CUDA probe required')
    if training.shape[0]!=len(sources) or truth.shape!=(len(sources),):raise ValueError('Aligned TRAIN cases required')
    torch.manual_seed(2026112305)
    mean,std=standardize(training);x=(training-mean)/std
    model=torch.nn.Linear(x.shape[1],1,device=training.device,dtype=training.dtype)
    with torch.no_grad():model.weight.zero_();model.bias.zero_()
    weights=torch.tensor(example_weights(sources),device=x.device,dtype=x.dtype)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=.01)
    losses=[]
    for step in range(1,501):
        optimizer.zero_grad(set_to_none=True)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),truth,reduction='none')*weights).mean()
        loss.backward();optimizer.step();losses.append(dict(step=step,weighted_binary_ce=float(loss.detach())))
    return model,mean,std,losses
