"""Actual inverse-repeat objective/observer composition for the prospective anchor."""
from collections import defaultdict
import math
from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_progression_parent_loss_v1 as retained
from tools import curriculum_ri_inverse_repeat_loss_v1 as inverse

METHOD='per_original_inverse_repeat_retained19_plus_fixed4_parent_RI_KL_v1'
COMPONENTS=(*retained.COMPONENTS,'parent_RI_KL','parent_RI_anchor_adjustment','retained_objective_without_anchor')
attach=inverse.attach


def weighted_loss(logits,question,device,observer=None):
    import torch
    if (question.get(inverse.MARKER)!=inverse.METHOD
            or type(question.get(inverse.FIELD)) is not float
            or not math.isfinite(question[inverse.FIELD]) or not 0<question[inverse.FIELD]<=16
            or not isinstance(question.get(inverse.IDENTIFIER),str) or not question[inverse.IDENTIFIER]):
        raise ValueError('Exact positive bounded inverse-repeat metadata required')
    components=anchor.parts(logits,question,device)
    components['retained_objective_without_anchor']=components['objective']-components['parent_RI_anchor_adjustment']
    if set(components)!=set(COMPONENTS):raise ValueError('Complete actual composed objective components required')
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'],record_id=question[inverse.IDENTIFIER],
            example_weight=question[inverse.FIELD],**{k:float(v.detach()) for k,v in components.items()}))
    return components['objective']*question[inverse.FIELD]


def summarize(pending):
    if not pending:raise ValueError('Actual weighted optimized examples required')
    groups=defaultdict(list)
    for row in pending:
        if (row['source'] not in retained.MASS or not row['record_id']
                or not math.isfinite(row['example_weight']) or not 0<row['example_weight']<=16
                or any(not math.isfinite(row[k]) for k in COMPONENTS)
                or abs(row['objective']-row['standard_target_ce']-row['signed_objective_minus_standard_ce'])>1e-5
                or abs(row['objective']-row['retained_objective_without_anchor']-row['parent_RI_anchor_adjustment'])>1e-5
                or abs(row['parent_RI_anchor_adjustment']-anchor.STRENGTH*row['parent_RI_KL'])>1e-5
                or row['parent_RI_KL'] < -1e-5):
            raise ValueError('Finite reconciled actual loss components required')
        if row['source'] not in anchor.RI_SOURCES and (row['parent_RI_KL']!=0 or row['parent_RI_anchor_adjustment']!=0):
            raise ValueError('No parent RI penalty may enter an unrelated source')
        groups[row['source']].append(row)
    n=len(pending)
    weighted=lambda key:sum(r['example_weight']*r[key] for r in pending)/n
    return dict(loss=weighted('objective'),ce=weighted('standard_target_ce'),
        retained_objective_without_anchor=weighted('retained_objective_without_anchor'),
        parent_RI_KL=weighted('parent_RI_KL'),parent_RI_anchor_adjustment=weighted('parent_RI_anchor_adjustment'),
        signed_objective_adjustment=weighted('signed_objective_minus_standard_ce'),
        source_counts={s:len(v) for s,v in groups.items()},
        source_loss_components={s:{k:sum(r[k] for r in v)/len(v) for k in COMPONENTS} for s,v in groups.items()},
        source_weighted_component_contributions={s:{k:sum(r['example_weight']*r[k] for r in v)/n
            for k in COMPONENTS} for s,v in groups.items()},
        effective_source_mass={s:sum(r['example_weight'] for r in v)/n for s,v in groups.items()},
        source_example_weight_range={s:[min(r['example_weight'] for r in v),max(r['example_weight'] for r in v)] for s,v in groups.items()},
        loss_method=METHOD,unweighted_objective_method=anchor.METHOD)
