"""Versioned calls recipe and metrics preserving the qualifying basic RI loss."""
from collections import defaultdict
import math
from tools.build_curriculum_calls_repair import MASS
from tools.curriculum_calls_fit import require_recipe as old_recipe
from tools.curriculum_calls_repair import profile as old_profile, prediction_index
from tools.curriculum_riichi_control import cached_parts, COMPONENTS as BASIC_COMPONENTS
from tools.curriculum_calls_riichi_retention_v1 import METHOD, source_key
from tools.curriculum_parent import ParentNotReady

COMPONENTS=(*BASIC_COMPONENTS,'advanced_action_ce')
STAGE='stage2-calls-riichi-retention-v1'
CANDIDATE='registered_final_full_two_epochs_calls_riichi_retention_v1'


def require_parent_kind(parent):
    if (parent['training_config'].get('loss_method')!='max_discard_class_plus_conditional_v1'
        or parent['curriculum'].get('stage')!='stage1-riichi-class-control-v1'):
        raise ParentNotReady('This calls experiment requires a qualified max-class basic parent')


def require_recipe(config,data,counts):
    if (config.get('stage')!=STAGE or config.get('candidate_rule')!=CANDIDATE
        or config.get('calls_loss_method')!=METHOD):
        raise ValueError('Explicit RI-retention calls recipe required')
    old=dict(config,stage='stage2-calls-repair-v1',candidate_rule='registered_final_full_two_epochs_calls_repair_v1')
    return old_recipe(old,data,counts)


def profile(records,predictions,mass,base,ce):
    result=old_profile(records,predictions,mass,base,ce)
    saved=prediction_index(records,predictions);by=defaultdict(list)
    for row in records:
        q=row['questions']['action'];key=source_key(q['src']);p=saved[row['_meta']['id']]
        if key.startswith('basic:'):
            mapped=dict(q,src='curriculum1:'+key.removeprefix('basic:'))
            parts=dict(cached_parts(mapped,p['keys'],p['p'],ce),advanced_action_ce=0.)
        else:
            value=ce(q,p['keys'],p['p']);parts={name:0. for name in COMPONENTS}
            parts.update(objective=value,standard_action_ce=value,advanced_action_ce=value)
        by[key].append(parts)
    if set(by)!=set(MASS):raise ValueError('All twelve source strata required')
    means={s:{name:sum(row[name] for row in rows)/len(rows) for name in COMPONENTS} for s,rows in by.items()}
    result.update(source_loss_components=means,source_objective={s:row['objective'] for s,row in means.items()},
        weighted_class_objective=sum(mass[s]*means[s]['objective'] for s in MASS),
        weighted_target_ce_is_original_action_ce=True,loss_method=METHOD)
    return result


def summarize(pending,weights):
    """Reconstruct actual optimizer-accumulation means, never the raw CE alias."""
    if set(weights)!=set(MASS) or any(not math.isfinite(v) or v<=0 for v in weights.values()):
        raise ValueError('All registered positive twelve-source weights required')
    by=defaultdict(list)
    for row in pending:
        if row['source'] not in MASS or any(not math.isfinite(row[name]) for name in COMPONENTS):
            raise ValueError('Finite actual twelve-source component observations required')
        by[row['source']].append(row)
    if set(by)!=set(MASS):raise ValueError('Actual update is missing a registered source')
    means={s:{name:sum(row[name] for row in rows)/len(rows) for name in COMPONENTS} for s,rows in by.items()}
    counts={s:len(rows) for s,rows in by.items()};n=sum(counts.values())
    weighted=lambda name:sum(counts[s]*weights[s]*means[s][name] for s in by)/n
    return dict(source_loss_components=means,source_objective={s:m['objective'] for s,m in means.items()},
        source_ce={s:m['standard_action_ce'] for s,m in means.items()},source_counts=counts,
        loss=weighted('objective'),ce=weighted('standard_action_ce'))
