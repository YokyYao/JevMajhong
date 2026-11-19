"""Fourth-place pursuit metrics, source weighting and explicitly scoped guards."""
from collections import Counter,defaultdict
import math


def source_weights(records,config):
    counts=Counter(r['questions']['action']['src'] for r in records)
    if len(records)!=905:raise ValueError('The fixed905 training membership must remain')
    mass=dict(config['fixed_loss_source_mass']);strict='endgame_terminal_fourth_chase_improve';keep='endgame_terminal_fourth_chase_keep'
    n,rest=counts.get(strict,0),counts.get(keep,0)
    if n+rest!=55:raise ValueError('All55 ordinary fourth training menus must remain')
    if not n:mass[keep]=.20
    elif not rest:mass[strict]=.20
    else:
        mass[strict]=min(.15,8*n/905);mass[keep]=.20-mass[strict]
    if set(counts)!=set(mass) or abs(sum(mass.values())-1.)>1e-12:
        raise ValueError('Training source classes or normalization changed')
    weights={k:v*905/counts[k] for k,v in mass.items()}
    if max(weights.values())>8+1e-12:raise ValueError('A training source exceeds the registered amplification cap')
    return {'counts':dict(counts),'mass':mass,'weights':weights}


def chase_validation(records,predictions,reference):
    from tools.audit_endgame_target_learning import validation_metrics
    from .fourth_chase import fourth_chase_action_keys
    from .public_policy import action_keys
    from .reference import restrict_distribution,distribution
    from .native_terminal import is_native_terminal
    legacy=validation_metrics(records,predictions,reference)
    cohorts=defaultdict(list);third_native=[];fourth_native=[]
    for row in records:
        meta,q=row['_meta'],row['questions']['action'];rid=meta['id'];p=predictions[rid]
        distribution(p,list(q['criteria']));rank=meta.get('source_initial_placement')
        if rank==3 and q['src']=='replay_native':
            current=restrict_distribution(p,action_keys(row));prior=reference[rid]['probabilities']
            third_native.append(sum(v*math.log(max(v,1e-30)/max(current[k],1e-30)) for k,v in prior.items() if v>0))
        if rank!=4 or q['src']!='endgame_terminal':continue
        values=meta['fourth_chase_top2_probabilities'];escape=meta['terminal_escape4_probabilities']
        eligible=fourth_chase_action_keys(row);chosen=max(eligible,key=p.get)
        prior=reference[rid]['probabilities'];baseline=max(action_keys(row),key=prior.get)
        normalized=restrict_distribution(p,eligible)
        current={'top2':values[chosen],'fourth':1-escape[chosen],'original_top2':values[baseline],
            'original_fourth':1-escape[baseline],
            'soft_top2':sum(normalized[k]*values[k] for k in normalized),
            'old_utility_at_actual_choice':meta['expected_terminal_utilities'][chosen]}
        group='native' if is_native_terminal(meta) else 'legacy';cohorts[group].append(current)
        if group=='native':
            fourth_native.append(sum(v*math.log(max(v,1e-30)/max(normalized[k],1e-30)) for k,v in prior.items() if v>0))
    if (len(third_native)!=8 or len(fourth_native)!=8
            or {k:len(v) for k,v in cohorts.items()}!={'legacy':43,'native':8}):
        raise ValueError('Preserve original43/new8 fourth and8 third/8 fourth native cohorts')
    def mean(rows):return dict({key:sum(v[key] for v in rows)/len(rows) for key in rows[0]},positions=len(rows))
    fourth=mean(cohorts['legacy']+cohorts['native'])
    return {'legacy':legacy,'fourth':fourth,'fourth_cohorts':{k:mean(v) for k,v in cohorts.items()},
        'native_third_kl':{'positions':8,'mean':sum(third_native)/8},
        'native_fourth_kl_report_only':{'positions':8,'mean':sum(fourth_native)/8}}


def chase_guards(step,current,initial,source_config,*,exposure_met=True):
    from .constrained_head import guarded_validation
    old=guarded_validation(step,current['legacy'],initial['legacy'],source_config,exposure_met=step>0)
    keys=('replay_guard_passed','placement_strata_guard_passed','reference_guard_passed',
          'leading_native_reference_guard_passed','third_pursuit_guard_passed','win_stratum_guard_passed')
    flags={k:old[k] for k in keys}
    native_limit=max(.05,initial['native_third_kl']['mean'])
    flags['native_third_reference_guard_passed']=current['native_third_kl']['mean']<=native_limit
    flags['exposure_guard_passed']=step==0 or exposure_met
    common=all(flags.values())
    floor=max(initial['fourth']['top2'],initial['fourth']['original_top2'])
    improves=current['fourth']['top2']>floor+1e-12
    return dict(flags,common_guards_passed=common,fourth_top2_floor=floor,
        fourth_top2_improved=improves,eligible_nonzero=step>0 and common and improves,
        all_guards_passed=step==0 or common and improves,
        native_third_kl_limit=native_limit,fourth_escape_constraint='report_only',
        native_fourth_kl_constraint='report_only')


def fourth_probabilities(label,keys,seat,worlds,parent_identity,split):
    from .context import placements,placement_objective,terminal_utility
    if (split not in ('train','validation') or label['source']['split']!=split
            or label['source']['seat']!=seat or label['baseline_identity_sha256']!=parent_identity
            or label['status']!='complete_terminal_menu' or not label['continued_until_native_match_done']
            or label['worlds_per_action']!=worlds or label['actual_actions']!=len(keys)
            or list(label['record']['questions']['action']['criteria'])!=keys or len(label['outcomes'])!=len(keys)):
        raise ValueError('Raw terminal menu, source split, seat or model identity changed')
    top2={};escape={};utility={}
    for key,outcomes in zip(keys,label['outcomes']):
        if len(outcomes)!=worlds:raise ValueError('Incomplete paired worlds')
        for value in outcomes:
            if (value['ledger_passed'] is not True or placements(value['scores'])!=value['ranks']
                    or terminal_utility(value['ranks'][seat])!=value['utility']
                    or value['hand_history']['final_scores']!=value['scores']):
                raise ValueError('Complete terminal scores/ranks/history do not reconcile')
        top2[key]=sum(o['ranks'][seat]<=2 for o in outcomes)/worlds
        escape[key]=sum(o['ranks'][seat]<4 for o in outcomes)/worlds
        utility[key]=sum(placement_objective(o['ranks'][seat]) for o in outcomes)/worlds
    return top2,escape,utility


def chase_training_metrics(records,predictions,reference):
    from .fourth_chase import fourth_chase_action_keys
    from .public_policy import action_keys
    counts=Counter();sums=Counter()
    for row in records:
        meta,q=row['_meta'],row['questions']['action'];rid=meta['id'];p=predictions[rid]
        eligible=fourth_chase_action_keys(row);chosen=max(eligible,key=p.get)
        prior=reference[rid]['probabilities'];baseline=max(action_keys(row),key=prior.get)
        top2=meta['fourth_chase_top2_probabilities'];escape=meta['terminal_escape4_probabilities']
        strict=meta['strict_fourth_top2_improvement'];counts['positions']+=1;counts['strict_positions']+=strict
        counts['strict_target_hits']+=strict and q['target'][chosen]>0
        counts['risky_choices_with_larger_fourth_tail']+=escape[chosen]<escape[baseline]-1e-12
        sums['chosen_top2']+=top2[chosen];sums['original_top2']+=top2[baseline]
        sums['chosen_fourth']+=1-escape[chosen];sums['original_fourth']+=1-escape[baseline]
    return {'counts':dict(counts),'means':{k:v/len(records) for k,v in sums.items()},'training_only':True}


def progress_state(step,current,guards,previous,initial):
    hard=current['fourth']['top2'];soft=current['fourth']['soft_top2'];tail=current['fourth']['fourth']
    if step==0:
        return {'selected_step':0,'selected_hard':None,'selected_fourth':None,'best_soft':soft,
            'best_hard':hard,'stale':0,'unsafe_stale':0,'stop':False}
    value=dict(previous);selected=False
    if guards['eligible_nonzero'] and (value['selected_hard'] is None or hard>value['selected_hard']+1e-12
            or abs(hard-value['selected_hard'])<=1e-12 and tail<value['selected_fourth']-1e-12):
        value.update(selected_step=step,selected_hard=hard,selected_fourth=tail);selected=True
    advancing=selected or hard>value['best_hard']+1e-12 or soft>value['best_soft']+1e-6
    value['stale']=0 if advancing else value['stale']+1
    value['best_soft']=max(value['best_soft'],soft);value['best_hard']=max(value['best_hard'],hard)
    unsafe=not guards['common_guards_passed'] or hard<initial['fourth']['top2']-1e-12
    value['unsafe_stale']=value['unsafe_stale']+1 if unsafe else 0
    value['stop']=value['stale']>=2 or value['unsafe_stale']>=2
    return value
