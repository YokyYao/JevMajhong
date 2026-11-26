"""Fixed cached-score and loss-weight diagnostics; no policy/threshold fitting."""
from collections import Counter,defaultdict
import math
import statistics

from tools.audit_curriculum_riichi_execution import auc
from tools.curriculum_basic_repair import MASS
from tools.curriculum_calls_repair import prediction_index


def distribution(values):
    if not values or any(not math.isfinite(v) for v in values):raise ValueError('Finite nonempty diagnostic cohort required')
    values=sorted(values)
    def quantile(p):
        index=(len(values)-1)*p;low=math.floor(index);high=math.ceil(index)
        return values[low]+(index-low)*(values[high]-values[low])
    return dict(n=len(values),mean=statistics.mean(values),minimum=values[0],maximum=values[-1],
        p10=quantile(.1),median=quantile(.5),p90=quantile(.9))


def summarize_cases(cases):
    if not cases or {r['truth'] for r in cases}!={False,True}:raise ValueError('Both riichi label classes required')
    confusion={method:Counter() for method in ('raw_argmax','fixed_full_class_0_5')};disagree=Counter()
    for r in cases:
        for method in confusion:
            actual=r[method];truth=r['truth']
            confusion[method]['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn']+=1
        if r['raw_argmax']!=r['fixed_full_class_0_5']:
            disagree['total']+=1;disagree['teacher_yes' if r['truth'] else 'teacher_no']+=1
            disagree['raw_reach_only' if r['raw_argmax'] else 'marginal_reach_only']+=1
    bins=[]
    for index in range(10):
        cohort=[r for r in cases if min(9,int(r['p_riichi']*10))==index]
        bins.append(dict(lower=index/10,upper=(index+1)/10,n=len(cohort),
            mean_prediction=statistics.mean(r['p_riichi'] for r in cohort) if cohort else None,
            observed_teacher_acceptance=statistics.mean(r['truth'] for r in cohort) if cohort else None))
    cohorts={}
    for truth in (False,True):
        selected=[r for r in cases if r['truth']==truth]
        cohorts['teacher_yes' if truth else 'teacher_no']={field:distribution([r[field] for r in selected])
            for field in ('p_riichi','logit_margin_vs_best_discard','discard_choices','class_nll','conditional_discard_nll','normalized_action_nll')}
    return dict(windows=len(cases),teacher_acceptance=statistics.mean(r['truth'] for r in cases),
        class_probability_auc=auc([(r['p_riichi'],r['truth']) for r in cases]),
        argmax_logit_margin_auc=auc([(r['logit_margin_vs_best_discard'],r['truth']) for r in cases]),
        class_brier=statistics.mean((r['p_riichi']-r['truth'])**2 for r in cases),
        mean_class_nll=statistics.mean(r['class_nll'] for r in cases),
        mean_conditional_discard_nll=statistics.mean(r['conditional_discard_nll'] for r in cases),
        mean_normalized_action_nll=statistics.mean(r['normalized_action_nll'] for r in cases),
        max_nll_decomposition_error=max(r['nll_decomposition_error'] for r in cases),
        confusion={k:{name:c[name] for name in ('tp','fn','tn','fp')} for k,c in confusion.items()},
        fixed_decoder_disagreements=dict(disagree),cohorts=cohorts,equal_width_calibration_bins=bins,
        class_ece10=sum(b['n']/len(cases)*abs(b['mean_prediction']-b['observed_teacher_acceptance']) for b in bins if b['n']),
        imitation_label_calibration_not_win_probability=True)


def score_rows(records,predictions):
    saved=prediction_index(records,predictions);by_source=defaultdict(list);cases=[];normalization_error=0.
    for row in records:
        q=row['questions']['action'];source_key=q['src'].removeprefix('curriculum1:')
        if source_key not in MASS:raise ValueError('Six-source basic schema required')
        source,category=source_key.split(':');types={v['type'] for v in q['criteria'].values()}
        if category=='plain':
            if 'reach' in types:raise ValueError('Riichi offer incorrectly tagged plain')
            continue
        p=saved[row['_meta']['id']];keys=p['keys'];probabilities=p['p'];total=sum(probabilities)
        reach=[k for k in keys if q['criteria'][k]['type']=='reach'];discards=[k for k in keys if q['criteria'][k]['type']=='dahai']
        if len(reach)!=1 or not discards or types!={'reach','dahai'}:raise ValueError('This diagnosis requires one reach and ordinary discard alternatives')
        truth=q['criteria'][q['label']]['type']=='reach'
        if truth!=(category=='riichi_yes'):raise ValueError('Riichi source tag and hard label differ')
        prob={k:v/total for k,v in zip(keys,probabilities)};normalization_error=max(normalization_error,abs(total-1))
        pr=prob[reach[0]];pd=sum(prob[k] for k in discards);best=max(prob[k] for k in discards)
        if min(pr,pd,best,prob[q['label']])<=0:raise ValueError('Zero-mass score cannot support an exact finite log decomposition')
        action_nll=-math.log(prob[q['label']]);class_nll=-math.log(pr if truth else pd)
        conditional=0. if truth else -math.log(prob[q['label']]/pd)
        error=abs(action_nll-class_nll-conditional)
        if error>1e-10:raise ValueError('Action CE does not decompose into class and conditional CE')
        chosen=keys[max(range(len(keys)),key=probabilities.__getitem__)]
        item=dict(id=row['_meta']['id'],group=row['_meta']['group_id'],source=source,truth=truth,
            raw_argmax=q['criteria'][chosen]['type']=='reach',fixed_full_class_0_5=probabilities[keys.index(reach[0])]>=.5,
            p_riichi=pr,logit_margin_vs_best_discard=math.log(pr)-math.log(best),discard_choices=len(discards),
            class_nll=class_nll,conditional_discard_nll=conditional,normalized_action_nll=action_nll,nll_decomposition_error=error)
        cases.append(item);by_source[source].append(item)
    if set(by_source)!={'legacy','teacher'}:raise ValueError('Both actual source cohorts required')
    return dict(sources={s:summarize_cases(v) for s,v in by_source.items()},pooled=summarize_cases(cases),
        max_saved_probability_sum_error=normalization_error,one_reach_candidate_verified=True),cases


def update_mass(updates,weights,expected_counts):
    if set(weights)!=set(MASS) or set(expected_counts)!=set(MASS):raise ValueError('All six actual loss sources required')
    count=sum(expected_counts.values())
    if any(not math.isfinite(weights[k]) or abs(weights[k]*expected_counts[k]/count-MASS[k])>1e-12 for k in MASS):
        raise ValueError('Actual per-example weights differ from registered six-source masses')
    result=[]
    for row in updates:
        counts=row['source_counts'];ces=row['source_ce']
        if set(counts)!=set(MASS) or set(ces)!=set(MASS) or any(type(n) is not int or n<=0 for n in counts.values()):
            raise ValueError('Actual optimizer update lacks a source or has invalid counts')
        if any(not math.isfinite(v) or v<0 for v in ces.values()):raise ValueError('Invalid actual source CE')
        weighted={k:weights[k]*counts[k] for k in MASS};total=sum(weighted.values())
        positive=sum(v for k,v in weighted.items() if k.endswith(':riichi_yes'))
        negative=sum(v for k,v in weighted.items() if k.endswith(':riichi_no'))
        loss=sum(weighted[k]*ces[k] for k in MASS)/sum(counts.values())
        if abs(loss-row['loss'])>1e-5:raise ValueError('Actual weighted objective does not reconcile')
        result.append(dict(step=row['local_step'],positive_loss_weight=positive,negative_loss_weight=negative,
            all_source_loss_weight=total,riichi_positive_fraction=positive/(positive+negative),
            riichi_fraction_of_all_weight=(positive+negative)/total,
            positive_ce=sum(weighted[k]*ces[k] for k in MASS if k.endswith(':riichi_yes'))/positive,
            negative_ce=sum(weighted[k]*ces[k] for k in MASS if k.endswith(':riichi_no'))/negative,
            loss_reconstruction_error=abs(loss-row['loss'])))
    return result


def summarize_mass(rows):
    if not rows:raise ValueError('A nonempty actual update interval is required')
    pos=sum(r['positive_loss_weight'] for r in rows);neg=sum(r['negative_loss_weight'] for r in rows)
    return dict(first_step=rows[0]['step'],last_step=rows[-1]['step'],updates=len(rows),
        aggregate_riichi_positive_fraction=pos/(pos+neg),
        per_update_riichi_positive_fraction=distribution([r['riichi_positive_fraction'] for r in rows]),
        steps_positive_fraction_below_0_25=sum(r['riichi_positive_fraction']<.25 for r in rows),
        steps_positive_fraction_above_0_75=sum(r['riichi_positive_fraction']>.75 for r in rows),
        positive_weighted_ce=sum(r['positive_loss_weight']*r['positive_ce'] for r in rows)/pos,
        negative_weighted_ce=sum(r['negative_loss_weight']*r['negative_ce'] for r in rows)/neg,
        maximum_loss_reconstruction_error=max(r['loss_reconstruction_error'] for r in rows),
        loss_weight_mass_is_not_measured_gradient_magnitude=True)
