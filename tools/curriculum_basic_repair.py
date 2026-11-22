"""Six-source basic-course loss accounting; no public input transformation."""
from collections import Counter,defaultdict
import math

MASS={'legacy:plain':.325,'teacher:plain':.325,
      'legacy:riichi_yes':.06125,'legacy:riichi_no':.06125,
      'teacher:riichi_yes':.11375,'teacher:riichi_no':.11375}


def kind(key):
    if key not in MASS:raise ValueError('Unknown basic-repair source stratum')
    return key.split(':')[1]


def weights(counts):
    if set(counts)!=set(MASS) or any(type(n) is not int or n<=0 for n in counts.values()):raise ValueError('Six positive source counts required')
    n=sum(counts.values());result={k:MASS[k]*n/counts[k] for k in MASS}
    if max(result.values())>8:raise ValueError('Prospective per-record loss cap8 exceeded')
    return result


def accumulation(counts):
    for size in range(80,129):
        n=sum(counts.values());updates=math.ceil(n/size);tail=n-(updates-1)*size
        if updates%2==0 and tail>=.75*size and min(counts.values())>=updates:return size,updates
    raise ValueError('No count-only accumulation satisfies the registered six-source budget')


def profile(records,predictions,mass,base,ce):
    byid={r['_meta']['id']:r for r in records};source_ce=defaultdict(list);source_hits=defaultdict(list)
    aggregate_ce=defaultdict(list);aggregate_hits=defaultdict(list);confusion={p:Counter() for p in ('legacy','teacher')}
    for p in predictions:
        row=byid[p['id']];q=row['questions']['action'];key=q['src'].removeprefix('curriculum1:');category=kind(key)
        chosen=p['keys'][max(range(len(p['p'])),key=p['p'].__getitem__)];value=ce(q,p['keys'],p['p'])
        hit=chosen==q['label'];source_ce[key].append(value);source_hits[key].append(hit)
        aggregate_ce[category].append(value);aggregate_hits[category].append(hit)
        if category!='plain':
            truth=category=='riichi_yes';actual=q['criteria'][chosen]['type']=='reach'
            confusion[key.split(':')[0]]['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn']+=1
    if set(source_ce)!=set(mass) or len(predictions)!=len(records):raise ValueError('Incomplete six-source validation profile')
    mean=lambda groups:{k:sum(v)/len(v) for k,v in groups.items()}
    result=dict(base,natural_target_ce=base['weighted_target_ce'],source_stratum_ce=mean(source_ce),
        source_stratum_accuracy=mean(source_hits),source_stratum_counts={k:len(v) for k,v in source_ce.items()},
        stratum_ce=mean(aggregate_ce),stratum_accuracy=mean(aggregate_hits),stratum_counts={k:len(v) for k,v in aggregate_ce.items()},
        riichi_by_source={k:dict(c) for k,c in confusion.items()},riichi_type_counts=dict(confusion['legacy']+confusion['teacher']))
    result['weighted_target_ce']=sum(mass[k]*result['source_stratum_ce'][k] for k in mass)
    return result


def grade(validation):
    def recall(c):
        yes=c.get('tp',0)+c.get('fn',0);no=c.get('tn',0)+c.get('fp',0)
        if not yes or not no:raise ValueError('Both actual riichi classes required for grading')
        return c.get('tp',0)/yes,c.get('tn',0)/no
    accept,decline=recall(validation['riichi_type_counts']);by=validation['riichi_by_source']
    legacy_yes,legacy_no=recall(by['legacy']);teacher_yes,teacher_no=recall(by['teacher'])
    plain_ce_limit=1.3489726760783527*1.05;accuracy_limit=.4845576407506702
    original=accept>=.6 and decline>=.6 and validation['stratum_accuracy']['plain']>=accuracy_limit and validation['stratum_ce']['plain']<=plain_ce_limit
    legacy=legacy_yes>=.6 and legacy_no>=.6 and validation['source_stratum_accuracy']['legacy:plain']>=accuracy_limit and validation['source_stratum_ce']['legacy:plain']<=plain_ce_limit
    teacher=teacher_yes>=.6 and teacher_no>=.6
    return dict(original_aggregate_gate=dict(passed=original,riichi_acceptance=accept,riichi_correct_decline=decline,
        min_both_riichi_recall=.6,min_plain_exact=accuracy_limit,max_plain_ce=plain_ce_limit,development_gate_not_match_strength_proof=True),
        unchanged_legacy_gate=dict(passed=legacy,riichi_acceptance=legacy_yes,riichi_correct_decline=legacy_no),
        independent_teacher_source_gate=dict(passed=teacher,riichi_acceptance=teacher_yes,riichi_correct_decline=teacher_no),
        all_source_quality_passed=original and legacy and teacher,not_policy_strength_proof=True)
