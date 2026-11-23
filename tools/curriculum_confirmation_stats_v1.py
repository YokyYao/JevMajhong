"""Predeclared paired whole-seed statistics; not a simulator or release gate.

This validates the result matrix and arithmetic only. A final runner still
must bind model/policy identities, recompute native ledgers, freeze the sole
candidate and prove the reserved grid was not previously consumed.
"""
import math
import statistics

GROUPS=32
PROFILES=('efficiency','calling')
SEATS=tuple(range(4))
T_CRITICAL_DF31_975=2.0395134463964077
PRIMARY=('top2_rate','net_points')


def matrix(rows,seed_start):
    expected={(seed,profile,seat) for seed in range(seed_start,seed_start+GROUPS)
        for profile in PROFILES for seat in SEATS}
    result={}
    for row in rows:
        seed,profile,seat=(row.get(k) for k in ('seed','profile','seat'))
        key=seed,profile,seat
        if (type(seed) is not int or type(seat) is not int or key not in expected or key in result
            or row.get('completed') is not True or row.get('ledger_passed') is not True):
            raise ValueError('Exact complete unique32-seed/four-seat/two-profile matrix required')
        scores=row.get('final_scores');rank=row.get('rank');hands=row.get('hands')
        if (not isinstance(scores,list) or len(scores)!=4 or any(type(s) is not int for s in scores)
            or type(rank) is not int or rank not in range(1,5)
            or rank!=sorted(range(4),key=lambda p:(-scores[p],p)).index(seat)+1
            or row.get('net_points')!=scores[seat]-25000
            or type(hands) is not int or hands<=0
            or any(type(row.get(k)) is not int or not 0<=row[k]<=hands for k in ('wins','dealin_hands'))):
            raise ValueError('Finite actual score/rank/hand accounting differs')
        result[key]=row
    if set(result)!=expected:raise ValueError('All256 exact matched complete East results required')
    return result


def interval(values):
    if len(values)!=GROUPS or any(not math.isfinite(v) for v in values):
        raise ValueError('Exactly32 finite independent logical-seed differences required')
    mean=statistics.mean(values);std=statistics.stdev(values)
    radius=T_CRITICAL_DF31_975*std/math.sqrt(GROUPS)
    return dict(gain=mean,ci95=[mean-radius,mean+radius],independent_seed_groups=GROUPS,
        df=31,cluster_std=std,zero_cluster_variation=std==0.,
        all_observed_differences_zero=all(v==0 for v in values))


def compare(candidate,reference,seed_start):
    left=matrix(candidate,seed_start);right=matrix(reference,seed_start)
    functions=dict(top2_rate=lambda r:float(r['rank']<=2),net_points=lambda r:r['net_points'],
        hand_win_rate=lambda r:r['wins']/r['hands'],first_rate=lambda r:float(r['rank']==1),
        fourth_rate=lambda r:float(r['rank']==4),dealin_hand_rate=lambda r:r['dealin_hands']/r['hands'])
    metrics={}
    for name,function in functions.items():
        differences=[statistics.mean(function(left[seed,profile,seat])-function(right[seed,profile,seat])
            for profile in PROFILES for seat in SEATS) for seed in range(seed_start,seed_start+GROUPS)]
        metrics[name]=interval(differences)
        metrics[name].update(candidate=statistics.mean(function(r) for r in left.values()),
            reference=statistics.mean(function(r) for r in right.values()))
    nonnegative=all(metrics[k]['ci95'][0]>=0 for k in PRIMARY)
    degenerate=any(metrics[k]['all_observed_differences_zero'] for k in PRIMARY)
    return dict(method='paired_logical_seed_mean_t_df31_v1',matches_per_policy=256,
        independent_seed_groups=32,matches_per_seed=8,metrics=metrics,primary_metrics=list(PRIMARY),
        primary_point_means_not_worse=all(metrics[k]['gain']>=0 for k in PRIMARY),
        joint_primary_lower_bounds_nonnegative=nonnegative,
        degenerate_zero_difference_not_a_parity_proof=degenerate,
        primary_lower_bounds_nominal_joint_confidence=.95,
        joint_bound_rule='two97.5_percent_one_sided_t_lower_bounds_Bonferroni_alpha0.05',
        lower_bounds_are_approximate_seed_sampling_inference=True,
        separate_policy_identity_and_native_ledger_audit_required=True,
        fourth_place_escape_is_not_a_per_state_veto=True,
        task_completion_or_release_authorized=False)

