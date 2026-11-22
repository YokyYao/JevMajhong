"""Conditional scout summaries; no SFT targets or positive-CI filter."""
import math
import statistics


def utility(result):
    return float(result['top2'])+.02*(4-result['rank'])/3+.005*math.tanh(result['net_from_anchor']/8000)


def summarize(pairs,first,second):
    n=len(pairs)
    if not n:raise ValueError('Actual paired worlds required')
    plus=sum(bool(p[first]['top2']) and not p[second]['top2'] for p in pairs)
    minus=sum(bool(p[second]['top2']) and not p[first]['top2'] for p in pairs)
    def wilson(k):
        # Two marginal intervals with Bonferroni protection; Wilson coverage
        # is approximate, and this conditional interval is not strength proof.
        z=statistics.NormalDist().inv_cdf(1-.05/4);p=k/n;den=1+z*z/n
        mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return max(0.,mid-half),min(1.,mid+half)
    lo_plus,hi_plus=wilson(plus);lo_minus,hi_minus=wilson(minus)
    arms={key:dict(top2_rate=statistics.mean(float(p[key]['top2']) for p in pairs),
        mean_rank=statistics.mean(p[key]['rank'] for p in pairs),
        mean_net_from_anchor=statistics.mean(p[key]['net_from_anchor'] for p in pairs),
        descriptive_utility=statistics.mean(utility(p[key]) for p in pairs)) for key in (first,second)}
    return dict(worlds=n,arms=arms,paired_first_minus_second_top2=(plus-minus)/n,
        positive_disagreements=plus,negative_disagreements=minus,zero_differences=n-plus-minus,
        paired_discordance_wilson_bonferroni_approx95=[lo_plus-hi_minus,hi_plus-lo_minus],
        no_observed_disagreement_is_not_equivalence=plus+minus==0,
        both_arms_zero_top2_tail_unresolved=all(not p[key]['top2'] for p in pairs for key in (first,second)),
        conditional_world_prior_only_not_policy_strength=True,training_target_admitted=False)
