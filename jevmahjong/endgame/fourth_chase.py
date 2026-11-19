"""Explicit fourth-place top-two pursuit without an escape-probability veto."""
import math


def fourth_chase_action_keys(payload):
    from .public_policy import action_keys
    context=payload['state'].get('endgame_context',{})
    criteria=payload['questions']['action']['criteria']
    # Preserve the public win/continue gate. Ordinary legal actions can take
    # risk when the current public context confirms last place in the endgame.
    if (context.get('late_match_window') and context.get('placement')==4
            and not any(v['type']=='hora' for v in criteria.values())):
        return list(criteria)
    return action_keys(payload)


def top2_chase_target(reference,top2,escape,eligible):
    from .reference import distribution
    distribution(reference,list(top2))
    if (not eligible or not set(eligible)<=set(top2) or set(escape)!=set(top2)
            or any(not math.isfinite(v) or not 0<=v<=1 for v in [*top2.values(),*escape.values()])
            or any(top2[k]>escape[k]+1e-12 for k in top2)):
        raise ValueError('Invalid legal fourth-place terminal probabilities')
    best=max(top2[k] for k in eligible)
    winners=[k for k in eligible if top2[k]>=best-1e-12]
    # Escape only breaks exact top-two ties; it never excludes a higher-top2
    # action merely because that action also has a larger fourth-place tail.
    tie_escape=max(escape[k] for k in winners)
    winners=[k for k in winners if escape[k]>=tie_escape-1e-12]
    mass=sum(reference[k] for k in winners)
    return {k:(reference[k]/mass if mass>0 else 1/len(winners)) if k in winners else 0. for k in reference}
