"""Prospective teacher objective: placement first, leading ties take the win.

No runtime policy or student fit uses this function. Existing scalar teacher
records remain unchanged. The initial rank must come from public scores.
"""
import math
import statistics

from tools.curriculum_public_win_planner_v1 import choose as scalar_choose

METHOD = 'public_initial_rank_lex_top2_rank_then_behind_net_v1'


def choose(pairs, accept, decline, initial_rank):
    if type(initial_rank) is not int or initial_rank not in range(1, 5):
        raise ValueError('Explicit public initial rank1..4 required')
    # Reuse only the established finite/full-terminal pair validation.
    scalar_choose(pairs, accept, decline)
    totals = {}
    means = {}
    for arm in (accept, decline):
        outcomes = [pair[arm] for pair in pairs]
        primary = sum(int(r['top2']) for r in outcomes)
        secondary = -sum(r['rank'] for r in outcomes)
        tertiary = sum(math.tanh(r['net_from_anchor']/8000) for r in outcomes) if initial_rank >= 3 else 0.
        totals[arm] = primary, secondary, tertiary
        means[arm] = dict(top2=primary/len(outcomes), rank=-secondary/len(outcomes),
            net=statistics.mean(r['net_from_anchor'] for r in outcomes))
    selected = decline if totals[decline] > totals[accept] else accept
    a, d = totals[accept], totals[decline]
    reason = ('top2' if d[0] != a[0] else 'terminal_rank' if d[1] != a[1] else
              'behind_net_tiebreak' if initial_rank >= 3 and d[2] != a[2] else 'tie_take_win')
    return dict(method=METHOD, selected=selected, worlds=len(pairs), initial_public_rank=initial_rank,
        mean_outcomes=means, decision_component=reason,
        no_last_place_rate_veto=True, no_positive_confidence_gate=True,
        exact_primary_secondary_ties_use_net_only_when_initial_rank3or4=True,
        teacher_admitted=False, training_labels=False)
