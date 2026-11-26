"""Count-only placement mixing proposal; repeated views are not new data.

This specifies virtual exposure multiplicities, never edits original records,
selects labels by value, admits a parent, or fixes a training budget.
"""
from fractions import Fraction

from tools.curriculum_yaku_objective_v1 import MASS as REPLAY_MASS, RECORDS

WIN_CONTEXTS = ('hold_first', 'hold_second', 'third_gap3000', 'fourth_gap5000', 'fourth_gap13000')
NONWIN_CONTEXTS = WIN_CONTEXTS[1:]
CATEGORIES = ('progression_draw', 'post_call', 'call_response')
CONTROL_CONTEXTS = ('sufficient_behind', 'leading_accept_control', 'valuable_still_behind')
SPECIAL_OPERATIONS = ('win_reach', 'win_ankan', 'win_kakan')
NEW_SOURCES = (
    tuple('curriculum4:rank_win:'+c for c in WIN_CONTEXTS)
    + tuple('curriculum4:rank_nonwin:'+k+':'+c for k in CATEGORIES for c in NONWIN_CONTEXTS)
    + tuple('curriculum4:rank_win:control:'+c+':'+k for c in CONTROL_CONTEXTS for k in ('ron', 'tsumo'))
    + tuple('curriculum4:rank_win:targeted:'+k+':'+c for k in SPECIAL_OPERATIONS for c in NONWIN_CONTEXTS[:3])
)
MAX_WEIGHT = 16
MAX_REPEATS = 8
METHOD = 'placement_replay80_new32_equal20_capped_uniform_virtual_exposures_v1'


def masses():
    old = {s: Fraction(str(v)) for s, v in REPLAY_MASS.items()}
    total = sum(old.values())
    return {**{s: Fraction(4, 5)*v/total for s, v in old.items()},
            **{s: Fraction(1, 5)/len(NEW_SOURCES) for s in NEW_SOURCES}}


def layout(counts, family_counts):
    """Keep replay once, find bounded equal-per-source virtual repeats.

    Multiplicity depends only on source counts and fixed loss masses. Every
    distinct original in the same new source gets the same multiplicity.
    A missing/sparse source fails rather than changing the proposed task mix.
    """
    mass = masses()
    if (len(NEW_SOURCES) != 32 or len(mass) != 57 or set(counts) != set(mass)
            or set(family_counts) != set(mass)
            or any(type(n) is not int or n <= 0 for n in counts.values())
            or any(type(n) is not int or not 0 < n <= counts[s] for s, n in family_counts.items())
            or sum(counts[s] for s in REPLAY_MASS) != RECORDS):
        raise ValueError('Complete57-source unique TRAIN counts, families and exact3B replay required')
    if any(family_counts[s] < 2 for s in NEW_SOURCES):
        raise ValueError('More independent TRAIN families needed; copies cannot repair coverage')
    repeats = dict.fromkeys(mass, 1)
    for _ in range(128):
        total = sum(counts[s]*repeats[s] for s in mass)
        changed = False
        for source in NEW_SOURCES:
            ratio = mass[source]*total/(counts[source]*MAX_WEIGHT)
            required = max(repeats[source], -(-ratio.numerator//ratio.denominator))
            if required > MAX_REPEATS:
                raise ValueError('More unique TRAIN data needed; repeat cap8 cannot support this mix')
            changed |= required != repeats[source]
            repeats[source] = required
        if not changed:
            break
    else:
        raise ValueError('Count-only repeat layout did not converge')
    expanded = {s: counts[s]*repeats[s] for s in mass}
    total = sum(expanded.values())
    exact_weights = {s: mass[s]*total/expanded[s] for s in mass}
    if any(v <= 0 or v > MAX_WEIGHT for v in exact_weights.values()):
        raise ValueError('Replay count layout exceeds scalar cap; do not silently duplicate replay')
    if sum(mass.values()) != 1 or sum(exact_weights[s]*expanded[s]/total for s in mass) != 1:
        raise AssertionError('Normalized exact-rational source loss accounting failed')
    return dict(method=METHOD, proposed_loss_mass={s: float(v) for s, v in mass.items()},
        unique_train_records=sum(counts.values()), unique_train_records_by_source=dict(counts),
        distinct_train_families_by_source=dict(family_counts),
        virtual_repeats_per_original_by_source=repeats, virtual_exposures_per_pass=total,
        virtual_exposures_by_source=expanded, extra_virtual_exposures=total-sum(counts.values()),
        scalar_example_weights={s: float(v) for s, v in exact_weights.items()},
        replay_loss_mass=.8, new_loss_mass=.2, equal_new_source_mass=1/160,
        maximum_scalar_example_weight=MAX_WEIGHT, maximum_repeats_per_unique=MAX_REPEATS,
        original_records_unchanged=True, validation_repeated=False,
        all_sources_in_every_optimizer_update_claimed=False,
        copies_are_not_independent_samples=True, virtual_expansion_materialized=False,
        parent_optimizer_and_update_budget_registered=False, student_fit_admitted=False)

