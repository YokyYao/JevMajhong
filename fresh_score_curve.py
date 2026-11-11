"""Rank saved continuation checkpoints by matched complete-East net points."""
from clean_policy_evaluation import aggregate, SUITES


def policy_name(step):
    if not isinstance(step, int) or step <= 0: raise ValueError('Positive checkpoint step required')
    return f'step-{step:06d}'


def summarize(rows, cfg, steps):
    steps = sorted(steps)
    if not steps or len(set(steps)) != len(steps): raise ValueError('Distinct updated checkpoints required')
    confirmation = cfg['stage'] == 'independent_confirmation'
    if confirmation and len(steps) != 1: raise ValueError('Confirmation must evaluate one frozen development winner')
    names = ['parent', *(policy_name(s) for s in steps)]
    seeds = range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])
    expected = {(suite, seed, seat, name) for suite in SUITES for seed in seeds
        for seat in range(4) for name in names}
    actual = {(r['suite'], r['seed'], r['seat'], r['policy']) for r in rows}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError('Missing, duplicate or foreign score-curve game')
    comparisons = {}
    for step in steps:
        name = policy_name(step)
        paired = [dict(r, policy='reference' if r['policy'] == 'parent' else 'candidate')
            for r in rows if r['policy'] in ('parent', name)]
        comparisons[str(step)] = aggregate(paired, cfg)
    # Each comparison reuses the same parent games. Do not run the parent again
    # for another checkpoint, and do not count copies as independent evidence.
    best = max(steps, key=lambda s: (comparisons[str(s)]['overall']['candidate_minus_reference']['gain'], -s))
    best_gain = comparisons[str(best)]['overall']['candidate_minus_reference']['gain']
    return {'games': len(rows), 'policies': names, 'steps': steps,
        'parent_games_computed_once': cfg['seeds'] * 4 * len(SUITES),
        'comparisons': comparisons, 'best_step': best,
        'confirmation_step': best if best_gain > 0 and not confirmation else None,
        'evaluated_confirmation_step': best if confirmation else None,
        'selection': ('frozen development winner; no reselection' if confirmation else
            'highest complete-development paired mean net points; ties choose earliest step'),
        'validation_nll_used_to_rank': False, 'multiple_candidates_compared': len(steps),
        'individual_development_intervals_not_selection_adjusted': not confirmation,
        'requires_independent_confirmation': not confirmation, 'deployed': False}
