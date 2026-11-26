"""Keep the registered curve16 subset separate from a full stage-end64 result."""
import math


def require_aggregate(actual, expected):
    for key, value in actual.items():
        other = expected[key]
        if type(value) in (int, float):
            if (type(other) not in (int, float) or not math.isfinite(other)
                    or not math.isclose(value, other, rel_tol=0, abs_tol=1e-10)):
                raise ValueError('Native, summary and journal aggregates differ')
        elif value != other:
            raise ValueError('Aggregate accounting definitions differ')


def source_groups(curriculum, step):
    if (curriculum['curve_seed_groups'] != 2 or curriculum['stage_end_seed_groups'] != 8
            or step not in curriculum['validation_and_game_steps']):
        raise ValueError('Use only the registered curve16 and endpoint64 development grids')
    groups = 8 if step == curriculum['final_optimizer_steps'] else 2
    seeds = range(curriculum['curve_seed_start'], curriculum['curve_seed_start']+groups)
    if any(2026300000 <= seed < 2026300032 for seed in seeds):
        raise ValueError('Reserved final seeds cannot enter development reports')
    return groups


def project_step(curriculum, step, protocol, summary, rows, journal):
    """Validate the entire original grid, then take its preregistered first seeds.

    The caller must also check native score ledgers and actual parameter files.
    This function generates neither matches nor predictions.
    """
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    groups = source_groups(curriculum, step); start = curriculum['curve_seed_start']
    seeds = list(range(start, start+groups)); profiles = curriculum['opponent_profiles']
    expected = {(seed, profile, seat) for seed in seeds for profile in profiles for seat in range(4)}
    grid = 'stage_end64' if groups == 8 else 'curve16'
    if (profiles != ['efficiency', 'calling'] or protocol['seeds'] != seeds
            or protocol['profiles'] != profiles or protocol['seats'] != 4 or protocol['groups'] != groups
            or len(rows) != groups*8 or {(r['seed'], r['profile'], r['seat']) for r in rows} != expected
            or any(r['completed'] is not True or r['ledger_passed'] is not True for r in rows)
            or summary['matches'] != groups*8 or summary['seed_groups'] != groups
            or summary['checkpoint_sha256'] != protocol['checkpoint']['identity_sha256']
            or summary['policy_identity'] != protocol['policy_identity']
            or journal['checkpoint_sha256'] != summary['checkpoint_sha256']
            or journal['policy_identity'] != summary['policy_identity'] or journal['grid'] != grid):
        raise ValueError('Exact complete original development grid, policy and weights required')
    full = aggregate_complete_matches(rows)
    require_aggregate(full, summary); require_aggregate(full, journal)
    subset = [r for r in rows if start <= r['seed'] < start+2]
    curve = aggregate_complete_matches(subset)
    if curve['matches'] != 16 or curve['seed_groups'] != 2:
        raise ValueError('The actual preregistered first-two-seed curve subset is incomplete')
    if groups == 8:
        require_aggregate(curve, summary['curve_grid_subset'])
    return dict(journal, **curve, grid='curve16', original_grid=grid,
        original_matches=full['matches'], original_seed_groups=full['seed_groups'],
        curve16_projected_from_completed64=(groups == 8)), (full if groups == 8 else None)
