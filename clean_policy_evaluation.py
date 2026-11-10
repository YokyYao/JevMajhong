"""Whole-wall paired net-point aggregation for the clean-policy comparison."""
import math
import statistics

from guide_development_audit import interval

POLICIES = ('reference', 'candidate')
SUITES = ('closed', 'mixed', 'strong')


def aggregate(rows, cfg):
    seeds = range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])
    expected = {(s, n, p, k) for s in SUITES for n in seeds for p in range(4) for k in POLICIES}
    mapping = {(r['suite'], r['seed'], r['seat'], r['policy']): r for r in rows}
    if len(mapping) != len(rows) or set(mapping) != expected:
        raise ValueError('Missing, foreign, or duplicate full-East game')
    for row in rows:
        if row['rank'] not in (1, 2, 3, 4) or row['hands'] < 1 or not math.isfinite(row['score']):
            raise ValueError('Invalid terminal score, rank, or hands')
        if not row['settlement_ledger_passed'] or sum(row['point_components'].values()) != row['score'] - 25000:
            raise ValueError('Terminal net points differ from audited ledger')
        if not 0 <= row['hand_wins'] <= row['hands'] or not 0 <= row['dealins'] <= row['hands']:
            raise ValueError('Impossible win or deal-in count')

    def stats(selected):
        hands = sum(r['hands'] for r in selected)
        return {'games': len(selected), 'mean_net_points': statistics.mean(r['score']-25000 for r in selected),
                'first_rate': statistics.mean(r['rank']==1 for r in selected),
                'top2_rate': statistics.mean(r['rank']<=2 for r in selected),
                'fourth_rate': statistics.mean(r['rank']==4 for r in selected),
                'mean_rank': statistics.mean(r['rank'] for r in selected),
                'hand_win_rate': sum(r['hand_wins'] for r in selected)/hands,
                'dealin_rate': sum(r['dealins'] for r in selected)/hands,
                'calls_per_game': statistics.mean(r['calls'] for r in selected),
                'mean_point_components': {k:statistics.mean(r['point_components'].get(k,0) for r in selected)
                    for k in sorted({k for r in selected for k in r['point_components']})}}

    def comparison(suites):
        points = [statistics.mean(mapping[s,n,p,'candidate']['score']-mapping[s,n,p,'reference']['score']
                    for s in suites for p in range(4)) for n in seeds]
        return {'gain':statistics.mean(points), 'ci95':interval(points), 'independent_wall_streams':len(points)}

    suites = {s:{'policies':{k:stats([r for r in rows if r['suite']==s and r['policy']==k]) for k in POLICIES},
                'candidate_minus_reference':comparison([s])} for s in SUITES}
    overall = {'policies':{k:stats([r for r in rows if r['policy']==k]) for k in POLICIES},
               'candidate_minus_reference':comparison(SUITES)}
    primary = overall['candidate_minus_reference']
    return {'games':len(rows), 'suites':suites, 'overall':overall,
            'primary':'equal-weight three opponent suites, aggregate four seats within whole wall before CI',
            'positive_signal':primary['gain']>0, 'ci_excludes_zero_positive':primary['ci95'][0]>0,
            'stage':cfg['stage'], 'not_real_player_win_rate':True, 'deployed':False}
