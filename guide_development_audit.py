"""Independent aggregation for three-suite, whole-wall point development."""
import math
import statistics

POLICIES = ('reference','previous','candidate')
SUITES = ('closed','mixed','strong')


def interval(values):
    from scipy.stats import t
    if len(values) < 2: raise ValueError('Multiple independent walls required')
    mean = statistics.mean(values)
    half = float(t.ppf(.975,len(values)-1)) * statistics.stdev(values) / math.sqrt(len(values))
    return [mean-half,mean+half]


def aggregate(rows, cfg):
    seeds = list(range(cfg['development_seed_start'],cfg['development_seed_start']+cfg['development_seeds']))
    expected = {(suite,seed,seat,name) for suite in SUITES for seed in seeds for seat in range(4) for name in POLICIES}
    mapping = {(r['suite'],r['seed'],r['seat'],r['policy']):r for r in rows}
    if len(mapping) != len(rows) or set(mapping) != expected: raise ValueError('Duplicate, missing or foreign game')
    for row in rows:
        if row['rank'] not in (1,2,3,4) or row['hands'] < 1: raise ValueError('Invalid terminal game')
        if not row['settlement_ledger_passed'] or sum(row['point_components'].values()) != row['score']-25000:
            raise ValueError('Score/point-component ledger mismatch')
        if not 0 <= row['hand_wins'] <= row['hands'] or not 0 <= row['dealins'] <= row['hands']:
            raise ValueError('Invalid win/deal-in count')
    def stats(subset):
        hands = sum(r['hands'] for r in subset)
        components = {k:sum(r['point_components'].get(k,0) for r in subset)/len(subset)
                      for k in sorted({k for r in subset for k in r['point_components']})}
        return {'games':len(subset),'mean_net_points':statistics.mean(r['score']-25000 for r in subset),
                'first_rate':statistics.mean(r['rank']==1 for r in subset),
                'top2_rate':statistics.mean(r['rank']<=2 for r in subset),
                'fourth_rate':statistics.mean(r['rank']==4 for r in subset),
                'mean_rank':statistics.mean(r['rank'] for r in subset),
                'hand_win_rate':sum(r['hand_wins'] for r in subset)/hands,
                'dealin_rate':sum(r['dealins'] for r in subset)/hands,
                'calls_per_game':sum(r['calls'] for r in subset)/len(subset),
                'mean_point_components':components}
    def comparison(suites, left, right):
        wall_means = [statistics.mean(mapping[s,seed,p,left]['score']-mapping[s,seed,p,right]['score']
                                     for s in suites for p in range(4)) for seed in seeds]
        return {'gain':statistics.mean(wall_means),'ci95':interval(wall_means),'independent_wall_streams':len(seeds)}
    suite_results = {s:{'policies':{n:stats([r for r in rows if r['suite']==s and r['policy']==n]) for n in POLICIES},
                        'candidate_minus_reference':comparison([s],'candidate','reference'),
                        'candidate_minus_previous':comparison([s],'candidate','previous')} for s in SUITES}
    overall = {'policies':{n:stats([r for r in rows if r['policy']==n]) for n in POLICIES},
               'candidate_minus_reference':comparison(SUITES,'candidate','reference'),
               'candidate_minus_previous':comparison(SUITES,'candidate','previous')}
    primary = overall['candidate_minus_reference']
    return {'games':len(rows),'suites':suite_results,'overall':overall,
            'primary':'equal-weighted three-suite paired terminal net points; aggregate within logical wall before CI',
            'development_positive_signal':primary['gain']>0,
            'meets_development_stability_condition':primary['ci95'][0]>0
                and overall['policies']['candidate']['mean_net_points']>=0
                and suite_results['strong']['candidate_minus_reference']['gain']>=0,
            'not_final_confirmation':True,'deployed':False}
