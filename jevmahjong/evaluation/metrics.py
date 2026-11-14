"""Whole-wall paired net-point aggregation for the clean-policy comparison."""
import math
import statistics



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
        if not 0 <= row['hand_wins'] <= row['hands'] or row['dealins'] < 0:
            raise ValueError('Impossible win or deal-in count')

    def stats(selected):
        hands = sum(r['hands'] for r in selected)
        return {'games': len(selected), 'mean_net_points': statistics.mean(r['score']-25000 for r in selected),
                'first_rate': statistics.mean(r['rank']==1 for r in selected),
                'top2_rate': statistics.mean(r['rank']<=2 for r in selected),
                'fourth_rate': statistics.mean(r['rank']==4 for r in selected),
                'mean_rank': statistics.mean(r['rank'] for r in selected),
                'hand_win_rate': sum(r['hand_wins'] for r in selected)/hands,
                'dealin_rate': sum(hand_dealin_count(r['mjai_log'], r['seat']) for r in selected)/hands,
                'dealin_event_rate': sum(r['dealins'] for r in selected)/hands,
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

def interval(values):
    from scipy.stats import t
    if len(values) < 2: raise ValueError('Multiple independent walls required')
    mean = statistics.mean(values)
    half = float(t.ppf(.975,len(values)-1)) * statistics.stdev(values) / math.sqrt(len(values))
    return [mean-half,mean+half]


from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.training.rewards import fast_game_stats


def hand_dealin_count(log, seat):
    count = 0; hit = False
    for event in log:
        if event['type'] == 'start_kyoku': hit = False
        elif event['type'] == 'hora' and event['target'] == seat and event['actor'] != seat: hit = True
        elif event['type'] == 'end_kyoku': count += int(hit)
    return count

def summarize(rows, cfg):
    result = aggregate(rows, cfg)
    indexed = {(r['suite'], r['seed'], r['seat'], r['policy']): r for r in rows}
    for r in rows:
        ledger = settlement_ledger(r['mjai_log'], r['final_scores'])
        actual = fast_game_stats(r['mjai_log'], r['seat'], cfg['fast_gamma'])
        if (ledger['final_scores'][r['seat']] != r['score']
                or ledger['components'][r['seat']] != r['point_components']
                or any(abs(actual[k] - r[k]) > 1e-12 for k in actual)):
            raise ValueError('Reward or point summary differs from actual native event log')
        if r['fast_hand_count'] != r['hands'] or r['fast_hand_wins'] != r['hand_wins']:
            raise ValueError('Fast reward and arena hand counts differ')
        if r['dealins'] != sum(e['type'] == 'hora' and e['target'] == r['seat'] and e['actor'] != r['seat'] for e in r['mjai_log']):
            raise ValueError('Ron payment event count differs')
        if not 0 <= r['fast_reward_sum'] <= r['hands']:
            raise ValueError('Invalid discounted wins')
    gains = [statistics.mean(indexed[s, n, p, 'candidate']['fast_reward_sum'] / indexed[s, n, p, 'candidate']['hands']
        - indexed[s, n, p, 'reference']['fast_reward_sum'] / indexed[s, n, p, 'reference']['hands']
        for s in SUITES for p in range(4)) for n in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])]
    result['fast_reward_paired'] = {'gain': statistics.mean(gains), 'ci95': interval(gains),
        'unit': 'discounted own wins per hand, equal-weight match means, whole-wall paired CI'}
    for name in POLICIES:
        subset = [r for r in rows if r['policy'] == name]; wins = sum(r['hand_wins'] for r in subset)
        hands = sum(r['hands'] for r in subset)
        result['overall']['policies'][name].update(
            dealin_event_rate=sum(r['dealins'] for r in subset) / hands,
            dealin_rate=sum(hand_dealin_count(r['mjai_log'], r['seat']) for r in subset) / hands,
            discounted_own_wins_per_hand=statistics.mean(r['fast_reward_sum'] / r['hands'] for r in subset),
            mean_table_draws_when_winning=sum(r['winning_table_draws'] for r in subset) / wins if wins else None,
            kan_opportunities={k: sum(r['kan_opportunities'].get(k, 0) for r in subset) for k in ('ankan', 'daiminkan', 'kakan')},
            kan_choices={k: sum(r['kan_choices'].get(k, 0) for r in subset) for k in ('ankan', 'daiminkan', 'kakan')})
    objective = cfg.get('confirmation_objective', 'joint_fast_points')
    if objective not in ('joint_fast_points', 'net_points', 'hand_win'):
        raise ValueError('Unknown confirmation objective')
    result['eligible_for_confirmation'] = (result['overall']['candidate_minus_reference']['gain'] > 0
        and (objective == 'net_points' or result['fast_reward_paired']['gain'] > 0))
    if objective=='hand_win':
        def paired_rate(field):
            values=[]
            for n in range(cfg['seed_start'],cfg['seed_start']+cfg['seeds']):
                differences=[]
                for s in SUITES:
                    for p in range(4):
                        a,b=indexed[s,n,p,'candidate'],indexed[s,n,p,'reference']
                        numerator=lambda r:r['hand_wins'] if field=='wins' else hand_dealin_count(r['mjai_log'],r['seat'])
                        differences.append(numerator(a)/a['hands']-numerator(b)/b['hands'])
                values.append(statistics.mean(differences))
            return dict(gain=statistics.mean(values),ci95=interval(values),
                unit='equal-weight match-level per-hand rates, paired by whole wall')
        result['hand_win_paired']=paired_rate('wins');result['dealin_rate_paired']=paired_rate('dealin')
        for name in POLICIES:
            subset=[r for r in rows if r['policy']==name]
            result['overall']['policies'][name]['mean_match_hand_win_rate']=statistics.mean(r['hand_wins']/r['hands'] for r in subset)
        limit=cfg['max_dealin_rate_increase']
        if not math.isfinite(limit) or not 0<=limit<=1:raise ValueError('Valid predeclared deal-in constraint required')
        result['primary']='own hand win-rate gain, equal-weight matches and whole-wall paired CI; speed and net points reported separately'
        result['positive_signal']=result['hand_win_paired']['gain']>0
        result['ci_excludes_zero_positive']=result['hand_win_paired']['ci95'][0]>0
        result['eligible_for_confirmation']=result['positive_signal'] and result['dealin_rate_paired']['gain']<=limit
    if 'confirmation_objective' in cfg: result['confirmation_objective'] = objective
    result['not_deployed'] = True
    return result
