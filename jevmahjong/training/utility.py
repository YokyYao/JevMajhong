"""Relabel the same intervention hand with fast-win and risk utility."""
import math
from .rewards import fast_reward, hand_outcomes


def validate_config(config):
    if config['mode'] not in ('risk_adjusted_fast_win_v1', 'net_points_with_fast_support_v1', 'own_win_first_v1'):
        raise ValueError('Unsupported training utility')
    for name in ('dealin_penalty', 'net_weight', 'point_scale'):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError('Finite nonnegative utility weights required')
    if config['point_scale'] == 0:
        raise ValueError('Positive point scale required')
    if config['mode'] in ('net_points_with_fast_support_v1','own_win_first_v1'):
        if not math.isfinite(config['fast_weight']) or config['fast_weight'] < 0:
            raise ValueError('Finite nonnegative fast-win weight required')
    if config['mode']=='own_win_first_v1':
        if config['net_weight']!=0 or config['win_weight']!=1 or not 0<config['speed_gamma']<1:
            raise ValueError('Win-first utility requires unit own-win credit, zero point reward and a valid speed discount')


def hand_utility(log, hand_index, seat, draws_before, gamma, config):
    validate_config(config)
    hands = hand_outcomes(log)
    if not 0 <= hand_index < len(hands):
        raise ValueError('Intervention hand missing')
    fast = fast_reward(hands[hand_index], seat, draws_before,
        config['speed_gamma'] if config['mode']=='own_win_first_v1' else gamma)
    index = -1; points = 0; dealin = False; settled = False
    for event in log:
        kind = event['type']
        if kind == 'start_kyoku': index += 1
        if index != hand_index: continue
        if kind == 'reach_accepted' and event['actor'] == seat:
            points -= 1000
        elif kind in ('hora', 'ryukyoku'):
            if len(event['deltas']) != 4:
                raise ValueError('Four-seat settlement required')
            points += event['deltas'][seat]; settled = True
            if kind == 'hora' and event['target'] == seat and event['actor'] != seat:
                dealin = True
        elif kind == 'end_kyoku': break
    if not settled:
        raise ValueError('Completed hand settlement required')
    if config['mode'] == 'risk_adjusted_fast_win_v1':
        utility = fast - config['dealin_penalty'] * dealin + config['net_weight'] * math.tanh(points / config['point_scale'])
    elif config['mode']=='net_points_with_fast_support_v1':
        utility = config['fast_weight'] * fast - config['dealin_penalty'] * dealin + config['net_weight'] * points / config['point_scale']
    else:
        own_win=float(any(w['actor']==seat for w in hands[hand_index]['winners']))
        utility=own_win+config['fast_weight']*fast-config['dealin_penalty']*dealin
        return {'utility':utility,'fast_reward':fast,'dealin':dealin,'net_points':points,'own_win':own_win}
    return {'utility': utility, 'fast_reward': fast, 'dealin': dealin, 'net_points': points}


def relabel(payload, source_config, config):
    controls = {g['seat']: g for g in payload['controls']}
    games = {(g['seed'], g['seat'], g['replica']): g for g in payload['games']}
    rows = []
    for original in payload['rows']:
        row = dict(original); _, seat, _ = row['episode_key']
        options = (row['hand_index'], seat, row['draws_before'], source_config['fast_gamma'], config)
        control = hand_utility(controls[seat]['mjai_log'], *options)
        branch = hand_utility(games[tuple(row['episode_key'])]['mjai_log'], *options)
        row.update(training_advantage=branch['utility'] - control['utility'],
            training_credit={'control': control, 'branch': branch}, training_target=config['mode'])
        rows.append(row)
    return rows
