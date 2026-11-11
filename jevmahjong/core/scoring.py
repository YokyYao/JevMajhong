"""Settlement ledger and an independent hand-score cross-check for native replays.

Raw observations and MJAI logs are private. Unsupported exceptional wins are
reported explicitly; ledger checks still include every event in those hands.
"""
from collections import Counter


def settlement_ledger(log, final_scores):
    scores, pool = [25000] * 4, 0
    components = [Counter() for _ in range(4)]
    hands = wins = deposits = 0
    for index, event in enumerate(log):
        kind = event['type']
        if kind == 'start_kyoku':
            if event['scores'] != scores or event['kyotaku'] != pool:
                raise ValueError(f'Hand-boundary ledger mismatch at event {index}')
            hands += 1
        elif kind == 'reach_accepted':
            p = event['actor']; scores[p] -= 1000; pool += 1; deposits += 1
            components[p]['riichi_deposits'] -= 1000
        elif kind in ('hora', 'ryukyoku'):
            delta = event['deltas']
            if len(delta) != 4: raise ValueError('Incomplete settlement')
            if kind == 'hora':
                if sum(delta) != pool * 1000:
                    raise ValueError(f'Win settlement / deposit-pool mismatch at {index}: {delta}, pool={pool}')
                pool = 0; wins += 1
            elif sum(delta) != 0:
                raise ValueError(f'Nonconserving drawn-hand settlement at {index}')
            for p, change in enumerate(delta):
                scores[p] += change
                category = ('draw_settlement' if kind == 'ryukyoku' else 'win_income' if p == event['actor']
                            else 'tsumo_payments' if event['actor'] == event['target'] else 'ron_payments')
                components[p][category] += change
        elif kind == 'end_game' and pool:
            # RiichiEnv v0.4.11 state/mod.rs::_process_end_game awards an
            # unclaimed pool to the leader (lower seat wins a tied score).
            # Its MJAI end_game event omits this delta; reconstruct the rule
            # explicitly rather than accepting an arbitrary terminal residual.
            leader = max(range(4), key=lambda p: (scores[p], -p))
            scores[leader] += pool * 1000
            components[leader]['final_unclaimed_pool'] += pool * 1000
            pool = 0
        if sum(scores) + pool * 1000 != 100000:
            raise ValueError(f'Points not conserved at event {index}')
    if scores != list(final_scores):
        raise ValueError(f'Terminal ledger mismatch: ledger={scores}, engine={final_scores}, pool={pool}')
    return {'hands': hands, 'wins': wins, 'riichi_deposits': deposits,
            'unclaimed_riichi_sticks': pool, 'final_scores': scores,
            'components': [dict(c) for c in components]}


def _indicator(tile):
    if tile in ('E', 'S', 'W', 'N', 'P', 'F', 'C'):
        return (27 + ('E', 'S', 'W', 'N', 'P', 'F', 'C').index(tile)) * 4
    return ('mps'.index(tile[1]) * 9 + int(tile[0]) - 1) * 4


def independent_win_score(history, event, snapshot, simultaneous_wins=1):
    """Compare all four payments using MahjongRepository/mahjong, not RiichiEnv.

    Hand IDs and open melds come from pre-win observations. Situational yaku
    come from event order; no han/fu or score prediction is read from the engine.
    Kan, first-turn wins and simultaneous ron are conservatively out of scope.
    """
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.meld import Meld

    last_start = max(i for i, e in enumerate(history) if e['type'] == 'start_kyoku')
    hand_log = history[last_start:]
    calls = {'chi', 'pon', 'daiminkan', 'ankan', 'kakan'}
    if simultaneous_wins != 1: return {'status': 'skipped', 'reason': 'simultaneous_ron'}
    if any(e['type'] in {'daiminkan', 'ankan', 'kakan'} for e in hand_log):
        return {'status': 'skipped', 'reason': 'kan_hand_special_rules'}
    actor, target = event['actor'], event['target']
    if not any(e['type'] == 'dahai' and e['actor'] == actor for e in hand_log):
        return {'status': 'skipped', 'reason': 'first_turn_win'}
    tsumo = actor == target
    obs, win_tile = snapshot['observation'], snapshot['win_tile']
    if obs['player_id'] != actor: raise ValueError('Winner observation identity mismatch')
    tiles = list(obs['hands'][actor])
    if not tsumo: tiles.append(win_tile)
    melds = []
    for meld in obs['melds'][actor]:
        kind = {'Chi': Meld.CHI, 'Pon': Meld.PON}[meld['meld_type']]
        melds.append(Meld(kind, meld['tiles'], opened=meld['opened']))
        tiles.extend(meld['tiles'])
    reach_indices = [i for i, e in enumerate(hand_log) if e['type'] == 'reach' and e['actor'] == actor]
    riichi = obs['riichi_declared'][actor]
    if bool(reach_indices) != riichi: raise ValueError('Riichi event/observation mismatch')
    double = ippatsu = False
    if riichi:
        r = reach_indices[-1]
        double = not any(e['type'] in calls or (e['type'] == 'dahai' and e['actor'] == actor)
                         for e in hand_log[:r])
        after = hand_log[r + 1:]
        ippatsu = (not any(e['type'] in calls for e in after)
                   and sum(e['type'] == 'dahai' and e['actor'] == actor for e in after) == 1)
    last_draw = sum(e['type'] == 'tsumo' for e in hand_log) == 70
    config = HandConfig(is_tsumo=tsumo, is_riichi=riichi and not double,
        is_daburu_riichi=double, is_ippatsu=ippatsu,
        is_haitei=last_draw and tsumo, is_houtei=last_draw and not tsumo,
        player_wind=27 + (actor - obs['oya']) % 4, round_wind=27 + obs['round_wind'],
        tsumi_number=obs['honba'], kyoutaku_number=obs['riichi_sticks'],
        options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))
    value = HandCalculator().estimate_hand_value(tiles, win_tile, melds=melds,
        dora_indicators=obs['dora_indicators'],
        ura_dora_indicators=[_indicator(t) for t in event.get('ura_markers', [])], config=config)
    if value.error:
        return {'status': 'mismatch', 'reason': value.error,
                'flags': {'tsumo': tsumo, 'riichi': riichi, 'double_riichi': double,
                          'ippatsu': ippatsu, 'last_draw': last_draw}}
    cost = value.cost; expected = [0] * 4
    for p in range(4):
        if p == actor: continue
        if tsumo:
            expected[p] = -(cost['main'] + cost['main_bonus'] if p == obs['oya'] or actor == obs['oya']
                            else cost['additional'] + cost['additional_bonus'])
        elif p == target: expected[p] = -(cost['main'] + cost['main_bonus'])
    expected[actor] = -sum(expected) + cost['kyoutaku_bonus']
    return {'status': 'passed' if expected == event['deltas'] else 'mismatch',
            'expected': expected, 'actual': event['deltas'], 'han': value.han, 'fu': value.fu,
            'yaku': [str(y) for y in value.yaku], 'open': any(m.opened for m in melds)}
