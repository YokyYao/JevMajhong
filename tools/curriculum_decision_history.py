"""Public history constraints for live draw, response and post-call decisions.

The prior is deliberately restricted to opponents with no declined riichi ron
in their generated history. Non-riichi temporary furiten is reconstructed
since their latest discard. This is not a calibrated action-history posterior.
"""
import json
import random

from tools.curriculum_win_context import public_tile
from tools.curriculum_public_worlds import sample_allocations, SamplingExhausted

METHOD='public_inventory_live_decision_missed_win_history_v2'
LIVE_TRIGGERS=('dahai','tsumo','kakan','ankan','chi','pon')
ORPHANS={0,8,9,17,18,26,27,28,29,30,31,32,33}


def tile_type(tile):
    value=public_tile(tile)
    return 'mpsz'.index(value[1])*9+(5 if value[0]=='0' else int(value[0]))-1


def public_history(events):
    parsed=[json.loads(e) if isinstance(e,str) else e for e in events]
    starts=[i for i,e in enumerate(parsed) if e.get('type')=='start_kyoku']
    if not starts:raise ValueError('A verified current-hand public prefix is required')
    result=[]
    for event in parsed[starts[-1]:]:
        kind=event.get('type')
        if kind=='start_kyoku':result.append(dict(type=kind));continue
        if kind in ('hora','ryukyoku','end_kyoku','end_game'):
            raise ValueError('A live decision prefix cannot contain a settled current hand')
        if kind=='dora':result.append(dict(type=kind));continue
        if kind not in ('tsumo','dahai','reach','reach_accepted','chi','pon','ankan','kakan','daiminkan'):
            raise ValueError('Unsupported current-hand event')
        actor=event.get('actor')
        if type(actor) is not int or actor not in range(4):raise ValueError('Invalid public actor')
        projected=dict(type=kind,actor=actor)
        if kind in ('dahai','kakan'):
            projected['tile_type']=tile_type(event['pai'])
        elif kind=='ankan':
            consumed=event.get('consumed')
            if not isinstance(consumed,list) or len(consumed)!=4 or len({tile_type(t) for t in consumed})!=1:
                raise ValueError('Invalid public ankan identity')
            projected['tile_type']=tile_type(consumed[0])
        if kind=='dahai':
            if type(event.get('tsumogiri')) is not bool:raise ValueError('Missing public discard provenance')
            projected['tsumogiri']=event['tsumogiri']
        # Own/opponent draw identities and start-hand concealed tiles are never copied.
        result.append(projected)
    if result[-1]['type'] not in LIVE_TRIGGERS:
        raise ValueError('Prefix does not end at a supported live decision')
    return result


def kokushi_completion(hand,tile):
    types=[t//4 for t in hand]+[tile]
    return len(types)==14 and set(types)==ORPHANS


def history_flags(history,own_seat,riichi,allocation,waits):
    """waits(player, proposed_hand) returns native34-class winning waits."""
    if len(riichi)!=4 or any(type(v) is not bool for v in riichi):raise ValueError('Explicit riichi flags required')
    if type(own_seat) is not int or own_seat not in range(4):raise ValueError('Invalid own seat')
    if not history or history[0]!={'type':'start_kyoku'} or history[-1]['type'] not in LIVE_TRIGGERS:
        raise ValueError('A projected current-hand live prefix is required')
    hands=allocation['opponent_hands']
    if set(hands)!=set(range(4))-{own_seat}:raise ValueError('All three proposed hands are required')
    flags={}
    rejected=[]
    for player,hand in hands.items():
        waiting=set(waits(player,hand))
        if any(type(t) is not int or t not in range(34) for t in waiting):raise ValueError('Invalid native wait classes')
        if riichi[player] and not waiting:
            rejected.append(dict(player=player,reason='declared_riichi_not_tenpai'));continue
        own_river=set()
        temporary=False
        reached=False
        needs_unavailable_riichi_missed_flag=False
        # The final trigger is still being answered, so it is not a missed ron.
        # A source player's current discard does clear that player's temporary flag.
        for index,event in enumerate(history):
            kind,actor=event['type'],event.get('actor')
            if kind in ('reach','reach_accepted') and actor==player:reached=True
            if kind=='dahai' and actor==player:
                own_river.add(event['tile_type']);temporary=False
                continue
            if index==len(history)-1:continue
            if actor==player or kind not in ('dahai','kakan','ankan'):continue
            tile=event['tile_type']
            if tile not in waiting or waiting&own_river:continue
            if kind=='ankan' and not kokushi_completion(hand,tile):continue
            if riichi[player]:
                if reached:needs_unavailable_riichi_missed_flag=True
            elif not temporary:
                temporary=True
        if riichi[player] and not reached:
            raise ValueError('Declared riichi is missing from the public prefix')
        if needs_unavailable_riichi_missed_flag:
            rejected.append(dict(player=player,reason='hypothesis_would_require_past_riichi_ron_decline'))
        flags[player]=False if riichi[player] else temporary
    return dict(compatible=not rejected,opponent_temporary_furiten=flags,rejections=rejected,
        restricted_no_missed_riichi_prior=True,full_public_action_likelihood_model=False)


def sample_history_allocations(snapshot,history,riichi,seed,count,max_attempts,shanten,waits):
    if type(seed) is not int or type(count) is not int or not 1<=count<=128 or type(max_attempts) is not int or not count<=max_attempts<=2000000:
        raise ValueError('Explicit bounded history-prior budget required')
    rng=random.Random(seed);samples=[];attempts=0;rejected=0
    while attempts<max_attempts:
        try:
            drawn=sample_allocations(snapshot,rng.getrandbits(64),1,max_attempts-attempts,shanten)
        except SamplingExhausted as error:
            attempts+=error.attempts
            break
        attempts+=drawn['attempts']
        candidate=drawn['allocations'][0]
        flags=history_flags(history,snapshot['own_seat'],riichi,candidate,waits)
        if not flags['compatible']:
            rejected+=1;continue
        candidate['opponent_temporary_furiten']=flags['opponent_temporary_furiten']
        samples.append(candidate)
        if len(samples)==count:
            return dict(method=METHOD,seed=seed,attempts=attempts,history_rejected_tenpai_allocations=rejected,
                allocations=samples,requested=count,calibrated_public_history_posterior=False,
                native_environment_constructed=False,training_labels=False)
    raise SamplingExhausted(attempts,len(samples),count)
