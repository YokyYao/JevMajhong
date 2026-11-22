"""Native v2 worlds for ordinary draw, discard response and post-call choices.

For the pinned RiichiEnv0.4.11. Current robbed-kan windows are not yet admitted.
Sampled opponents require the no-prior-declined-legal-win certificate. The
observer's own known prior declines are allowed and retained by cloning.
Current robbed-kan choices remain excluded. No active fit uses this.
"""
from collections import Counter
from importlib.metadata import version

from tools.curriculum_decision_history import public_history
from tools.curriculum_win_context import public_tile
from tools.curriculum_native_worlds import refresh_discard_claims


def decision_scope(source,own_obs,trigger):
    from riichienv import Phase
    if own_obs.player_id not in source.active_players:raise ValueError('Observer is not an active decision maker')
    kind=trigger.get('type')
    if source.phase==Phase.WaitResponse and kind=='dahai' and trigger.get('actor')!=own_obs.player_id:return 'response'
    if source.phase==Phase.WaitAct and trigger.get('actor')==own_obs.player_id:
        if kind=='tsumo':return 'draw'
        if kind in ('chi','pon'):return 'post_call'
    raise ValueError('Current kan/robbed-kan or other unverified decision scope')


def decision_inventory(own_obs,scope):
    from tools.curriculum_public_inventory import from_observation
    if scope not in ('draw','post_call','response'):raise ValueError('Explicit verified current decision scope required')
    # Reuse only the physical13/14-minus-meld count/inventory contract. Its
    # legacy schema name is not a certificate that this non-win can win.
    result=from_observation(own_obs,'ron' if scope=='response' else 'tsumo')
    result.update(decision_scope=scope,inventory_is_not_a_win_certificate=True)
    return result


def construct_world(source,own_obs,trigger,allocation,opponents_no_declined_wins):
    from riichienv.convert import tid_to_mpsz
    from jevmahjong.core.state import build_request
    if version('riichienv')!='0.4.11':raise ValueError('This native adapter requires the verified engine version')
    if opponents_no_declined_wins is not True:
        raise ValueError('A no-prior-declined-win certificate for sampled opponents is required')
    scope=decision_scope(source,own_obs,trigger)
    history=public_history(source.mjai_log)
    if history[-1]['type']!=trigger['type']:raise ValueError('Current source event differs from the public prefix')
    seat=own_obs.player_id
    if trigger['type']=='tsumo' and (trigger.get('actor')!=seat or source.drawn_tile is None
        or public_tile(trigger['pai'])!=tid_to_mpsz(source.drawn_tile)):
        raise ValueError('Current self-draw identity differs')
    before=build_request(own_obs)[0]
    inventory=decision_inventory(own_obs,scope)
    sizes=inventory['opponent_concealed_counts'];hands=allocation['opponent_hands']
    if set(hands)!=set(sizes) or any(len(hands[p])!=n for p,n in sizes.items()):
        raise ValueError('Proposed concealed sizes differ from the public meld counts')
    expected=Counter(allocation['unknown_nonhand_tiles'])
    expected.update(tile for hand in hands.values() for tile in hand)
    if expected!=Counter(inventory['unseen_tiles']):
        raise ValueError('Allocation does not partition the actual public unknown inventory')
    fork=source.clone()
    fork.hands=[list(own_obs.hand) if player==seat else list(allocation['opponent_hands'][player]) for player in range(4)]
    old=list(source.wall)
    fixed={4+2*i-source.rinshan_draw_count:tile for i,tile in enumerate(source.dora_indicators)}
    if any(index<0 or index>=len(old) or old[index]!=tile for index,tile in fixed.items()):
        raise ValueError('Known indicator slots differ from the pinned native layout')
    unknown=list(allocation['unknown_nonhand_tiles'])
    if len(unknown)!=len(old)-len(fixed):raise ValueError('Unknown tile/nonhand slot counts differ')
    pending=iter(unknown)
    fork.wall=[fixed[i] if i in fixed else next(pending) for i in range(len(old))]
    observed=Counter(tile for p,hand in enumerate(fork.hands) if p!=seat for tile in hand)
    observed.update(tile for i,tile in enumerate(fork.wall) if i not in fixed)
    if observed!=expected:raise ValueError('Sampled hidden physical inventory changed')
    missed=list(source.missed_agari_doujun)
    if set(allocation['opponent_temporary_furiten'])!=set(range(4))-{seat} or any(type(v) is not bool for v in allocation['opponent_temporary_furiten'].values()):
        raise ValueError('Explicit sampled-opponent temporary flags required')
    for player,value in allocation['opponent_temporary_furiten'].items():missed[player]=value
    fork.missed_agari_doujun=missed
    receipt=None
    if trigger['type']=='dahai':
        physical=source.last_discard[1]
        receipt=refresh_discard_claims(fork,trigger,physical)
    fresh=fork.get_obs_py([seat])[seat]
    if build_request(fresh)[0]!=before:
        raise ValueError('World construction changed the observer input or legal menu')
    if source.missed_agari_doujun[seat]!=fork.missed_agari_doujun[seat]:
        raise ValueError('Known observer temporary-furiten state changed')
    return fork,dict(own_input_and_menu_exact=True,sampled_reactions_refreshed=receipt,
        decision_scope=scope,own_known_declines_allowed_and_state_cloned=True,
        history_restriction='no_past_declined_opponent_riichi_ron',calibrated_history_posterior=False,
        full_match_continuation_not_yet_verified=True,training_labels=False)
