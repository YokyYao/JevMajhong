"""Construct current native worlds without stale sampled-hand reaction caches.

For the pinned RiichiEnv0.4.11. Current robbed-kan windows are not yet admitted.
Permanent missed-riichi flags must be false in the generated source history;
history sampling rejects hypotheses requiring one. No active fit uses this.
"""
from collections import Counter
from importlib.metadata import version

from tools.curriculum_public_history import public_history
from tools.curriculum_win_context import public_tile


def refresh_discard_claims(env,trigger,physical_tile):
    from riichienv import Phase,ActionType,HandEvaluator
    from riichienv.convert import tid_to_mpsz
    if trigger.get('type')!='dahai' or env.phase!=Phase.WaitResponse:
        raise ValueError('A current ordinary discard response is required')
    actor=trigger['actor']
    if env.last_discard!=(actor,physical_tile) or public_tile(trigger['pai'])!=tid_to_mpsz(physical_tile):
        raise ValueError('Public current discard identity differs')
    scratch=env.clone()
    rivers=[list(r) for r in scratch.discards]
    if not rivers[actor] or rivers[actor][-1]!=physical_tile:raise ValueError('Current source river differs')
    rivers[actor].pop();scratch.discards=rivers
    hands=[list(h) for h in scratch.hands];hands[actor].append(physical_tile);scratch.hands=hands
    scratch.turn_count=max(0,env.turn_count-1)
    scratch.apply_event(dict(type='dahai',actor=actor,pai=trigger['pai'],tsumogiri=trigger['tsumogiri']))
    claims=scratch.current_claims
    # Replay parsing canonicalizes equal non-red physical copies. Only its
    # type-level legal calculation is used; restore the actual public tile id
    # before copying the generated offers into the unchanged real world.
    for actions in claims.values():
        for action in actions:action.tile=physical_tile
    missed=list(env.missed_agari_doujun)
    for player in range(4):
        if player==actor or missed[player]:continue
        waiting=set(HandEvaluator(list(env.hands[player]),list(env.melds[player])).get_waits())
        if (physical_tile//4 in waiting and not waiting&{t//4 for t in env.discards[player]}
            and not any(a.action_type==ActionType.RON for a in claims.get(player,[]))):
            # Live discard processing also marks shape-complete/no-yaku
            # missed opportunities. Replay apply_event intentionally does not.
            missed[player]=True
    env.missed_agari_doujun=missed
    env.current_claims=claims
    env.active_players=sorted(claims)
    return dict(active_players=list(env.active_players),reaction_counts={p:len(v) for p,v in claims.items()})


def construct_world(source,own_obs,trigger,allocation,source_no_declined_wins):
    from riichienv import ActionType
    from riichienv.convert import tid_to_mpsz
    from jevmahjong.core.state import build_request,legal_candidates
    from tools.curriculum_public_inventory import from_observation
    if version('riichienv')!='0.4.11':raise ValueError('This native adapter requires the verified engine version')
    if source_no_declined_wins is not True:
        raise ValueError('A source-generator certificate of no prior declined legal wins is required')
    if trigger.get('type') not in ('dahai','tsumo'):
        raise ValueError('Current robbed-kan worlds need a separately audited constructor')
    history=public_history(source.mjai_log)
    if history[-1]['type']!=trigger['type']:raise ValueError('Current source event differs from the public prefix')
    seat=own_obs.player_id
    if trigger['type']=='tsumo' and (trigger.get('actor')!=seat or source.drawn_tile is None
        or public_tile(trigger['pai'])!=tid_to_mpsz(source.drawn_tile)):
        raise ValueError('Current self-draw identity differs')
    before=build_request(own_obs)[0]
    inventory=from_observation(own_obs,'ron' if trigger['type']=='dahai' else 'tsumo')
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
    expected=Counter(allocation['unknown_nonhand_tiles'])
    expected.update(tile for hand in allocation['opponent_hands'].values() for tile in hand)
    if expected!=Counter(inventory['unseen_tiles']):raise ValueError('Allocation does not partition the actual public unknown inventory')
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
    if not any(a.action_type in (ActionType.RON,ActionType.TSUMO) for a in legal_candidates(fresh)):
        raise ValueError('The native observed win is no longer legal')
    return fork,dict(own_input_and_menu_exact=True,sampled_reactions_refreshed=receipt,
        history_restriction='no_past_declined_opponent_riichi_ron',calibrated_history_posterior=False,
        full_match_continuation_not_yet_verified=True,training_labels=False)
