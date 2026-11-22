"""Actual legal ron passes, constructor clones, and known observer furiten."""
import json
import random


def audit():
    from riichienv import RiichiEnv,GameRule,ActionType,Phase,HandEvaluator
    from jevmahjong.core.state import legal_candidates,build_request
    from tools.curriculum_decision_history import public_history,history_flags
    from tools.curriculum_native_decisions import construct_world
    results=[]
    def legal(obs,kind,tile=None):
        # Native candidate deduplication keeps one representative physical
        # copy for an otherwise identical non-red discard signature.
        return next(a for a in legal_candidates(obs) if a.action_type==kind and (tile is None or a.tile//4==tile//4))
    def passive(env,observations):
        while env.phase==Phase.WaitResponse:
            assert not any(a.action_type==ActionType.RON for p,o in observations.items() if p!=0 for a in legal_candidates(o))
            observations=env.step({p:legal(o,ActionType.PASS) for p,o in observations.items()})
        return observations
    def has_ron(observations):
        return 0 in observations and any(a.action_type==ActionType.RON for a in legal_candidates(observations[0]))
    for riichi in (False,True):
        initial=[0,4,8,48,52,56,96,100,104,124,125,126,16]
        draws=[108,17,18,116,120,19]
        remaining=sorted(set(range(136))-set(initial)-set(draws));random.Random(2026402200).shuffle(remaining)
        hands={0:initial,1:remaining[:13],2:remaining[13:26],3:remaining[26:39]}
        queue={p:list(h) for p,h in hands.items()};deal=[]
        for _ in range(3):
            for p in range(4):deal.extend(queue[p][:4]);del queue[p][:4]
        for p in range(4):deal.append(queue[p][0])
        wall=deal+draws+remaining[39:];assert len(wall)==136 and len(set(wall))==136
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=2026402200)
        observations=env.reset(oya=0,wall=wall)
        if riichi:observations=env.step({0:legal(observations[0],ActionType.RIICHI)})
        observations=passive(env,env.step({0:legal(observations[0],ActionType.DISCARD,108)}))
        observations=env.step({1:legal(observations[1],ActionType.DISCARD,17)})
        assert has_ron(observations),'The first actual discard must offer legal ron'
        observations=passive(env,observations)
        observations=env.step({2:legal(observations[2],ActionType.DISCARD,18)})
        assert not has_ron(observations),'A passed ron must block the next same-cycle ron'
        observations=passive(env,observations)
        observations=passive(env,env.step({3:legal(observations[3],ActionType.DISCARD,116)}))
        assert set(observations)=={0} and env.drawn_tile==120
        obs=observations[0];visible=obs.to_dict();known={4+2*i-env.rinshan_draw_count for i in range(len(env.dora_indicators))}
        allocation=dict(opponent_hands={p:list(h) for p,h in enumerate(env.hands) if p!=0},
            unknown_nonhand_tiles=[tile for i,tile in enumerate(env.wall) if i not in known])
        waits=lambda p,h:list(HandEvaluator(h,list(visible['melds'][p])).get_waits())
        flags=history_flags(public_history(env.mjai_log),0,visible['riichi_declared'],allocation,waits)
        assert flags['compatible'];allocation['opponent_temporary_furiten']=flags['opponent_temporary_furiten']
        rebuilt,receipt=construct_world(env,obs,env.mjai_log[-1],allocation,True)
        assert build_request(rebuilt.get_obs_py([0])[0])[0]==build_request(obs)[0]
        for name,branch in (('source',env.clone()),('reconstructed',rebuilt)):
            current=branch.get_obs_py([0]);current=passive(branch,branch.step({0:legal(current[0],ActionType.DISCARD,120)}))
            current=branch.step({1:legal(current[1],ActionType.DISCARD,19)})
            assert has_ron(current)==(not riichi),'Known passed-riichi state or non-riichi reset was lost'
        results.append(dict(riichi=riichi,first_ron_was_legal=True,actual_ron_passed=True,same_cycle_ron_blocked=True,
            truth_reconstruction_menu_exact=True,after_next_own_draw_discard_ron_allowed=not riichi,
            source_and_reconstructed_future_behavior_equal=True))
    return dict(status='actual_known_observer_decline_controls_passed',cases=results,
        physical_walls_verified=True,models_loaded=False,new_training_labels=0,
        scope='Two actual native ron-decline fixtures and true-world reconstruction; not all nested sampled-world histories or a calibrated prior')
