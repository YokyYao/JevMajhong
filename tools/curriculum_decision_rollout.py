"""Complete a native intervention match using only per-seat observations."""
from tools.audit_curriculum_history_worlds import derived_seed


def finish(env,seat,first_key,future_seed,agents=None,record_events=False):
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.state import build_request
    from jevmahjong.evaluation.agents import EfficiencyAgent
    from tools.curriculum_endgame_ledger import ledger
    if agents is None:agents=[EfficiencyAgent() for _ in range(4)];kind='four_public_efficiency_agents_fixture_not_teacher'
    else:
        if len(agents)!=4:raise ValueError('Four public-observation continuation agents required')
        kind='explicit_supplied_public_observation_agents_not_automatically_admitted_teacher'
    anchor=list(env.scores());combined=list(env.mjai_log);cursor=len(combined)
    observations=env.get_obs_py(list(env.active_players));initial=True;hands=0;resets=0
    for step in range(30000):
        if env.done():break
        marker=(env.round_wind,env.kyoku_idx,env.honba)
        selected={p:agents[p].act(obs) for p,obs in sorted(observations.items())}
        if initial:
            request,actions,_=build_request(observations[seat]);keys=list(request['questions']['action']['criteria'])
            if first_key not in keys:raise ValueError('Intervention key absent from the exact current native menu')
            selected[seat]=actions[keys.index(first_key)];initial=False
        observations=env.step(selected)
        log=list(env.mjai_log);changed=(env.round_wind,env.kyoku_idx,env.honba)!=marker
        if changed and not env.done():
            new_start=next(i for i in range(cursor,len(log)) if log[i]['type']=='start_kyoku')
            combined.extend(log[cursor:new_start]);hands+=1
            if hands>64:raise RuntimeError('Bounded continuation exceeded64 subsequent hands')
            context=dict(oya=env.oya,round_wind=env.round_wind,scores=list(env.scores()),honba=env.honba,kyotaku=env.riichi_sticks)
            fresh=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=derived_seed(future_seed,hands))
            observations=fresh.reset(**context)
            if (fresh.oya!=context['oya'] or fresh.round_wind!=context['round_wind'] or list(fresh.scores())!=context['scores']
                or fresh.honba!=context['honba'] or fresh.riichi_sticks!=context['kyotaku']):
                raise ValueError('Independent later-hand seed changed public match context')
            env=fresh;initial_log=list(env.mjai_log)
            combined.extend(e for e in initial_log if e['type']!='start_game');cursor=len(initial_log);resets+=1
        else:combined.extend(log[cursor:]);cursor=len(log)
        if not observations and not env.done():raise RuntimeError('Live match returned no next native decision')
    if not env.done():raise RuntimeError('A full-terminal intervention failed to finish within its bound')
    checked=ledger(combined,env.scores())
    result=dict(terminal_scores=list(env.scores()),rank=int(env.ranks()[seat]),top2=env.ranks()[seat]<=2,
        net_from_anchor=env.scores()[seat]-anchor[seat],full_match_completed=True,future_hand_reseeds=resets,
        merged_ledger_hands=checked['hands'],next_hand_context_preserved=True,continuation=kind,
        independent_han_fu_verification=False,teacher_admitted=False,training_label=False)
    if record_events:result['merged_mjai_log']=combined
    return result
