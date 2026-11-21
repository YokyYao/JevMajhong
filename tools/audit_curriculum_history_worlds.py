"""Check public-history priors and native paired continuations, without labels."""
import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def derived_seed(seed,hand):
    return int.from_bytes(hashlib.sha256(f'public-endgame-future-v1:{seed}:{hand}'.encode()).digest()[:8],'little')


def reactions(env):
    return {p:sorted((int(a.action_type),a.tile,tuple(sorted(a.consume_tiles))) for a in actions)
        for p,actions in env.current_claims.items()}


def controlled_cache_fixture():
    """A valid custom wall creates no-yaku furiten, then a newly sampled ron."""
    from riichienv import RiichiEnv,GameRule,ActionType,Phase,HandEvaluator
    from jevmahjong.core.state import legal_candidates
    from tools.curriculum_endgame_ledger import ledger as settlement_ledger
    from tools.curriculum_public_inventory import from_observation
    from tools.curriculum_public_history import public_history,history_flags
    from tools.curriculum_native_worlds import construct_world
    hands={0:[40,41,42,44,45,46,48,49,50,52,53,54,56],
        1:[0,4,8,12,17,20,60,64,68,76,80,108,109]}
    reserved={35,72,57}
    pool=sorted(set(range(136))-set(hands[0])-set(hands[1])-reserved)
    rng=random.Random(2026400600);rng.shuffle(pool)
    hands[2]=pool[:13];hands[3]=[57,*pool[13:25]]
    queues={p:list(h) for p,h in hands.items()};deal=[]
    for _ in range(3):
        for offset in range(4):
            player=(2+offset)%4
            deal.extend(queues[player][:4]);del queues[player][:4]
    for offset in range(4):deal.append(queues[(2+offset)%4][0])
    remaining=sorted(set(range(136))-set(deal)-{35,72})
    draw=remaining.pop(0)
    remaining.insert(len(remaining)-4,35)
    wall=[*deal,72,draw,*remaining]
    if len(wall)!=136 or set(wall)!=set(range(136)):raise ValueError('Controlled physical wall is invalid')
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=2026400600*2**32)
    observations=env.reset(oya=2,wall=wall)
    if env.dora_indicators!=[35]:raise ValueError('Controlled public indicator differs')
    if any(a.action_type==ActionType.TSUMO for a in legal_candidates(observations[2])):raise ValueError('Controlled dealer unexpectedly can win')
    discard=next(a for a in legal_candidates(observations[2]) if a.action_type==ActionType.DISCARD and a.tile==72)
    observations=env.step({2:discard})
    if not env.missed_agari_doujun[1]:raise ValueError('Native no-yaku shape did not set temporary furiten')
    while env.phase==Phase.WaitResponse:
        selected={}
        for player,obs in observations.items():
            actions=legal_candidates(obs)
            if any(a.action_type==ActionType.RON for a in actions):raise ValueError('Controlled setup would decline a legal win')
            selected[player]=next(a for a in actions if a.action_type==ActionType.PASS)
        observations=env.step(selected)
    if env.current_player!=3 or any(a.action_type==ActionType.TSUMO for a in legal_candidates(observations[3])):
        raise ValueError('Controlled next player differs or can win')
    discard=next(a for a in legal_candidates(observations[3]) if a.action_type==ActionType.DISCARD and a.tile==57)
    observations=env.step({3:discard})
    obs=observations[0];visible=obs.to_dict();history=public_history(env.mjai_log)
    waiting=lambda p,hand:list(HandEvaluator(hand,list(visible['melds'][p])).get_waits())
    truth=dict(opponent_hands={p:list(h) for p,h in enumerate(env.hands) if p!=0})
    flags=history_flags(history,0,visible['riichi_declared'],truth,waiting)
    if not flags['compatible'] or flags['opponent_temporary_furiten'][1] is not True:
        raise ValueError('Public history missed the native no-yaku temporary flag')
    target=[108,109,110,0,1,2,32,33,34,96,97,98,58]
    inventory=from_observation(obs,'ron')
    if not set(target)<=set(inventory['unseen_tiles']):raise ValueError('New ron hand uses a publicly unavailable tile')
    remaining=sorted(set(inventory['unseen_tiles'])-set(target));rng.shuffle(remaining)
    proposal=dict(opponent_hands={1:target,2:remaining[:13],3:remaining[13:26]},unknown_nonhand_tiles=remaining[26:])
    derived=history_flags(history,0,visible['riichi_declared'],proposal,waiting)
    if not derived['compatible'] or derived['opponent_temporary_furiten'][1] is not False:
        raise ValueError('New proposed hand incorrectly retains the old no-yaku flag')
    proposal['opponent_temporary_furiten']=derived['opponent_temporary_furiten']
    if any(a.action_type==ActionType.RON for a in env.current_claims.get(1,[])):
        raise ValueError('Controlled original unexpectedly already offers ron to player1')
    fork,receipt=construct_world(env,obs,env.mjai_log[-1],proposal,True)
    if not any(a.action_type==ActionType.RON for a in fork.current_claims.get(1,[])):
        raise ValueError('Fresh claims failed to expose the newly sampled ron')
    active=fork.get_obs_py(list(fork.active_players));selected={}
    for player,view in active.items():
        selected[player]=next(a for a in legal_candidates(view) if a.action_type==(ActionType.RON if player==1 else ActionType.PASS))
    prefix=len(fork.mjai_log);fork.step(selected)
    if not any(e['type']=='hora' and e['actor']==1 for e in fork.mjai_log[prefix:]):
        raise ValueError('The newly sampled ron was not executed')
    settlement_ledger(fork.mjai_log,fork.scores())
    return dict(no_yaku_temporary_flag_true_in_native_source=True,history_reconstructed_true_flag=True,
        proposed_hand_clears_unrelated_old_flag=True,previously_absent_ron_rebuilt_and_executed=True,
        own_input_and_menu_exact=receipt['own_input_and_menu_exact'],point_ledger_passed=True,
        scripted_fixture_not_training_label=True)


def finish_fixture(env,seat,first_kind,future_seed):
    """All later decisions use public-observation Efficiency agents.

    Later hands get independent, paired seeds. Recreate only at an already
    settled hand boundary, preserving every public match field and merging
    its financial event stream. This is an integration fixture, not a teacher.
    """
    from riichienv import RiichiEnv,GameRule,ActionType
    from jevmahjong.core.state import legal_candidates,build_request
    from tools.curriculum_endgame_ledger import ledger as settlement_ledger
    from jevmahjong.evaluation.agents import EfficiencyAgent
    agent=EfficiencyAgent();combined=list(env.mjai_log);cursor=len(combined)
    observations=env.get_obs_py(list(env.active_players));initial=True;hands=0;resets=0
    for step in range(30000):
        if env.done():break
        marker=(env.round_wind,env.kyoku_idx,env.honba)
        selected={p:agent.act(obs) for p,obs in observations.items()}
        if initial:
            actions=legal_candidates(observations[seat])
            if first_kind=='accept':
                selected[seat]=next(a for a in actions if a.action_type in (ActionType.RON,ActionType.TSUMO))
            else:
                options=[a for a in actions if a.action_type==ActionType.PASS]
                if options:selected[seat]=options[0]
                else:
                    request,menu,features=build_request(observations[seat])
                    candidates=[(a,f) for a,f in zip(menu,features) if a.action_type==ActionType.DISCARD]
                    selected[seat]=min(candidates,key=lambda pair:(pair[1]['sh'],-pair[1]['ukeire'],pair[1]['red']))[0]
            initial=False
        observations=env.step(selected)
        log=list(env.mjai_log);changed=(env.round_wind,env.kyoku_idx,env.honba)!=marker
        if changed and not env.done():
            new_start=next(i for i in range(cursor,len(log)) if log[i]['type']=='start_kyoku')
            combined.extend(log[cursor:new_start]);hands+=1
            if hands>64:raise RuntimeError('Fixture exceeded its bounded hand count')
            context=dict(oya=env.oya,round_wind=env.round_wind,scores=list(env.scores()),honba=env.honba,kyotaku=env.riichi_sticks)
            fresh=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=derived_seed(future_seed,hands))
            observations=fresh.reset(**context)
            if (fresh.oya!=context['oya'] or fresh.round_wind!=context['round_wind'] or list(fresh.scores())!=context['scores']
                or fresh.honba!=context['honba'] or fresh.riichi_sticks!=context['kyotaku']):
                raise ValueError('Reseeding the next hand changed public match context')
            env=fresh;initial_log=list(env.mjai_log)
            combined.extend(e for e in initial_log if e['type']!='start_game');cursor=len(initial_log);resets+=1
        else:
            combined.extend(log[cursor:]);cursor=len(log)
        if not observations and not env.done():raise RuntimeError('Native branch returned no legal next decision')
    if not env.done():raise RuntimeError('A full-match fixture did not finish')
    ledger=settlement_ledger(combined,env.scores())
    return dict(terminal_scores=list(env.scores()),rank=int(env.ranks()[seat]),top2=env.ranks()[seat]<=2,
        full_match_completed=True,future_hand_reseeds=resets,merged_ledger_hands=ledger['hands'],
        next_hand_context_preserved=True,continuation='four_public_efficiency_agents_fixture_not_teacher')


def audit(registered,out,seed_start=2026400400,games=12,cases=8,worlds=2,late_only=False,full=False):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    if out.exists() or not 1<=games<=64 or not 1<=cases<=64 or not 1<=worlds<=16:raise ValueError('Fresh output and bounded fixture budget required')
    if set(range(seed_start,seed_start+games))&set(range(2026300000,2026300032)):raise ValueError('Reserved final seeds cannot be inspected')
    reg=json.loads(registered.read_text());sys.path.insert(0,reg['source'])
    from riichienv import RiichiEnv,GameRule,ActionType,HandEvaluator,calculate_shanten
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import legal_candidates,build_request
    from jevmahjong.evaluation.agents import EfficiencyAgent,CallingOpponent
    from tools.curriculum_endgame import in_endgame_scope,public_placement
    from tools.curriculum_public_inventory import from_observation
    from tools.curriculum_public_history import public_history,history_flags,sample_history_allocations
    from tools.curriculum_public_worlds import SamplingExhausted
    from tools.curriculum_native_worlds import construct_world
    controlled=controlled_cache_fixture()
    counts=Counter();records=[];declined=0
    for seed in range(seed_start,seed_start+games):
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
        observations=env.reset();agents=[CallingOpponent() if p==1 else EfficiencyAgent() for p in range(4)]
        for step in range(30000):
            if env.done():break
            trigger=env.mjai_log[-1]
            for seat,obs in observations.items():
                wins=[a for a in legal_candidates(obs) if a.action_type in (ActionType.RON,ActionType.TSUMO)]
                if not wins:continue
                if trigger['type'] not in ('dahai','tsumo'):
                    counts['current_kan_trigger_not_admitted']+=1;continue
                kind='ron' if wins[0].action_type==ActionType.RON else 'tsumo'
                visible=obs.to_dict();melds=visible['melds'];riichi=visible['riichi_declared']
                history=public_history(env.mjai_log)
                waiting=lambda player,hand:list(HandEvaluator(hand,list(melds[player])).get_waits())
                # Positive control reads the actual private hands only here;
                # the independent sampler below receives public inventory and history.
                truth=dict(opponent_hands={p:list(h) for p,h in enumerate(env.hands) if p!=seat})
                flags=history_flags(history,seat,riichi,truth,waiting)
                if not flags['compatible']:raise ValueError('The source truth fails its no-declined-win public-history control')
                fixed={4+2*i-env.rinshan_draw_count for i in range(len(env.dora_indicators))}
                truth.update(unknown_nonhand_tiles=[tile for i,tile in enumerate(env.wall) if i not in fixed],
                    opponent_temporary_furiten=flags['opponent_temporary_furiten'])
                copy,receipt=construct_world(env,obs,trigger,truth,declined==0)
                if reactions(copy)!=reactions(env) or list(copy.active_players)!=list(env.active_players):
                    raise ValueError('Rebuilt truth reaction cache differs from the native original')
                if copy.missed_agari_doujun!=env.missed_agari_doujun:
                    raise ValueError('Reconstructed truth temporary furiten differs')
                counts['truth_cache_and_furiten_controls']+=1
                request=build_request(obs)[0]
                count_riichi=sum(bool(riichi[p]) for p in range(4) if p!=seat)
                if in_endgame_scope(request['state']):counts['eligible_late_windows_riichi_'+str(count_riichi)]+=1
                if counts['sampled_cases']>=cases or (late_only and not in_endgame_scope(request['state'])):continue
                pool=from_observation(obs,kind)
                try:drawn=sample_history_allocations(pool,history,riichi,derived_seed(seed,step*4+seat),worlds,200000,calculate_shanten,waiting)
                except SamplingExhausted:
                    counts['sampling_exhausted_cases_not_labelled']+=1;continue
                counts['sampled_cases']+=1;counts['source_'+kind]+=1
                counts['start_rank_'+str(public_placement(request['state'])['rank'])]+=1
                counts['riichi_opponent_cases']+=bool(pool['riichi_tenpai_required'])
                counts['sampled_cases_riichi_'+str(count_riichi)]+=1
                counts['allocation_proposals']+=drawn['attempts']
                counts['history_rejected_tenpai_allocations']+=drawn['history_rejected_tenpai_allocations']
                for index,allocation in enumerate(drawn['allocations']):
                    fork,receipt=construct_world(env,obs,trigger,allocation,declined==0)
                    counts['constructed_worlds']+=1
                    counts['changed_reaction_caches']+=reactions(fork)!=reactions(env)
                    counts['derived_true_temporary_flags']+=sum(allocation['opponent_temporary_furiten'].values())
                    if full:
                        future=derived_seed(seed,(step*4+seat)*32+index)
                        arms={arm:finish_fixture(fork.clone(),seat,arm,future) for arm in ('accept','decline')}
                        counts['completed_paired_full_matches']+=1
                        counts['future_hand_reseeds']+=sum(a['future_hand_reseeds'] for a in arms.values())
                        records.append(dict(seed=seed,step=step,seat=seat,world=index,arms=arms,
                            source_placement=public_placement(request['state']),source_scores=request['state']['scores'],
                            source_round_wind=request['state']['round_wind'],source_hand_number=request['state']['hand_number'],
                            source_honba=request['state']['honba'],source_sticks=request['state']['sticks'],
                            riichi_opponents=count_riichi,not_training_labels=True))
            selected={p:agents[p].act(obs) for p,obs in observations.items()}
            for player,choice in selected.items():
                offered=any(a.action_type in (ActionType.RON,ActionType.TSUMO) for a in legal_candidates(observations[player]))
                declined+=offered and choice.action_type not in (ActionType.RON,ActionType.TSUMO)
            if declined:raise ValueError('Source generator unexpectedly declined a legal win')
            observations=env.step(selected)
        if not env.done():raise RuntimeError('Source game did not complete')
        counts['complete_source_games']+=1
        if counts['sampled_cases']>=cases:break
    result=dict(status='native_history_world_integration_checked',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        seed_start=seed_start,requested_games=games,requested_cases=cases,requested_worlds_per_case=worlds,late_only=late_only,
        counts=dict(counts),source_legal_win_declines=declined,full_paired_fixtures_requested=full,
        controlled_cache_and_no_yaku_furiten_fixture=controlled,
        native_current_robbed_kan_worlds_supported=False,calibrated_public_history_posterior=False,
        whole_unknown_history_reconstruction=False,labels_generated=0,stage4_fit_admitted=False,
        full_continuations_use_efficiency_fixture_not_teacher=True,active_training_files_unchanged=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_name(out.stem+'-paired-fixtures-private.jsonl').open('x') as f:
        for row in records:f.write(json.dumps(row)+'\n')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed-start',type=int,default=2026400400)
    p.add_argument('--games',type=int,default=12);p.add_argument('--cases',type=int,default=8);p.add_argument('--worlds',type=int,default=2)
    p.add_argument('--late-only',action='store_true');p.add_argument('--full-continuations',action='store_true')
    a=p.parse_args();audit(a.registered,a.out,a.seed_start,a.games,a.cases,a.worlds,a.late_only,a.full_continuations)
