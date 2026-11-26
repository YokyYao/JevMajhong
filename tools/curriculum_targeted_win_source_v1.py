"""Targeted complete-hand sources after real native actions, not labels."""
from collections import Counter
import random

class SourceUnavailable(RuntimeError):pass


def wall_for(kind,target,seed,dealer=3):
    if kind not in ('win_reach','win_ankan','win_kakan') or len(target)!=14 or max(Counter(target).values())>4:
        raise ValueError('Registered complete14-tile target required')
    rng=random.Random(seed);pool=list(range(136));rng.shuffle(pool)
    win=target[-1];trash=[t for t in range(34) if t not in target]
    rng.shuffle(trash);first_trash,call_trash=trash[:2]
    if kind=='win_kakan':
        quad=next(t for t,n in Counter(target).items() if n==4)
        concealed=list(target);[concealed.remove(quad) for _ in range(3)]
        initial=[quad,quad]+concealed
        initial.remove(win);initial.append(call_trash)
    else:
        initial=list(target);initial.remove(win)
        quad=None
    def take(tile):
        physical=next(t for t in pool if t//4==tile);pool.remove(physical);return physical
    own=[take(t) for t in initial];first=take(first_trash);last=take(win)
    draw1=take(quad) if quad is not None else next(t for t in pool if t//4!=win)
    if quad is None:pool.remove(draw1)
    safe=[]
    # Pon resets the turn to the caller. After the caller discards, all
    # three other players draw before the caller's next self-draw.
    for _ in range(3 if quad is not None else 2):
        tile=next(t for t in pool if t//4!=win);pool.remove(tile);safe.append(tile)
    players=[(dealer+i)%4 for i in range(4)]
    hands={dealer:own}
    for player in players[1:]:hands[player]=pool[:13];del pool[:13]
    queue={p:list(h) for p,h in hands.items()};dealt=[]
    for _ in range(3):
        for player in players:dealt.extend(queue[player][:4]);del queue[player][:4]
    for player in players:dealt.append(queue[player][0])
    wall=dealt+[first,draw1,*safe,last]+pool
    if len(wall)!=136 or sorted(wall)!=list(range(136)):raise ValueError('Full physical wall must partition136 tiles')
    return wall,dict(own_initial=own,first_trash=first,call_trash_type=call_trash,
        win_tile=last,quad_type=quad)


def construct(kind,target,seed,scores,honba=0,sticks=0):
    from riichienv import RiichiEnv,GameRule,ActionType,Phase
    from jevmahjong.core.state import legal_candidates
    dealer=3;wall,plan=wall_for(kind,target,seed,dealer)
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=seed)
    observations=env.reset(oya=dealer,round_wind=0,scores=scores,honba=honba,kyotaku=sticks,wall=wall)
    if set(observations)!={dealer} or Counter(observations[dealer].hand)!=Counter(plan['own_initial']+[plan['first_trash']]):
        raise ValueError('Pinned native deal order does not match full-wall source')
    trace=[];pon_done=False;first_done=False
    for step in range(64):
        if env.done():raise SourceUnavailable('Opponent or early native settlement before target window')
        current=env.mjai_log[-1]
        if (first_done and dealer in observations and current['type']=='tsumo'
            and current.get('actor')==dealer and env.drawn_tile==plan['win_tile']):
            obs=observations[dealer];types={a.action_type for a in legal_candidates(obs)}
            required={'win_reach':ActionType.RIICHI,'win_ankan':ActionType.ANKAN,'win_kakan':ActionType.KAKAN}[kind]
            if ActionType.TSUMO not in types or required not in types:
                raise SourceUnavailable('Required native win/special operation is not jointly legal')
            return env,obs,dict(kind=kind,seed=seed,wall=wall,trace=trace,seat=dealer,oya=dealer,
                round_wind=0,honba=honba,initial_scores=scores,initial_sticks=sticks,
                opponents_no_prior_legal_win_declines=True,
                targeted_complete_hand_not_natural_game_distribution=True)
        selected={}
        for player,obs in sorted(observations.items()):
            # Physical source replay must retain the exact drawn copy. The
            # model menu deliberately collapses equivalent physical discards.
            menu=list(obs.legal_actions())
            wins=[a for a in menu if a.action_type in (ActionType.RON,ActionType.TSUMO)]
            if player!=dealer and wins:raise SourceUnavailable('A sampled source opponent could win before target')
            if player==dealer and not first_done:
                choices=[a for a in menu if a.action_type==ActionType.DISCARD and a.tile==plan['first_trash']]
                if len(choices)!=1:raise ValueError('Native first discard is not the registered trash tile')
                action=choices[0];first_done=True
            elif (player==dealer and kind=='win_kakan' and not pon_done
                and any(a.action_type==ActionType.PON for a in menu)):
                action=next(a for a in menu if a.action_type==ActionType.PON
                    and all(t//4==plan['quad_type'] for t in a.consume_tiles));pon_done=True
            elif player==dealer and kind=='win_kakan' and env.phase==Phase.WaitAct:
                choices=[a for a in menu if a.action_type==ActionType.DISCARD and a.tile//4==plan['call_trash_type']]
                if not choices:
                    raise ValueError('Post-pon trash absent: '+str(dict(step=step,pon_done=pon_done,
                        hand=[t//4 for t in obs.hand],trash=plan['call_trash_type'],
                        last_event=current['type'],drawn=env.drawn_tile,win=plan['win_tile'])))
                action=choices[0]
            elif any(a.action_type==ActionType.PASS for a in menu):
                action=next(a for a in menu if a.action_type==ActionType.PASS)
            elif any(a.action_type==ActionType.DISCARD and a.tile==env.drawn_tile for a in menu):
                action=next(a for a in menu if a.action_type==ActionType.DISCARD and a.tile==env.drawn_tile)
            else:raise SourceUnavailable('Unregistered native source transition')
            selected[player]=action
        trace.append([dict(player=p,type=int(a.action_type),tile=a.tile,consumed=list(a.consume_tiles))
            for p,a in selected.items()])
        observations=env.step(selected)
    raise SourceUnavailable('Target prefix exceeded64 native batches')


def restore(row):
    """Reconstruct full physical source; only its public observation is returned."""
    from riichienv import RiichiEnv,GameRule,ActionType
    from tools.curriculum_win_context import build_win_request
    source=row['_meta']['native_source'];seat=source['seat'];wall=source['wall']
    if len(wall)!=136 or sorted(wall)!=list(range(136)):
        raise ValueError('Native source lost its full physical wall')
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=source['seed'])
    observations=env.reset(oya=source['oya'],round_wind=source['round_wind'],scores=source['initial_scores'],
        honba=source['honba'],kyotaku=source['initial_sticks'],wall=wall)
    key=lambda a:(int(a.action_type),a.tile,tuple(sorted(a.consume_tiles)))
    for batch in source['trace']:
        if {a['player'] for a in batch}!=set(observations):raise ValueError('Source trace omitted a native actor')
        selected={}
        for saved in batch:
            player=saved['player'];menu=list(observations[player].legal_actions())
            options={key(a):a for a in menu};identity=(saved['type'],saved['tile'],tuple(sorted(saved['consumed'])))
            if identity not in options:raise ValueError('Source trace action is physically illegal')
            action=options[identity]
            if (player!=seat and any(a.action_type in (ActionType.RON,ActionType.TSUMO) for a in menu)
                and action.action_type not in (ActionType.RON,ActionType.TSUMO)):
                raise ValueError('A source opponent declined a legal win')
            selected[player]=action
        observations=env.step(selected)
        if env.done():raise ValueError('Source replay settled before the target')
    obs=env.get_obs_py([seat])[seat]
    if build_win_request(obs,env.mjai_log[-1])[0]!={k:v for k,v in row.items() if k!='_meta'}:
        raise ValueError('Full native reconstruction changed the public input/menu')
    return env,obs


