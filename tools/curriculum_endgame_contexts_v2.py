"""Score replay v2 accepts live post-call choices as well as win triggers.

The furiten sampler's stricter win-trigger history contract is unchanged.
This accounting projection needs only current-hand accepted deposits.
"""
import json


def accounting_history(events):
    parsed=[json.loads(e) if isinstance(e,str) else e for e in events]
    starts=[i for i,e in enumerate(parsed) if e.get('type')=='start_kyoku']
    if not starts:raise ValueError('A current-hand start is required')
    projected=[]
    live={'tsumo','dahai','chi','pon','ankan','kakan','daiminkan','reach','reach_accepted'}
    for event in parsed[starts[-1]:]:
        kind=event.get('type')
        if kind in ('start_kyoku','dora'):projected.append(dict(type=kind));continue
        if kind not in live:raise ValueError('A score replay requires a live unsettled hand prefix')
        actor=event.get('actor')
        if type(actor) is not int or actor not in range(4):raise ValueError('Invalid public accounting actor')
        projected.append(dict(type=kind,actor=actor))
    if len(projected)<2:raise ValueError('A current decision event is required')
    return projected


def score_contexts(seat,sticks):
    if type(seat) is not int or seat not in range(4) or type(sticks) is not int or sticks<0:
        raise ValueError('Explicit observing seat and current stick bank required')
    bank=1000*sticks;others=[p for p in range(4) if p!=seat]
    specs=[('hold_first',1,45000-bank,[25000,18000,12000]),
        ('hold_second',2,25000,[43000-bank,22000,10000]),
        ('third_gap3000',3,22000,[43000-bank,25000,10000]),
        ('fourth_gap5000',4,20000,[33000-bank,25000,22000]),
        ('fourth_gap13000',4,12000,[45000-bank,25000,18000])]
    result={}
    for name,rank,own,rest in specs:
        scores=[0]*4;scores[seat]=own
        for player,value in zip(others,rest):scores[player]=value
        order=sorted(range(4),key=lambda p:(-scores[p],p))
        if min(scores)<1000 or order.index(seat)+1!=rank:continue
        gap={'third_gap3000':3000,'fourth_gap5000':5000,'fourth_gap13000':13000}.get(name)
        if gap is not None and scores[order[1]]-scores[seat]!=gap:continue
        assert sum(scores)+bank==100000
        result[name]=dict(scores=scores,rank=rank)
    return result


def starting_scores(history,current_scores,current_sticks):
    if (type(current_sticks) is not int or current_sticks<0 or len(current_scores)!=4
        or any(type(v) is not int for v in current_scores)):
        raise ValueError('Four declared current scores and an integer stick bank required')
    projected=accounting_history(history)
    paid=[0]*4
    for event in projected:
        if event['type']=='reach_accepted':paid[event['actor']]+=1
    if any(v>1 for v in paid):raise ValueError('Duplicate riichi deposits in one hand')
    initial_sticks=current_sticks-sum(paid)
    initial=[value+1000*paid[player] for player,value in enumerate(current_scores)]
    if (len(initial)!=4 or any(type(v) is not int or v<1000 or v%100 for v in initial)
        or initial_sticks<0 or sum(initial)+1000*initial_sticks!=100000):
        raise ValueError('Requested score context cannot conserve the declared riichi bank')
    return initial,initial_sticks


def replay_score_context(source,source_engine_seed,hand_index,trace,seat,desired_scores):
    """Use the same physical wall/actions, with a new valid public score context.

    source_engine_seed+hand_index reproduces this version's per-hand shuffle.
    All score variants of a wall must share one training/validation group.
    """
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.state import legal_candidates,build_request
    if type(hand_index) is not int or hand_index<0:raise ValueError('Explicit original hand index required')
    initial,bank=starting_scores(source.mjai_log,desired_scores,source.riichi_sticks)
    fork=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=source_engine_seed+hand_index)
    observations=fork.reset(oya=source.oya,round_wind=source.round_wind,honba=source.honba,scores=initial,kyotaku=bank)
    key=lambda action:(int(action.action_type),action.tile,tuple(sorted(action.consume_tiles)))
    for batch in trace:
        if set(batch)!=set(observations):raise ValueError('Augmented replay reaction-window coverage differs')
        selected={}
        for player,saved in batch.items():
            menu={key(a):a for a in legal_candidates(observations[player])}
            if key(saved) not in menu:raise ValueError('A source action is illegal in the augmented context')
            selected[player]=menu[key(saved)]
        observations=fork.step(selected)
        if fork.done():raise ValueError('Augmentation settled before the source intervention')
    if list(fork.scores())!=list(desired_scores) or fork.riichi_sticks!=source.riichi_sticks:
        raise ValueError('Augmented current score/stick table differs')
    original=build_request(source.get_obs_py([seat])[seat])[0]
    augmented=build_request(fork.get_obs_py([seat])[seat])[0]
    left=dict(original['state']);right=dict(augmented['state']);left.pop('scores');right.pop('scores')
    if left!=right or original['questions']!=augmented['questions']:
        raise ValueError('Score augmentation changed hand information or the legal menu')
    return fork,dict(only_current_scores_changed=True,initial_scores=initial,initial_sticks=bank,
        replayed_batches=len(trace),same_physical_wall=True,wall_group_must_be_shared_across_contexts=True,
        training_labels=False)
