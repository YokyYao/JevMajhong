"""Financial ledger for complete native episodes with declared score contexts."""
from collections import Counter


def ledger(log,final_scores):
    starts=[event for event in log if event['type']=='start_kyoku']
    if not starts:raise ValueError('An explicit initial hand score context is required')
    initial=list(starts[0]['scores']);initial_pool=starts[0]['kyotaku']
    if (len(initial)!=4 or any(type(v) is not int or v%100 for v in initial)
        or type(initial_pool) is not int or initial_pool<0 or sum(initial)+1000*initial_pool!=100000):
        raise ValueError('Invalid declared initial point/stick inventory')
    scores=list(initial);pool=initial_pool;components=[Counter() for _ in range(4)]
    hands=wins=deposits=0
    for index,event in enumerate(log):
        kind=event['type']
        if kind=='start_kyoku':
            if event['scores']!=scores or event['kyotaku']!=pool:raise ValueError('Hand-boundary score/stick mismatch')
            hands+=1
        elif kind=='reach_accepted':
            actor=event['actor'];scores[actor]-=1000;pool+=1;deposits+=1
            components[actor]['riichi_deposits']-=1000
        elif kind in ('hora','ryukyoku'):
            delta=event['deltas']
            if len(delta)!=4 or any(type(v) is not int for v in delta):raise ValueError('Invalid settlement vector')
            if kind=='hora':
                if sum(delta)!=1000*pool:raise ValueError('Win/deposit-pool mismatch')
                pool=0;wins+=1
            elif sum(delta)!=0:raise ValueError('Draw does not conserve points')
            for player,value in enumerate(delta):
                scores[player]+=value
                category=('draw_settlement' if kind=='ryukyoku' else 'win_income' if player==event['actor']
                    else 'tsumo_payments' if event['actor']==event['target'] else 'ron_payments')
                components[player][category]+=value
        elif kind=='end_game' and pool:
            leader=max(range(4),key=lambda p:(scores[p],-p))
            scores[leader]+=1000*pool;components[leader]['final_unclaimed_pool']+=1000*pool;pool=0
        if sum(scores)+1000*pool!=100000:raise ValueError(f'Point conservation failed at event{index}')
    if scores!=list(final_scores):raise ValueError('Terminal scores differ from the financial event ledger')
    return dict(initial_scores=initial,initial_riichi_sticks=initial_pool,hands=hands,wins=wins,
        riichi_deposits=deposits,unclaimed_riichi_sticks=pool,final_scores=scores,
        components=[dict(v) for v in components],independent_han_fu_verification=False)
