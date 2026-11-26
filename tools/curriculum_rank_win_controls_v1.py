"""Public-only cohorts and complete current win menus for placement controls."""
import hashlib
from tools.curriculum_rank_nonwin_teacher_v2 import soft_target as action_soft_target
METHOD='public32_rank_priority_complete_current_win_menu_controls_v1'
BUCKETS=('sufficient_behind:ron','sufficient_behind:tsumo',
    'leading_accept_control:ron','leading_accept_control:tsumo',
    'valuable_still_behind:ron','valuable_still_behind:tsumo')


def bucket(row,score):
    if not score['supported']:return None
    rank=row['_meta']['placement']['rank']
    kind=next(iter(row['state']['win_context']['candidates'].values()))['kind']
    phase=('sufficient_behind' if rank>=3 and score['immediate_rank']<=2 else
        'valuable_still_behind' if rank>=3 and score['own_income']>=8000 else
        'leading_accept_control' if rank<=2 and score['immediate_rank']<=2 else None)
    return phase+':'+kind if phase else None


def select(rows,scores,excluded,quotas):
    selected=[]
    for split in ('train','validation'):
        for name in BUCKETS:
            candidates=[r for r in rows if r['_meta']['split']==split and r['_meta']['id'] not in excluded
                and r['_meta']['category']=='win' and bucket(r,scores[r['_meta']['id']])==name]
            def order(row):
                special=bool({v['type'] for v in row['questions']['action']['criteria'].values()} & {'reach','ankan','kakan','daiminkan'})
                digest=hashlib.sha256(('rank-win-controls-v1:'+name+':'+row['_meta']['id']).encode()).hexdigest()
                return not special,digest
            candidates.sort(key=order);used=set();chosen=[]
            for row in candidates:
                group=row['_meta']['group_id']
                if group in used:continue
                used.add(group);chosen.append(row)
                if len(chosen)==quotas[split][name]:break
            if len(chosen)!=quotas[split][name]:raise ValueError('Insufficient distinct families for '+split+':'+name)
            selected.extend((name,row) for row in chosen)
    return selected


def propose(env,obs,client,seed):
    from jevmahjong.core.state import build_request
    from tools.curriculum_rank_win_teacher_v1 import propose as original
    request=build_request(obs)[0];criteria=request['questions']['action']['criteria']
    wins=[k for k,v in criteria.items() if v['type']=='hora']
    if len(wins)!=1 or not 2<=len(criteria)<=16:raise ValueError('One legal win and complete native menu of2..16 options required')
    plan=original(env,obs,client,seed=seed)
    if plan['accept']!=wins[0]:raise ValueError('Public win identity changed')
    candidates=wins+[k for k in criteria if k not in wins]
    return dict(plan,baseline=wins[0],candidates=candidates,method=METHOD,
        legal_menu_size=len(criteria),tested_menu_size=len(candidates),complete_current_legal_menu=True,
        action_types={k:criteria[k]['type'] for k in candidates})


def soft_target(worlds,candidates,initial_rank,seed):
    result=action_soft_target(worlds,candidates,initial_rank,seed)
    return dict(result,method=METHOD,exact_placement_ties_take_current_legal_win=True,
        current_complete_legal_menu_compared=True,future_win_replanner_remains_two_arm_approximation=True)

