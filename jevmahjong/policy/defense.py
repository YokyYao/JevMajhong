"""Public pure-discard defense windows and an optional expanded action set."""
from collections import Counter
import json
from riichienv import ActionType,MeldType
from jevmahjong.core.state import build_request,legal_candidates
from jevmahjong.evaluation.agents import PolicyAgent,guarded_indices

VERSION='public-discard-defense-context-v1'


def is_defense_request(request):
    try:
        state=request['state'];seat=state['seat'];criteria=list(request['questions']['action']['criteria'].values())
        threats=set(state['riichi'])-{seat}
        return (bool(threats) and seat not in state['riichi'] and len(criteria)>=2
            and any(m['type']!=int(MeldType.Ankan) for m in state['melds'][seat])
            and all(c['type']=='dahai' for c in criteria)
            and min(c['sh'] for c in criteria)==0
            and any(c['sh']>0 and threats.issubset(set(c['safe'])) for c in criteria))
    except (KeyError,TypeError,ValueError):
        return False


def request_eligible(request):
    if not is_defense_request(request):raise ValueError('Public open-tenpai defense window required')
    threats=set(request['state']['riichi'])-{request['state']['seat']}
    return [i for i,c in enumerate(request['questions']['action']['criteria'].values())
        if c['sh']==0 or threats.issubset(set(c['safe']))]


def eligible_indices(obs,actions,feats):
    original=sorted(set(guarded_indices(obs,actions,feats))|{i for i,f in enumerate(feats) if not f})
    state=obs.to_dict();seat=obs.player_id
    threats={p for p in range(4) if p!=seat and state['riichi_declared'][p]}
    discards=[f for f in feats if f]
    if (not threats or state['riichi_declared'][seat]
            or not any(m.opened for m in state['melds'][seat])
            or not discards or min(f['sh'] for f in discards)!=0):return original
    return sorted(set(original)|{i for i,f in enumerate(feats) if f and threats.issubset(set(f['safe']))})


class DefensePolicyAgent(PolicyAgent):
    def __init__(self,client):
        super().__init__(client);self.admissions=Counter()

    def act(self,obs):
        from jevany.api import validate_distribution
        actions=legal_candidates(obs)
        if len(actions)==1:return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON,ActionType.TSUMO):return action
        request,actions,feats=build_request(obs)
        answer=self.client.system_one(**request)
        p,_=validate_distribution(answer['answers']['action']['probabilities'],list(request['questions']['action']['criteria']))
        original=set(guarded_indices(obs,actions,feats))|{i for i,f in enumerate(feats) if not f}
        eligible=eligible_indices(obs,actions,feats);chosen=max(eligible,key=lambda i:p[i])
        if set(eligible)-original:
            self.admissions['windows_with_added_safe_discards']+=1
            self.admissions['added_safe_discard_options']+=len(set(eligible)-original)
        self.admissions['selected_added_safe_discard']+=int(chosen not in original)
        self.choices[json.loads(actions[chosen].to_mjai())['type']]+=1
        return actions[chosen]
