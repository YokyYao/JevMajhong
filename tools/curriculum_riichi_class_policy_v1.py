"""Experimental fixed RI-class selection; never installed into production."""
from collections import Counter
import json
import math

METHOD='riichi_full_class_0_5_then_guarded_discard_v1'


def select(types,probabilities,eligible):
    if (not types or len(types)!=len(probabilities) or not eligible
        or any(type(i) is not int or not 0<=i<len(types) for i in eligible)
        or any(not math.isfinite(p) or p<0 for p in probabilities)
        or abs(sum(probabilities)-1)>1e-5):raise ValueError('Aligned valid probabilities and eligible indices required')
    allowed=sorted(set(eligible));baseline=max(allowed,key=probabilities.__getitem__)
    if set(types)!={'reach','dahai'}:return baseline,False
    reaches=[i for i,t in enumerate(types) if t=='reach']
    discards=[i for i in allowed if types[i]=='dahai']
    if len(reaches)!=1 or reaches[0] not in allowed or not discards:
        raise ValueError('RI-only scope must retain its one legal reach and a guarded discard')
    chosen=reaches[0] if probabilities[reaches[0]]>=.5 else max(discards,key=probabilities.__getitem__)
    return chosen,True


class RiichiClassPolicyAgent:
    def __init__(self,client,validator=None):
        self.client=client;self.validator=validator;self.choices=Counter();self.diagnostics=Counter()

    def act(self,obs):
        from riichienv import ActionType
        from jevmahjong.core.state import build_request,legal_candidates
        actions=legal_candidates(obs)
        if len(actions)==1:return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON,ActionType.TSUMO):return action
        from jevmahjong.evaluation.agents import guarded_indices
        validator=self.validator
        if validator is None:
            from jevany.api import validate_distribution
            validator=validate_distribution
        request,actions,features=build_request(obs);criteria=request['questions']['action']['criteria'];keys=list(criteria)
        answer=self.client.system_one(**request);raw=dict(answer['answers']['action']['probabilities'])
        validator(dict(raw),keys)
        # Preserve the existing cached diagnostic's raw0.5/tie semantics.
        # Relative argmax order is unchanged by ordinary sum renormalization.
        probabilities=[raw[k] for k in keys]
        eligible=sorted(set(guarded_indices(obs,actions,features))|{i for i,f in enumerate(features) if not f})
        chosen,applied=select([criteria[k]['type'] for k in keys],probabilities,eligible)
        baseline=max(eligible,key=probabilities.__getitem__)
        self.diagnostics['model_decisions']+=1
        if applied:
            self.diagnostics['RI_scope_windows']+=1
            self.diagnostics['RI_selected' if criteria[keys[chosen]]['type']=='reach' else 'RI_declined']+=1
        self.diagnostics['different_from_raw_guarded']+=chosen!=baseline
        self.choices[json.loads(actions[chosen].to_mjai())['type']]+=1
        return actions[chosen]
