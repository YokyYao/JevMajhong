"""Explicit same-model complete-current-menu late win route.

The placement input projection is unchanged. Early wins and unsupported
current robbed-kan windows keep the existing shortcut. Ordinary windows
keep full-legal RI0.5 execution. This research route is not installed in
production or in the active stage3 training.
"""
from collections import Counter
import json
import math

from tools.curriculum_endgame import in_endgame_scope
from tools.curriculum_placement_input_v1 import PlacementPolicyClient
from tools.curriculum_complete_legal_class_policy_v1 import CompleteLegalClassPolicyAgent

METHOD='all_late_same_model_complete_current_win_menu_and_full_RI_route_v2'


def select(request,probabilities):
    criteria=request['questions']['action']['criteria'];keys=list(criteria)
    wins=[k for k,v in criteria.items() if v['type']=='hora']
    if (len(wins)!=1 or len(keys)<2 or set(probabilities)!=set(keys)
        or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in probabilities.values())
        or abs(sum(probabilities.values())-1)>1e-5):
        raise ValueError('One legal win and aligned complete native preference distribution required')
    win=wins[0];other=max((k for k in keys if k!=win),key=probabilities.__getitem__)
    return win if probabilities[win]>=probabilities[other] else other


class CompleteWinPlacementPolicyAgent:
    def __init__(self,client,validator=None):
        self.client=client
        self.ordinary=CompleteLegalClassPolicyAgent(client,validator)
        self.choices=self.ordinary.choices;self.diagnostics=self.ordinary.diagnostics

    def act(self,obs,*,current_public_event=None):
        from riichienv import ActionType
        from jevmahjong.core.state import legal_candidates,build_request
        from tools.curriculum_win_context import build_win_request
        actions=legal_candidates(obs)
        wins=[a for a in actions if a.action_type in (ActionType.RON,ActionType.TSUMO)]
        if len(actions)==1 or not wins:return self.ordinary.act(obs)
        if not in_endgame_scope(build_request(obs)[0]['state']):return self.ordinary.act(obs)
        if len(wins)!=1 or (current_public_event is None and not obs.events):
            self.diagnostics['unsupported_win_shortcuts']+=1;return wins[0]
        trigger=current_public_event if current_public_event is not None else obs.events[-1]
        trigger=json.loads(trigger) if isinstance(trigger,str) else trigger
        if trigger['type'] not in ('dahai','tsumo'):
            self.diagnostics['current_kan_win_shortcuts']+=1;return wins[0]
        request,actions,_=build_win_request(obs,trigger)
        result=self.client.system_one(**request)
        chosen=select(request,result['answers']['action']['probabilities'])
        keys=list(request['questions']['action']['criteria']);action=actions[keys.index(chosen)]
        self.diagnostics['late_full_win_menu_model_calls']+=1
        self.diagnostics['explicit_current_public_event_calls']+=current_public_event is not None
        self.diagnostics['late_win_declines']+=action.action_type not in (ActionType.RON,ActionType.TSUMO)
        self.choices[json.loads(action.to_mjai())['type']]+=1
        return action


def agent(model,validator=None):
    return CompleteWinPlacementPolicyAgent(PlacementPolicyClient(model),validator)

