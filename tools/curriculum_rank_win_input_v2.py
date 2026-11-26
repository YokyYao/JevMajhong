"""Win route with an explicit current PUBLIC event for native observations.

RiichiEnv.get_obs_py may return an empty events list. Never guess a ron
payer from an old river: the game loop can pass its current public event.
Input features and normalized requests remain identical to schema v1.
"""
import json
from tools.curriculum_rank_win_input_v1 import (
    serving_request,training_record,continuation_keys,select,RankWinPolicyClient,
    RankWinPolicyAgent as PreviousAgent,SCHEMA,INSTRUCTION)
from tools.curriculum_endgame import in_endgame_scope

METHOD='same_model_late_win_explicit_public_event_PASS_or_best_guarded_discard_v2'


class RankWinPolicyAgent(PreviousAgent):
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
        request,actions,features=build_win_request(obs,trigger)
        allowed=continuation_keys(obs,request,actions,features)
        result=self.client.system_one(**request)
        chosen=select(request,result['answers']['action']['probabilities'],allowed)
        keys=list(request['questions']['action']['criteria']);action=actions[keys.index(chosen)]
        self.diagnostics['late_win_model_calls']+=1
        self.diagnostics['explicit_current_public_event_calls']+=current_public_event is not None
        self.diagnostics['late_win_declines']+=action.action_type not in (ActionType.RON,ActionType.TSUMO)
        self.choices[json.loads(action.to_mjai())['type']]+=1
        return action

