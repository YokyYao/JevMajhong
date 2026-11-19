"""Single-model native agent with fourth-place risk-taking action scope."""
import json

from .policy import ModelAgent
from .context import placements,TRAILING_INPUT_VERSION
from .fourth_chase import fourth_chase_action_keys
from .native_public import add_native_context
from .win_value import settlement_request
from .reference import distribution
from jevmahjong.core.state import build_request,legal_candidates


class FourthChaseAgent:
    def __init__(self,model,placement_aware=True,**kwargs):
        kwargs.setdefault('input_version',TRAILING_INPUT_VERSION)
        self.base=ModelAgent(model,placement_aware=placement_aware,**kwargs)
        self.model=model;self.placement_aware=placement_aware;self.choices=self.base.choices

    def act(self,obs):
        late=int(obs.round_wind)>0 or int(obs.kyoku_index)==3
        if (not self.placement_aware or not late or self.base.force_all_wins
                or placements(obs.to_dict()['scores'])[obs.player_id]!=4):
            return self.base.act(obs)
        actions=legal_candidates(obs)
        if len(actions)==1:return actions[0]
        request,actions,_=build_request(obs);request['state']['hand_number']=int(obs.kyoku_index)+1
        payload=settlement_request(add_native_context(request,obs,actions))
        keys=list(payload['questions']['action']['criteria']);eligible=fourth_chase_action_keys(payload)
        if len(eligible)==1:return actions[keys.index(eligible[0])]
        answer=self.model.system_one(**payload)
        probabilities=answer['answers']['action']['probabilities'];distribution(probabilities,keys)
        chosen=max(eligible,key=probabilities.get);action=actions[keys.index(chosen)]
        self.choices[json.loads(action.to_mjai())['type']]+=1
        return action
