"""Optional candidate policy admitting genbutsu for an open tenpai hand.

The original reference policy stays in agents.py. This policy adds legal safe
discards to the existing admissible set; the model still selects the action.
"""
from collections import Counter
import json
from riichienv import ActionType
from jevmahjong.core.state import build_request, legal_candidates
from jevmahjong.evaluation.agents import PolicyAgent, guarded_indices


def eligible_indices(obs, actions, feats):
    original = sorted(set(guarded_indices(obs,actions,feats)) | {i for i,f in enumerate(feats) if not f})
    state = obs.to_dict(); seat = obs.player_id
    threats = {p for p in range(4) if p != seat and state['riichi_declared'][p]}
    discards = [f for f in feats if f]
    if (not threats or state['riichi_declared'][seat]
            or not any(m.opened for m in state['melds'][seat])
            or not discards or min(f['sh'] for f in discards) != 0):
        return original
    safe = {i for i,f in enumerate(feats) if f and threats.issubset(set(f['safe']))}
    return sorted(set(original) | safe)


class DefensePolicyAgent(PolicyAgent):
    def __init__(self,client):
        super().__init__(client)
        self.admissions = Counter()

    def act(self,obs):
        from jevany.api import validate_distribution
        actions = legal_candidates(obs)
        if len(actions) == 1: return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON,ActionType.TSUMO): return action
        request,actions,feats = build_request(obs)
        result = self.client.system_one(**request)
        p,_ = validate_distribution(result['answers']['action']['probabilities'],
            list(request['questions']['action']['criteria']))
        old = set(guarded_indices(obs,actions,feats)) | {i for i,f in enumerate(feats) if not f}
        eligible = eligible_indices(obs,actions,feats)
        chosen = max(eligible,key=lambda i:p[i])
        if set(eligible)-old:
            self.admissions['windows_with_added_safe_discards'] += 1
            self.admissions['added_safe_discard_options'] += len(set(eligible)-old)
        self.admissions['selected_added_safe_discard'] += int(chosen not in old)
        self.choices[json.loads(actions[chosen].to_mjai())['type']] += 1
        return actions[chosen]
