"""Experimental native action selection; never installed into live serving."""
from collections import Counter
import json


class AllLegalPolicyAgent:
    """One ordinary forward pass; forced legal wins and all legal candidates.

    Compute the old guard only as a shadow diagnostic. It never changes the
    experimental choice. This is an ablation agent, not a deployment default.
    """
    def __init__(self, client):
        self.client = client
        self.choices = Counter()
        self.shadow = Counter()

    def act(self, obs):
        from riichienv import ActionType
        from jevany.api import validate_distribution
        from jevmahjong.core.state import legal_candidates, build_request
        from jevmahjong.evaluation.agents import guarded_indices
        actions = legal_candidates(obs)
        if len(actions) == 1:
            return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO):
                return action
        request, actions, features = build_request(obs)
        answer = self.client.system_one(**request)
        keys = list(request['questions']['action']['criteria'])
        probabilities, _ = validate_distribution(answer['answers']['action']['probabilities'], keys)
        best = max(range(len(actions)), key=lambda i: probabilities[i])
        old_allowed = set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f}
        self.shadow['model_decisions'] += 1
        self.shadow['raw_choice_excluded_by_original_guard'] += best not in old_allowed
        self.choices[json.loads(actions[best].to_mjai())['type']] += 1
        return actions[best]
