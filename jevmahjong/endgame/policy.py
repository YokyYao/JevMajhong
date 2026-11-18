"""One model policy with explicit released-v2 and endgame action paths."""
import json
from collections import Counter
from riichienv import ActionType
from jevmahjong.core.state import build_request, legal_candidates
from jevmahjong.evaluation.agents import guarded_indices
from .context import serving_request


class ModelAgent:
    def __init__(self, model, placement_aware=False):
        self.model = model
        self.placement_aware = placement_aware
        self.choices = Counter()

    def act(self, obs):
        from jevany.api import validate_distribution
        actions = legal_candidates(obs)
        if len(actions) == 1:
            return actions[0]
        late = int(obs.round_wind) > 0 or int(obs.kyoku_index) == 3
        compare_all = self.placement_aware and late
        if not compare_all:
            for action in actions:
                if action.action_type in (ActionType.RON, ActionType.TSUMO):
                    return action
        request, actions, feats = build_request(obs)
        request["state"]["hand_number"] = int(obs.kyoku_index) + 1
        if self.placement_aware:
            payload = serving_request(request)
        else:
            from desktop.jev_sidecar.unified_policy import serving_request as baseline_request
            payload = baseline_request(request)
        result = self.model.system_one(**payload)
        keys = list(payload["questions"]["action"]["criteria"])
        probabilities, _ = validate_distribution(result["answers"]["action"]["probabilities"], keys)
        eligible = list(range(len(actions))) if compare_all else sorted(
            set(guarded_indices(obs, actions, feats)) | {i for i, f in enumerate(feats) if not f})
        chosen = max(eligible, key=lambda i: probabilities[i])
        self.choices[json.loads(actions[chosen].to_mjai())["type"]] += 1
        return actions[chosen]
