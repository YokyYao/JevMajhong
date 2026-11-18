"""One model policy with explicit released-v2 and endgame action paths."""
import json
from collections import Counter
from riichienv import ActionType
from jevmahjong.core.state import build_request, legal_candidates
from jevmahjong.evaluation.agents import guarded_indices
from .context import model_request, placements


class ModelAgent:
    def __init__(self, model, placement_aware=False, ordinary_guard=False, protect_leading_wins=False,
                 force_all_wins=False):
        self.model = model
        self.placement_aware = placement_aware
        self.ordinary_guard = ordinary_guard
        self.protect_leading_wins = protect_leading_wins
        self.force_all_wins = force_all_wins
        self.choices = Counter()

    def act(self, obs):
        actions = legal_candidates(obs)
        if len(actions) == 1:
            return actions[0]
        late = int(obs.round_wind) > 0 or int(obs.kyoku_index) == 3
        compare_all = self.placement_aware and late
        if not compare_all or self.force_all_wins:
            for action in actions:
                if action.action_type in (ActionType.RON, ActionType.TSUMO):
                    return action
        request, actions, feats = build_request(obs)
        request["state"]["hand_number"] = int(obs.kyoku_index) + 1
        has_win = any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in actions)
        if (compare_all and has_win and self.protect_leading_wins
                and placements(request["state"]["scores"])[request["state"]["seat"]] <= 2):
            return next(a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO))
        if self.placement_aware:
            payload = model_request(request)
        else:
            from desktop.jev_sidecar.unified_policy import serving_request as baseline_request
            payload = baseline_request(request)
        from jevany.api import validate_distribution
        result = self.model.system_one(**payload)
        keys = list(payload["questions"]["action"]["criteria"])
        probabilities, _ = validate_distribution(result["answers"]["action"]["probabilities"], keys)
        eligible = list(range(len(actions))) if compare_all and (has_win or not self.ordinary_guard) else sorted(
            set(guarded_indices(obs, actions, feats)) | {i for i, f in enumerate(feats) if not f})
        chosen = max(eligible, key=lambda i: probabilities[i])
        self.choices[json.loads(actions[chosen].to_mjai())["type"]] += 1
        return actions[chosen]

