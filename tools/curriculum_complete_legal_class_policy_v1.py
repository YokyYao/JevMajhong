"""Prospective learned action execution over the complete native legal menu.

Keep the existing forced-win shortcuts and the fixed full-class RI0.5 rule.
The old shanten/risk filter is a shadow diagnostic only. This is not installed
in the active calls fit, production, or the frozen v8 comparison policy.
"""
from collections import Counter
import json

from tools.curriculum_riichi_class_policy_v1 import select as class_select

METHOD = 'complete_native_legal_menu_with_fixed_full_class_RI0_5_v1'


def select(types, probabilities):
    return class_select(types, probabilities, list(range(len(types))))


class CompleteLegalClassPolicyAgent:
    def __init__(self, client, validator=None):
        self.client = client; self.validator = validator
        self.choices = Counter(); self.diagnostics = Counter()

    def act(self, obs):
        from riichienv import ActionType
        from jevmahjong.core.state import build_request, legal_candidates
        from jevmahjong.evaluation.agents import guarded_indices
        actions = legal_candidates(obs)
        if len(actions) == 1:
            self.diagnostics['single_action_shortcuts'] += 1
            return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO):
                self.diagnostics['forced_win_shortcuts'] += 1
                return action
        validator = self.validator
        if validator is None:
            from jevany.api import validate_distribution
            validator = validate_distribution
        request, actions, features = build_request(obs)
        criteria = request['questions']['action']['criteria']; keys = list(criteria)
        raw = dict(self.client.system_one(**request)['answers']['action']['probabilities'])
        validator(dict(raw), keys)
        probabilities = [raw[k] for k in keys]; kinds = [criteria[k]['type'] for k in keys]
        chosen, riichi = select(kinds, probabilities)
        old = sorted(set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f})
        previous, _ = class_select(kinds, probabilities, old)
        self.diagnostics['model_decisions'] += 1
        self.diagnostics['RI_class_scope_windows'] += riichi
        self.diagnostics['chosen_previously_excluded'] += chosen not in old
        self.diagnostics['different_from_original_RI_class_choice'] += chosen != previous
        self.choices[json.loads(actions[chosen].to_mjai())['type']] += 1
        return actions[chosen]
