"""Prospective defensive action support; not used by any active fit or release.

Only ordinary discard menus (optionally also offering reach) facing an accepted
opponent RI, before our own RI, use all legal discards. Open and closed hands
are both in scope. The frozen RI-class choice itself is unchanged; all other
menus retain the original guard, including any mixed kan menu.
"""
from collections import Counter
import json

from tools.curriculum_riichi_class_policy_v1 import select

METHOD = 'RI_class_with_all_legal_defensive_discard_scope_v1'


def in_scope(types, state):
    seat = state.get('seat'); riichi = state.get('riichi')
    if (type(seat) is not int or not 0 <= seat < 4 or not isinstance(riichi, list)
            or len(riichi) != len(set(riichi))
            or any(type(p) is not int or not 0 <= p < 4 for p in riichi)
            or not types or any(not isinstance(t, str) for t in types)):
        raise ValueError('Verified public seat and RI seat-index list required')
    kinds = set(types)
    return ('dahai' in kinds and kinds <= {'dahai', 'reach'}
            and seat not in riichi and any(p != seat for p in riichi))


def eligible(types, state, original):
    if not original or any(type(i) is not int or not 0 <= i < len(types) for i in original):
        raise ValueError('Nonempty aligned original guarded set required')
    applies = in_scope(types, state)
    return (list(range(len(types))) if applies else sorted(set(original))), applies


class RiichiRiskScopePolicyAgent:
    def __init__(self, client, validator=None):
        self.client = client; self.validator = validator
        self.choices = Counter(); self.diagnostics = Counter()

    def act(self, obs):
        from riichienv import ActionType
        from jevmahjong.core.state import build_request, legal_candidates
        from jevmahjong.evaluation.agents import guarded_indices
        actions = legal_candidates(obs)
        if len(actions) == 1: return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO): return action
        validator = self.validator
        if validator is None:
            from jevany.api import validate_distribution
            validator = validate_distribution
        request, actions, features = build_request(obs)
        criteria = request['questions']['action']['criteria']; keys = list(criteria)
        raw = dict(self.client.system_one(**request)['answers']['action']['probabilities'])
        validator(dict(raw), keys)
        probabilities = [raw[k] for k in keys]; types = [criteria[k]['type'] for k in keys]
        old = sorted(set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f})
        allowed, risk = eligible(types, request['state'], old)
        chosen, riichi = select(types, probabilities, allowed)
        baseline, _ = select(types, probabilities, old)
        self.diagnostics['model_decisions'] += 1
        self.diagnostics['defensive_scope_windows'] += risk
        self.diagnostics['RI_class_scope_windows'] += riichi
        self.diagnostics['different_from_original_RI_class_choice'] += chosen != baseline
        self.diagnostics['chosen_previously_excluded'] += chosen not in old
        self.choices[json.loads(actions[chosen].to_mjai())['type']] += 1
        return actions[chosen]
