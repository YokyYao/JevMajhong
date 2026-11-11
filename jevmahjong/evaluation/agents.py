import base64
from collections import Counter
import hashlib
import json
import torch
from riichienv import ActionType
from jevmahjong.core.state import (build_request, describe, features, heuristic_index, legal_candidates, shanten)

class EfficiencyAgent:
    def act(self, obs):
        actions = legal_candidates(obs)
        if len(actions) == 1:
            return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO):
                return action
        _, visible, d = describe(obs)
        feats = features(obs, actions, visible, d)
        return actions[heuristic_index(obs, actions, feats)]


def guarded_indices(obs, actions, feats):
    """The common admissible action set for deployment and head-only RL."""
    eligible = list(range(len(actions)))
    if any(feats):
        ref = heuristic_index(obs, actions, feats)
        if feats[ref]:
            d = obs.to_dict()
            threats = {p for p in range(4) if p != obs.player_id and d["riichi_declared"][p]}
            fold = bool(threats) and min(f["sh"] for f in feats if f) >= 1
            ref_risk = len(threats - set(feats[ref]["safe"]))
            eligible = [i for i, f in enumerate(feats) if f and f["sh"] <= feats[ref]["sh"]
                        and (not fold or len(threats - set(f["safe"])) <= ref_risk)]
    if not eligible:
        raise ValueError("Guarded action set is empty")
    return eligible

class CallingOpponent(EfficiencyAgent):
    """Rule opponent only; never a supervision source for the learner."""
    def act(self, obs):
        actions = legal_candidates(obs)
        if len(actions) == 1: return actions[0]
        wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
        if wins: return wins[0]
        d = obs.to_dict(); before = shanten(tuple(sorted(t // 4 for t in obs.hand)))
        for action in actions:
            if action.action_type not in (ActionType.CHI, ActionType.PON): continue
            tile = action.tile // 4
            valued = action.action_type == ActionType.PON and (tile >= 31 or tile == 27 + d['round_wind']
                         or tile == 27 + (obs.player_id - d['oya']) % 4)
            visible_own = list(obs.hand) + [action.tile] + [t for meld in d['melds'][obs.player_id] for t in meld.tiles]
            tanyao = all(t // 4 < 27 and (t // 4) % 9 not in (0, 8) for t in visible_own)
            if not (valued or tanyao): continue
            remaining = list(obs.hand)
            for tile_id in action.consume_tiles: remaining.remove(tile_id)
            after = min(shanten(tuple(sorted(t // 4 for j, t in enumerate(remaining) if j != i)))
                        for i in range(len(remaining)))
            if after < before: return action
        return super().act(obs)


class PolicyAgent:
    def __init__(self, client):
        self.client = client
        self.choices = Counter()

    def act(self, obs):
        from jevany.api import validate_distribution
        actions = legal_candidates(obs)
        if len(actions) == 1: return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO): return action
        request, actions, feats = build_request(obs)
        result = self.client.system_one(**request)
        p, _ = validate_distribution(result['answers']['action']['probabilities'], list(request['questions']['action']['criteria']))
        # Guard discards, without silently dropping legal reach/kan with empty features.
        eligible = sorted(set(guarded_indices(obs, actions, feats)) | {i for i, f in enumerate(feats) if not f})
        chosen = actions[max(eligible, key=lambda i: p[i])]
        self.choices[json.loads(chosen.to_mjai())['type']] += 1
        return chosen
