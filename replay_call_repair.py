"""Answer-independent chi/pon candidates for RiichiEnv 0.4.11 replay data.

The replay iterator injects a chosen meld with the called tile in consumed and
can omit unchosen alternatives. Never serialize that answer-dependent list.
Native simulation/inference continues using semantic.build_request unchanged.
"""
from collections import Counter
from itertools import combinations
import json

from riichienv import Action, ActionType
from semantic import build_request, shanten

REPAIR_VERSION = 'public-chi-pon-v1'


class UnsafeReplayCall(ValueError):
    pass


def canonical_action(action):
    value = json.loads(action.to_mjai()) if hasattr(action, 'to_mjai') else dict(action)
    value = {k: v for k, v in value.items() if k not in ('actor', 'target', 'tsumogiri')}
    expected = {'chi': 2, 'pon': 2, 'daiminkan': 3}.get(value['type'])
    if expected:
        consumed = list(value.get('consumed', []))
        if len(consumed) == expected + 1:
            if value.get('pai') not in consumed:
                raise UnsafeReplayCall('called_tile_missing_from_full_meld')
            consumed.remove(value['pai'])
        if len(consumed) != expected:
            raise UnsafeReplayCall('invalid_consumed_count')
        value['consumed'] = sorted(consumed)
    elif 'consumed' in value:
        value['consumed'] = sorted(value['consumed'])
    return value


def canonical_signature(action):
    return json.dumps(canonical_action(action), sort_keys=True)


def public_reaction_actions(obs):
    """Reconstruct safe chi/pon sets without inspecting the expert answer.

    Kan-capable and possibly winning states are excluded explicitly until their
    full live legality (rinshan availability, furiten, yaku) is independently
    reconstructed. Exclusions are not converted into pass training targets.
    """
    hand, called, player = list(obs.hand), obs.last_discard, obs.player_id
    d = obs.to_dict()
    if called is None or len(hand) % 3 != 1:
        raise UnsafeReplayCall('missing_reaction_context')
    owners = [p for p, river in enumerate(d['discards']) if river and river[-1] == called]
    if len(owners) != 1 or owners[0] == player:
        raise UnsafeReplayCall('discard_source_unresolved')
    if d['riichi_declared'][player]:
        raise UnsafeReplayCall('riichi_reaction_requires_ron_legality')
    tile_type = called // 4
    if sum(t // 4 == tile_type for t in hand) >= 3:
        raise UnsafeReplayCall('kan_capable_requires_live_legality')
    if shanten(tuple(sorted([t // 4 for t in hand] + [tile_type]))) == -1:
        raise UnsafeReplayCall('winning_shape_requires_ron_legality')
    actions = {canonical_signature({'type': 'none'}): Action(ActionType.PASS)}
    for consumed in combinations(hand, 2):
        types = sorted([t // 4 for t in consumed] + [tile_type])
        if len(set(types)) == 1:
            kind = ActionType.PON
            forbidden = {tile_type}
        elif owners[0] == (player + 3) % 4 and types[0] < 27 and types[0] // 9 == types[2] // 9 and types == list(range(types[0], types[0] + 3)):
            kind = ActionType.CHI
            used = sorted(t // 4 for t in consumed)
            forbidden = {tile_type}
            if used[1] - used[0] == 1:
                forbidden |= {t for t in (used[0] - 1, used[1] + 1) if 0 <= t < 27 and t // 9 == tile_type // 9}
        else:
            continue
        remaining = list(hand)
        for tile in consumed:
            remaining.remove(tile)
        if not any(t // 4 not in forbidden for t in remaining):
            continue
        action = Action(kind, tile=called, consume_tiles=list(consumed))
        actions.setdefault(canonical_signature(action), action)
    return [actions[key] for key in sorted(actions)]


def build_replay_request(obs):
    payload, original, _ = build_request(obs)
    kinds = {json.loads(a.to_mjai())['type'] for a in original}
    audit = Counter()
    if kinds & {'chi', 'pon', 'daiminkan'}:
        actions = public_reaction_actions(obs)
        canonical_original = {canonical_signature(a) for a in original}
        canonical_rebuilt = {canonical_signature(a) for a in actions}
        audit['reaction_records_rebuilt'] = 1
        audit['missing_alternatives_restored'] = len(canonical_rebuilt - canonical_original)
        audit['nonlive_candidates_removed'] = len(canonical_original - canonical_rebuilt)
        audit['malformed_call_candidates_replaced'] = sum(
            len(json.loads(a.to_mjai()).get('consumed', [])) != {'chi': 2, 'pon': 2, 'daiminkan': 3}.get(json.loads(a.to_mjai())['type'], 0)
            for a in original)
    else:
        actions = original
    if kinds & {'chi', 'pon', 'daiminkan'}:
        payload['questions']['action']['criteria'] = {f'a{i}': canonical_action(a) for i, a in enumerate(actions)}
    return payload, [canonical_signature(a) for a in actions], audit
