"""Shared, player-visible representation for replay training and simulation."""
from collections import Counter
from functools import lru_cache
import base64
import json

from riichienv import ActionType, calculate_shanten
from riichienv.convert import tid_to_mpsz, mpsz_to_tid

INSTRUCTION = ("Choose the best legal action in four-player Mahjong Soul riichi mahjong. "
    "Maximize final placement: pursue first while avoiding costly deal-ins and last place. "
    "sh=shanten (lower is closer to tenpai); ukeire=visible remaining improving tiles; "
    "safe=opponent seats for which this tile is genbutsu. Probabilities are action preferences.")


def tile_name(tid):
    return tid_to_mpsz(int(tid))


def signature(action):
    value = json.loads(action.to_mjai())
    return json.dumps({k: sorted(v) if k == "consumed" else v
                       for k, v in value.items() if k not in ("actor", "target", "tsumogiri")}, sort_keys=True)


@lru_cache(maxsize=200000)
def shanten(types):
    return calculate_shanten([x * 4 for x in types])


def current_events(obs):
    events = [json.loads(x) if isinstance(x, str) else x for x in obs.events]
    start = max((i for i, e in enumerate(events) if e["type"] == "start_kyoku"), default=0)
    return events[start:]


def legal_candidates(obs):
    unique = {}
    for action in obs.legal_actions():
        unique.setdefault(signature(action), action)
    return [unique[key] for key in sorted(unique)]


def describe(obs):
    d = obs.to_dict()
    pid = obs.player_id
    # Strict allowlist: never serialize other hands or an environment/wall object.
    if any(hand for seat, hand in enumerate(d["hands"]) if seat != pid):
        raise ValueError("Observation exposes another player's concealed hand")
    visible = Counter(t // 4 for t in obs.hand)
    # Replay step observations contain synthetic initialization events in this
    # upstream release. Use the actual observation fields exclusively.
    for river in d["discards"]:
        visible.update(t // 4 for t in river)
    for melds in d["melds"]:
        for meld in melds:
            visible.update(t // 4 for t in meld.tiles)
            if meld.called_tile is not None:
                visible[meld.called_tile // 4] -= 1  # also present in the source river
    for tid in d["dora_indicators"]:
        visible[tid // 4] += 1
    if any(count > 4 for count in visible.values()):
        raise ValueError("Visible tile count exceeds four")
    state = {
        "seat": pid, "dealer": d["oya"], "round_wind": d["round_wind"],
        "honba": d["honba"], "sticks": d["riichi_sticks"], "scores": d["scores"],
        "hand": " ".join(tile_name(t) for t in sorted(obs.hand)),
        "dora_indicators": " ".join(tile_name(t) for t in d["dora_indicators"]),
        "riichi": [p for p in range(4) if d["riichi_declared"][p]],
        "rivers": [" ".join(tile_name(t) for t in river) for river in d["discards"]],
        "melds": [[{"type": int(m.meld_type), "tiles": " ".join(tile_name(t) for t in m.tiles)}
                   for m in melds] for melds in d["melds"]],
        "discards_count": sum(len(r) for r in d["discards"]),
        "hand_number": d["oya"] + 1,
    }
    public = json.loads(base64.b64decode(obs.serialize_to_base64()))
    for key in ("tsumogiri_flags", "riichi_sutehais", "last_tedashis", "last_discard"):
        state[key] = public[key]
    return state, visible, d


def features(obs, actions, visible, d):
    result = []
    for action in actions:
        values = {}
        if action.action_type == ActionType.DISCARD:
            remaining = list(obs.hand)
            remaining.remove(action.tile)
            types = tuple(sorted(t // 4 for t in remaining))
            distance = shanten(types)
            improving = [t for t in range(34) if visible[t] < 4
                         and shanten(tuple(sorted((*types, t)))) < distance]
            values = {"sh": distance, "ukeire": sum(4 - visible[t] for t in improving),
                      "safe": [p for p in range(4) if p != obs.player_id and
                               action.tile // 4 in {t // 4 for t in d["discards"][p]}],
                      "red": action.tile in (16, 52, 88)}
        result.append(values)
    return result


def build_request(obs):
    actions = legal_candidates(obs)
    state, visible, d = describe(obs)
    feats = features(obs, actions, visible, d)
    criteria = {}
    for i, (action, values) in enumerate(zip(actions, feats)):
        value = json.loads(action.to_mjai())
        for key in ("actor", "target", "tsumogiri"):
            value.pop(key, None)
        criteria[f"a{i}"] = {**value, **values}
    request = {"state": state, "questions": {"action": {
        "type": "choice", "instructions": INSTRUCTION, "criteria": criteria}}}
    return request, actions, feats


def heuristic_index(obs, actions, feats):
    for i, action in enumerate(actions):
        if action.action_type in (ActionType.RON, ActionType.TSUMO):
            return i
    d = obs.to_dict()
    threats = {p for p in range(4) if p != obs.player_id and d["riichi_declared"][p]}
    # Conservative closed-hand baseline. Decline calls, take riichi when no one threatens.
    if not threats:
        for i, action in enumerate(actions):
            if action.action_type == ActionType.RIICHI:
                return i
    discard = [(i, f) for i, f in enumerate(feats) if f]
    if discard:
        best_sh = min(f["sh"] for _, f in discard)
        fold = bool(threats) and best_sh >= 1
        return min(discard, key=lambda x: (
            len(threats - set(x[1]["safe"])) if fold else 0,
            x[1]["sh"], -x[1]["ukeire"], x[1]["red"], x[0]))[0]
    for i, action in enumerate(actions):
        if action.action_type == ActionType.PASS:
            return i
    return 0
