"""Public action constraints shared by training selection and native serving."""
from types import SimpleNamespace


def original_policy_keys(payload):
    state, criteria = payload["state"], payload["questions"]["action"]["criteria"]
    keys = list(criteria)
    wins = [k for k in keys if criteria[k]["type"] == "hora"]
    if wins:
        return wins[:1]
    # This branch is used for offline/native guard reconstruction. The desktop
    # win/pass gate above must remain portable and must not import Torch or
    # the native simulator merely to compare public scores.
    from riichienv import ActionType
    from jevmahjong.evaluation.agents import guarded_indices
    actions, features = [], []
    for key in keys:
        value = criteria[key]
        kind = {"dahai": ActionType.DISCARD, "reach": ActionType.RIICHI}.get(value["type"], ActionType.PASS)
        actions.append(SimpleNamespace(action_type=kind))
        features.append({k: value[k] for k in ("sh", "ukeire", "safe", "red")}
                        if value["type"] == "dahai" and value.get("sh") is not None else {})
    riichi = state.get("riichi") or []
    obs = SimpleNamespace(player_id=state["seat"],
                          to_dict=lambda: {"riichi_declared": [p in riichi for p in range(4)]})
    eligible = sorted(set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f})
    return [keys[i] for i in eligible]


def win_pass_allowed(payload):
    context = payload["state"].get("endgame_context", {})
    quote = context.get("current_win_settlement", {})
    return bool(context.get("late_match_window") and context.get("placement", 0) >= 3
        and quote.get("status") == "public_single_winner_baseline_scenarios"
        and not quote["winner_is_dealer"] and not quote["top2_in_any_baseline_scenario"])


def action_keys(payload):
    criteria = payload["questions"]["action"]["criteria"]
    if any(v["type"] == "hora" for v in criteria.values()) and win_pass_allowed(payload):
        return list(criteria)
    return original_policy_keys(payload)
