"""Public endgame context and terminal placement utility."""
import copy
import math

INSTRUCTION = (
    "Choose the best legal action in four-player Mahjong Soul East-only riichi mahjong. "
    "Maximize final match placement: pursue first and top two, avoid fourth and costly deal-ins. "
    "In late hands compare score gaps, honba, riichi sticks, payer, and dealer continuation. "
    "A small win may secure placement or permit continuation; do not reject it solely by face value. "
    "Declining ron can cause furiten. Use only currently public information. "
    "sh=shanten; ukeire=visible improving tiles; safe=genbutsu opponent seats. "
    "Probabilities are action preferences."
)
PLACEMENT_UTILITY = (1.0, 0.5, -0.5, -1.0)


def placements(scores):
    if len(scores) != 4 or any(not isinstance(v, int) for v in scores):
        raise ValueError("Four integer scores are required")
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    ranks = [0] * 4
    for rank, player in enumerate(order, 1):
        ranks[player] = rank
    return ranks


def public_context(state):
    seat = state["seat"]
    if seat not in range(4):
        raise ValueError("Invalid seat")
    scores = state["scores"]
    ranks = placements(scores)
    order = sorted(range(4), key=lambda p: ranks[p])
    wind, hand = state["round_wind"], state["hand_number"]
    if not isinstance(wind, int) or not isinstance(hand, int) or not 1 <= hand <= 4:
        raise ValueError("Verified round wind and within-round hand number required")
    rank = ranks[seat]
    target = order[1] if rank == 3 else order[2] if rank == 4 else None
    return {
        "game_mode": "four_player_east",
        "placement": rank,
        "placements": ranks,
        "late_match_window": wind > 0 or hand == 4,
        "late_window_is_not_guaranteed_terminal": True,
        "objective": "catch_up" if rank >= 3 else "preserve_and_improve",
        "target_seat": target,
        "gap_to_target": scores[target] - scores[seat] if target is not None else None,
        "gap_to_second": scores[order[1]] - scores[seat],
        "gap_to_third": scores[order[2]] - scores[seat],
        "lead_over_third": scores[seat] - scores[order[2]],
    }


def serving_request(request):
    # Use the released v2 feature transformation before adding only public context.
    from desktop.jev_sidecar.unified_policy import serving_request as play_v2_request
    payload = play_v2_request(request)
    payload["questions"]["action"]["instructions"] = INSTRUCTION
    payload["state"]["endgame_context"] = public_context(payload["state"])
    return payload


def terminal_utility(rank):
    if isinstance(rank, bool) or rank not in (1, 2, 3, 4):
        raise ValueError("Terminal placement must be 1..4")
    return PLACEMENT_UTILITY[rank - 1]


def target_from_utilities(values, temperature=0.25):
    if not values or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Finite positive target temperature required")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Finite terminal utilities required")
    maximum = max(values)
    weights = [math.exp((v - maximum) / temperature) for v in values]
    return [v / sum(weights) for v in weights]


def terminal_training_target(criteria, utilities, temperature=0.25):
    """Keep neutral legal wins; do not train uniform targets on unidentified menus."""
    keys = list(criteria)
    if len(keys) != len(utilities):
        raise ValueError("Terminal utility/menu size mismatch")
    probabilities = target_from_utilities(utilities, temperature)
    if max(utilities) - min(utilities) <= 1e-12:
        wins = [key for key in keys if criteria[key]["type"] == "hora"]
        if not wins:
            return None, "finite_MC_does_not_distinguish_keep_as_diagnostic"
        return {key: float(key == wins[0]) for key in keys}, "finite_MC_tie_preserve_existing_legal_win"
    return dict(zip(keys, probabilities)), "complete_terminal_MC_soft_target"


def settled_placement(scores, deltas, seat):
    if len(deltas) != 4:
        raise ValueError("Four actual settlement deltas required")
    return placements([s + d for s, d in zip(scores, deltas)])[seat]


def annotate_record(record):
    result = copy.deepcopy(record)
    result["state"].pop("endgame_context", None)
    payload = serving_request(result)
    result["state"] = payload["state"]
    result["questions"]["action"]["instructions"] = INSTRUCTION
    result["questions"]["action"]["criteria"] = payload["questions"]["action"]["criteria"]
    return result
