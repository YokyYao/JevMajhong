"""Single-winner score scenarios computed only from the current public request.

Unknown ura and unconfirmed situational yaku are excluded, not read from labels.
The quoted ranks are after this hypothetical settlement, never final match ranks.
"""
from .context import _discard_symbol, placements

SCORING_NOTE = (
    " current_win_settlement quotes public single-winner baseline scores and immediate ranks. "
    "Excluded unknown bonuses and other winners can change the actual settlement. "
    "An immediate rank is not the final match rank; consider dealer continuation.")


def tiles(value):
    if isinstance(value, str):
        parts = value.split()
        if not parts or any(_discard_symbol(t) is None for t in parts):
            from desktop.jev_sidecar.tiles import parse
            parts = parse(value)
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        raise ValueError("Visible tile notation is required")
    result = [_discard_symbol(t) for t in parts]
    if any(t is None or (t[1] == "z" and t[0] not in "1234567") for t in result):
        raise ValueError("Invalid visible tile")
    return result


def tile_kind(tile):
    return "mpsz".index(tile[1]) * 9 + (4 if tile[0] == "0" else int(tile[0]) - 1)


def allocate_groups(groups):
    """Assign distinct physical IDs, preserving red fives across hand and melds."""
    used, result = set(), []
    for group in groups:
        ids = []
        for tile in group:
            kind = tile_kind(tile)
            candidates = [kind * 4] if tile[0] == "0" else [kind * 4 + i for i in range(4)
                if not (kind < 27 and kind % 9 == 4 and i == 0)]
            available = [x for x in candidates if x not in used]
            if not available:
                raise ValueError("Tile multiplicity or red-five count is impossible")
            ids.append(available[0]); used.add(available[0])
        result.append(ids)
    return result


def quote_win(payload):
    state, q = payload["state"], payload["questions"]["action"]
    wins = [v for v in q["criteria"].values() if v["type"] == "hora"]
    if not wins:
        return {"status": "no_legal_win_in_current_menu"}
    context = state.get("endgame_context", {})
    hand = tiles(state["hand"])
    method = context.get("win_method") or ("tsumo" if len(hand) % 3 == 2 else "ron")
    seat, dealer = state["seat"], state["dealer"]
    if dealer not in range(4) or state.get("dora_indicators") is None:
        return {"status": "public_scoring_metadata_incomplete"}
    payer = context.get("ron_payer") if method == "ron" else None
    if method == "ron" and (payer not in range(4) or payer == seat):
        return {"status": "current_ron_payer_unknown"}
    winner = (_discard_symbol(wins[0].get("pai"))
              or _discard_symbol(context.get("public_winning_tile"))
              or _discard_symbol(state.get("screen_observation", {}).get("winning_tile")))
    if method == "ron":
        winner = winner or context.get("public_discard_tile")
        if winner is None:
            return {"status": "current_winning_tile_unknown"}
        hand = [*hand, winner]
    own_melds = state["melds"][seat]
    if own_melds is None:
        return {"status": "own_melds_unknown"}
    meld_tiles = [tiles(m["tiles"]) for m in own_melds]
    if len(hand) + len(own_melds) * 3 != 14:
        return {"status": "incomplete_public_hand_structure"}
    groups = allocate_groups([hand, *meld_tiles])
    if winner is not None:
        win_ids = [i for i, tile in zip(groups[0], hand) if tile == winner]
        if not win_ids:
            return {"status": "winning_tile_not_in_own_hand"}
        win_ids = win_ids[-1:]
    else:
        # Native legacy records sometimes omit the drawn tile. Enumerate only
        # own concealed tile kinds; do not invent a particular winning wait.
        win_ids = list({i // 4: i for i in groups[0]}.values())
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.meld import Meld
    melds = []
    for source, ids in zip(own_melds, groups[1:]):
        kind = source["type"]
        if kind not in (0, 1, 2, 3, 4):
            return {"status": "unsupported_public_meld_type"}
        melds.append(Meld((Meld.CHI, Meld.PON, Meld.KAN, Meld.KAN, Meld.SHOUMINKAN)[kind],
                          ids, opened=kind != 3))
    indicators = [tile_kind(t) * 4 for t in tiles(state["dora_indicators"])]
    flags = state.get("screen_observation", {}).get("confirmed_win_flags", {})
    allowed_flags = {"ippatsu", "double_riichi", "haitei", "houtei", "rinshan", "chankan"}
    if set(flags) - allowed_flags or any(type(v) is not bool for v in flags.values()):
        raise ValueError("Only explicitly confirmed public win flags are permitted")
    riichi = seat in state.get("riichi", [])
    config = HandConfig(is_tsumo=method == "tsumo", is_riichi=riichi and not flags.get("double_riichi", False),
        is_daburu_riichi=riichi and flags.get("double_riichi", False),
        is_ippatsu=riichi and flags.get("ippatsu", False), is_rinshan=flags.get("rinshan", False),
        is_chankan=flags.get("chankan", False), is_haitei=flags.get("haitei", False),
        is_houtei=flags.get("houtei", False), player_wind=27 + (seat - dealer) % 4,
        round_wind=27 + state["round_wind"], tsumi_number=state["honba"],
        kyoutaku_number=state["sticks"], options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))
    physical = [i for group in groups for i in group]
    calculator, scenarios = HandCalculator(), []
    for win_id in win_ids:
        value = calculator.estimate_hand_value(physical, win_id, melds=melds,
            dora_indicators=indicators, ura_dora_indicators=[], config=config)
        if value.error:
            continue
        if any(y.is_yakuman for y in value.yaku) and any(m.opened for m in melds):
            return {"status": "open_yakuman_responsibility_not_in_public_state"}
        cost, delta = value.cost, [0] * 4
        for p in range(4):
            if p == seat:
                continue
            if method == "tsumo":
                delta[p] = -(cost["main"] + cost["main_bonus"] if p == dealer or seat == dealer
                             else cost["additional"] + cost["additional_bonus"])
            elif p == payer:
                delta[p] = -(cost["main"] + cost["main_bonus"])
        delta[seat] = -sum(delta) + cost["kyoutaku_bonus"]
        scores = [s + d for s, d in zip(state["scores"], delta)]
        scenarios.append({"han": value.han, "fu": value.fu, "deltas": delta,
                          "scores_after": scores, "rank_after": placements(scores)[seat]})
    if not scenarios:
        return {"status": "no_score_without_unconfirmed_special_yaku"}
    unique = {tuple(s["deltas"]): s for s in scenarios}
    scenarios = sorted(unique.values(), key=lambda s: (s["deltas"][seat], s["rank_after"]))
    return {"status": "public_single_winner_baseline_scenarios", "win_method": method,
        "ron_payer": payer, "winner_is_dealer": seat == dealer,
        "winning_tile_known": winner is not None, "scenario_count": len(scenarios),
        "minimum_baseline_scenario": scenarios[0], "maximum_baseline_scenario": scenarios[-1],
        "top2_in_all_baseline_scenarios": all(s["rank_after"] <= 2 for s in scenarios),
        "top2_in_any_baseline_scenario": any(s["rank_after"] <= 2 for s in scenarios),
        "excluded_unconfirmed_bonuses": sorted((allowed_flags - set(flags)) | ({"ura_dora"} if riichi else set())
                                                | {"first_turn_yakuman"}),
        "assumptions": ["one_winner_claims_the_pool", "excluded_bonuses_may_increase_actual_payment",
                        "scenario_range_only_covers_winning_tile_ambiguity", "immediate_placement_is_not_final_match_placement"]}


def settlement_request(request):
    """Versioned v3 input; v2 source requests and caches remain untouched."""
    import copy
    from .context import model_request
    source = copy.deepcopy(request)
    screen = source["state"].get("screen_observation")
    if isinstance(screen, dict):
        allowed = {"temporal_information", "metadata_confirmed", "confirmed_metadata",
                   "reaction", "winning_tile", "confirmed_win_flags"}
        source["state"]["screen_observation"] = {k: v for k, v in screen.items() if k in allowed}
        reaction = source["state"]["screen_observation"].get("reaction")
        if isinstance(reaction, dict):
            source["state"]["screen_observation"]["reaction"] = {
                k: v for k, v in reaction.items() if k in ("seat", "tile", "offered")}
    payload = model_request(source)
    state = payload["state"]
    winning = next((v for v in payload["questions"]["action"]["criteria"].values() if v["type"] == "hora"), {})
    state["endgame_context"]["public_winning_tile"] = (_discard_symbol(winning.get("pai"))
        or _discard_symbol(state.get("screen_observation", {}).get("winning_tile"))
        or (state["endgame_context"].get("public_discard_tile") if state["endgame_context"].get("win_method") == "ron" else None))
    state["endgame_context"]["current_win_settlement"] = quote_win(payload)
    payload["questions"]["action"]["instructions"] += SCORING_NOTE
    return payload
