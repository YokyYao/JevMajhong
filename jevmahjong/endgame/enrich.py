"""Recover current public win context by replaying only each selected prefix."""
import json
from pathlib import Path
from riichienv import ActionType, GameRule, RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request
from jevmahjong.evaluation.agents import guarded_indices
from jevmahjong.training.futures import replay_prefix
from .collect import digest, write
from .context import serving_request
from .native_public import add_native_context
from .win_value import settlement_request


def enrich_positions(config):
    out = Path(config["out"])
    cache = out / "public-context-private.json"
    if cache.exists():
        raise ValueError("Existing public-context cache must be inspected, not overwritten")
    positions = json.loads((out / "selected-positions-private.json").read_text())
    sources, rows, counts = {}, {}, {}
    for position in positions:
        seed, seat = position["seed"], position["seat"]
        if seed not in sources:
            sources[seed] = json.loads((out / "sources" / (str(seed) + ".json")).read_text())
        env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_mjsoul(),
                        seed=engine_seed(seed, "spaced-v1"))
        replay_prefix(env, sources[seed]["mjai_log"], position["step"])
        obs = env.get_observations()[seat]
        request, actions, features = build_request(obs)
        request["state"]["hand_number"] = env.kyoku_idx + 1
        if digest(serving_request(request)) != position["request_sha256"]:
            raise ValueError("Source request changed before public enrichment")
        enriched = add_native_context(request, obs, actions)
        keys = list(request["questions"]["action"]["criteria"])
        wins = [i for i, a in enumerate(actions) if a.action_type in (ActionType.RON, ActionType.TSUMO)]
        eligible = wins[:1] if wins else sorted(set(guarded_indices(obs, actions, features))
                                                | {i for i, f in enumerate(features) if not f})
        row_id = digest([seed, position["step"], seat, position["request_sha256"]])
        rows[row_id] = {"source_request_sha256": position["request_sha256"],
                       "screen_observation": enriched["state"].get("screen_observation", {}),
                       "original_policy_eligible_keys": [keys[i] for i in eligible]}
        quote = settlement_request(enriched)["state"]["endgame_context"]["current_win_settlement"]
        counts[quote["status"]] = counts.get(quote["status"], 0) + 1
        write(out / "public-context-status-private.json", {"status": "recovering_current_public_context",
            "positions_complete": len(rows), "positions_total": len(positions)})
    write(cache, {"schema": "current-public-win-context-v1", "rows": rows,
                  "source_prefix_only": True, "future_results_used": False})
    result = {"status": "public_context_complete", "positions": len(rows), "quote_status_counts": counts,
              "future_results_used": False, "original_requests_and_terminal_labels_unchanged": True}
    write(out / "public-context-status-private.json", result)
    return result
