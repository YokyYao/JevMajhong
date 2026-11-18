"""Collect outcome-blind late-match windows from reproducible native matches."""
import hashlib
import json
from pathlib import Path
from riichienv import GameRule, RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request, legal_candidates
from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.evaluation.agents import EfficiencyAgent, CallingOpponent
from .context import placements, serving_request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def native_match(seed, per_group=4):
    env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_mjsoul(),
                   seed=engine_seed(seed, "spaced-v1"))
    observations = env.reset()
    ordinary, caller = EfficiencyAgent(), CallingOpponent()
    pools = {"win": [], "third": [], "fourth": [], "hold": []}
    boundaries = []
    previous = None
    for step in range(30000):
        if env.done():
            break
        marker = (env.round_wind, env.kyoku_idx, env.honba)
        if marker != previous:
            boundaries.append({"step": step, "round_wind": env.round_wind,
                "hand_number": env.kyoku_idx + 1, "honba": env.honba,
                "sticks": env.riichi_sticks, "scores": list(env.scores()),
                "placements": list(env.ranks())})
            previous = marker
        if env.round_wind > 0 or env.kyoku_idx == 3:
            for seat, obs in observations.items():
                actions = legal_candidates(obs)
                if len(actions) < 2:
                    continue
                request, _, _ = build_request(obs)
                actual_hand = int(obs.kyoku_index) + 1
                if actual_hand != env.kyoku_idx + 1:
                    raise ValueError("Observation/native hand number mismatch")
                request["state"]["hand_number"] = actual_hand
                payload = serving_request(request)
                if placements(payload["state"]["scores"]) != list(env.ranks()):
                    raise ValueError("Public score tie-break differs from native ranks")
                kinds = {v["type"] for v in payload["questions"]["action"]["criteria"].values()}
                rank = payload["state"]["endgame_context"]["placement"]
                bucket = "win" if "hora" in kinds else "third" if rank == 3 else "fourth" if rank == 4 else "hold"
                row = {"seed": seed, "step": step, "seat": seat, "bucket": bucket,
                       "request": payload, "request_sha256": digest(payload)}
                row["selection_key"] = digest([seed, row["request_sha256"], "endgame-v1"])
                pools[bucket].append(row)
        chosen = {p: (caller if p == (seed % 4) else ordinary).act(obs)
                  for p, obs in observations.items()}
        observations = env.step(chosen)
    if not env.done():
        raise RuntimeError("Bounded source match did not terminate")
    ledger = settlement_ledger(env.mjai_log, env.scores())
    chosen = []
    used = set()
    for bucket in ("win", "third", "fourth", "hold"):
        for row in sorted(pools[bucket], key=lambda x: x["selection_key"]):
            if row["request_sha256"] not in used:
                chosen.append(row)
                used.add(row["request_sha256"])
                break
    chosen = chosen[:per_group]
    return {"schema": "endgame-native-source-v1", "seed": seed,
        "source_policy": "three-efficiency-plus-one-calling-rule-opponent",
        "selected_before_future_labels": True, "selected": chosen,
        "bucket_opportunities": {k: len(v) for k, v in pools.items()},
        "hand_boundaries": boundaries, "final_scores": list(env.scores()),
        "final_ranks": list(env.ranks()), "ledger": ledger, "mjai_log": env.mjai_log,
        "source_steps": step}


def collect(config):
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {}
    for group in range(config["source_groups"]):
        seed = config["seed_start"] + group
        split = "train" if group < config["train_groups"] else "validation"
        file = out / "sources" / (str(seed) + ".json")
        if file.exists():
            source = json.loads(file.read_text())
            if source["seed"] != seed or source["schema"] != "endgame-native-source-v1":
                raise ValueError("Existing source identity mismatch")
        else:
            source = native_match(seed)
            write(file, source)
        for row in source["selected"]:
            row = dict(row, split=split, group_id="endgame-native:" + str(seed))
            rows.append(row)
            key = split + ":" + row["bucket"]
            counts[key] = counts.get(key, 0) + 1
        write(out / "collection-status-private.json", {
            "status": "collecting_outcome_blind_native_sources",
            "groups_complete": group + 1, "groups_total": config["source_groups"],
            "selected_positions": len(rows), "strata": counts,
            "optimizer_updates": 0})
    groups = {s: {r["group_id"] for r in rows if r["split"] == s}
              for s in ("train", "validation")}
    if groups["train"] & groups["validation"]:
        raise ValueError("Source group leakage")
    write(out / "selected-positions-private.json", rows)
    result = {"status": "native_positions_ready_for_terminal_labels",
        "groups": {k: len(v) for k, v in groups.items()}, "positions": len(rows),
        "strata": counts, "optimizer_updates": 0,
        "raw_sources_public": False, "model_inputs_are_public": True}
    write(out / "collection-result-private.json", result)
    write(out / "collection-status-private.json", result)
    return result

