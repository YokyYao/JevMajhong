"""Outcome-blind, paired source coverage audit; no model training or labels."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path

from riichienv import ActionType, GameRule, RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import (build_request, describe, features, heuristic_index,
                                 legal_candidates, signature)
from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.evaluation.agents import CallingOpponent, EfficiencyAgent
from .collect import digest, write
from .context import placements, serving_request
from .native_public import add_native_context
from .public_policy import win_pass_allowed
from .win_value import settlement_request

PROFILES = ("standard", "dama")
SCHEMA = "endgame-source-coverage-v1"


class DamaAgent(EfficiencyAgent):
    """Source diversity only: use the usual discard rule without declaring riichi."""
    def act(self, obs):
        actions = legal_candidates(obs)
        if not any(a.action_type == ActionType.RIICHI for a in actions):
            return super().act(obs)
        offered = [a for a in actions if a.action_type != ActionType.RIICHI]
        if not offered:
            raise ValueError("Cannot remove the only legal source action")
        _, visible, data = describe(obs)
        feats = features(obs, offered, visible, data)
        return offered[heuristic_index(obs, offered, feats)]


def public_stratum(payload):
    """No future outcome, source terminal score, or other concealed hand accepted."""
    if not win_pass_allowed(payload):
        return None
    state = payload["state"]
    seat = state["seat"]
    if state["melds"][seat] is None:
        return None
    closed = all(m["type"] == 3 for m in state["melds"][seat])
    style = ("closed_riichi" if seat in state["riichi"] else "closed_nonriichi") if closed else "open"
    second = sorted(range(4), key=lambda p: (-state["scores"][p], p))[1]
    gap = state["scores"][second] - state["scores"][seat]
    band = "le3000" if gap <= 3000 else "3001_8000" if gap <= 8000 else "gt8000"
    quote = state["endgame_context"]["current_win_settlement"]
    maximum = quote["maximum_baseline_scenario"]["deltas"][seat]
    rank = state["endgame_context"]["placement"]
    gain_band = "gain_le3900" if maximum <= 3900 else "gain_gt3900"
    return {"cell": f"{style}:{band}:rank{rank}:{gain_band}", "style": style,
            "gap_to_second": gap, "gap_band": band,
            "maximum_public_win_gain": maximum, "placement": rank,
            "target": style == "closed_nonriichi" and gap <= 8000 and maximum <= 3900}


def select_cells(rows, per_cell):
    """One hash-selected public position per cell by default; no reward selection."""
    cells, selected = Counter(), []
    seen = set()
    for row in sorted(rows, key=lambda r: r["selection_key"]):
        key = row["coverage_stratum"]["cell"]
        if cells[key] < per_cell and row["request_sha256"] not in seen:
            selected.append(row)
            cells[key] += 1
            seen.add(row["request_sha256"])
    return selected


def coverage_match(seed, profile, per_cell=1):
    if profile not in PROFILES or per_cell != 1:
        raise ValueError("Unregistered coverage source profile or per-cell quota")
    env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_mjsoul(),
                    seed=engine_seed(seed, "spaced-v1"))
    observations = env.reset()
    ordinary = DamaAgent() if profile == "dama" else EfficiencyAgent()
    caller = CallingOpponent()
    opportunities, action_counts, rows = Counter(), Counter(), []
    boundaries, previous = [], None
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
                if len(actions) < 2 or not any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in actions):
                    continue
                request, actions, _ = build_request(obs)
                request["state"]["hand_number"] = int(obs.kyoku_index) + 1
                if request["state"]["hand_number"] != env.kyoku_idx + 1:
                    raise ValueError("Observation/native hand number mismatch")
                legacy = serving_request(request)
                if placements(legacy["state"]["scores"]) != list(env.ranks()):
                    raise ValueError("Public/native placement mismatch")
                public = settlement_request(add_native_context(request, obs, actions))
                stratum = public_stratum(public)
                if stratum is None:
                    continue
                opportunities[stratum["cell"]] += 1
                row = {"seed": seed, "step": step, "seat": seat, "bucket": "win",
                    "request": legacy, "request_sha256": digest(legacy),
                    "coverage_stratum": stratum,
                    "public_request": public, "public_request_sha256": digest(public)}
                row["selection_key"] = digest([seed, profile, row["request_sha256"], SCHEMA])
                rows.append(row)
        chosen = {}
        for seat, obs in observations.items():
            is_caller = seat == seed % 4
            agent = caller if is_caller else ordinary
            action = agent.act(obs)
            legal = legal_candidates(obs)
            if signature(action) not in {signature(a) for a in legal}:
                raise ValueError("Illegal source action")
            name = json.loads(action.to_mjai())["type"]
            action_counts[("caller:" if is_caller else "ordinary:") + name] += 1
            if profile == "dama" and not is_caller and action.action_type == ActionType.RIICHI:
                raise ValueError("Dama source declared riichi")
            chosen[seat] = action
        observations = env.step(chosen)
    if not env.done():
        raise RuntimeError("Coverage source exceeded bounded match steps")
    ledger = settlement_ledger(env.mjai_log, env.scores())
    return {"schema": SCHEMA, "seed": seed, "profile": profile,
        "per_cell": per_cell, "source_steps": step,
        "source_policy": "three-" + profile + "-plus-one-calling-rule-opponent",
        "selected_before_future_labels": True, "selected": select_cells(rows, per_cell),
        "opportunities": dict(opportunities), "source_actions": dict(action_counts),
        "hand_boundaries": boundaries, "final_scores": list(env.scores()),
        "final_ranks": list(env.ranks()), "ledger": ledger, "mjai_log": env.mjai_log}


def _collect_group(out, seed, split):
    results = {}
    for profile in PROFILES:
        path = Path(out) / profile / "sources" / f"{seed}.json"
        if path.exists():
            source = json.loads(path.read_text())
            if (source.get("schema") != SCHEMA or source.get("seed") != seed
                    or source.get("profile") != profile or source.get("per_cell") != 1):
                raise ValueError("Existing coverage source mismatch")
        else:
            source = coverage_match(seed, profile)
            write(path, source)
        rows = [dict(r, split=split, group_id="endgame-native:" + str(seed)) for r in source["selected"]]
        results[profile] = {"rows": rows, "opportunities": source["opportunities"],
                            "source_actions": source["source_actions"]}
    return seed, results


def collect_coverage(config):
    groups = config["source_groups"]
    if config.get("schema") != SCHEMA or groups not in (64, 256):
        raise ValueError("Coverage pilot must use registered 64 or 256 paired groups")
    workers = config.get("workers", 2)
    if not 1 <= workers <= min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", 1))):
        raise ValueError("Workers exceed registered or allocated CPU resources")
    out = Path(config["out"])
    complete = {}
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    tasks = []
    try:
        for index in range(groups):
            split = "validation" if index % 4 == 3 else "train"
            tasks.append(executor.submit(_collect_group, str(out), config["seed_start"] + index, split))
        for future in as_completed(tasks):
            seed, result = future.result()
            complete[seed] = result
            status = _coverage_summary(complete, config)
            status["status"] = "collecting_public_source_coverage"
            write(out / "coverage-status-private.json", status)
            print(json.dumps({"groups_complete": len(complete), "groups_total": groups,
                              "target_positions": status["target_positions"]}), flush=True)
    finally:
        for task in tasks:
            task.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    for profile in PROFILES:
        rows = [r for seed in sorted(complete) for r in complete[seed][profile]["rows"]]
        train = {r["group_id"] for r in rows if r["split"] == "train"}
        validation = {r["group_id"] for r in rows if r["split"] == "validation"}
        if train & validation:
            raise ValueError("Coverage source split leakage")
        write(out / profile / "selected-positions-private.json", rows)
    result = _coverage_summary(complete, config)
    result["status"] = "coverage_complete_no_labels_or_updates"
    write(out / "coverage-result-private.json", result)
    write(out / "coverage-status-private.json", result)
    return result


def _coverage_summary(complete, config):
    counts, targets, target_groups, actions = {}, {}, {}, {}
    for profile in PROFILES:
        count, target, groups, act = Counter(), Counter(), {s: set() for s in ("train", "validation")}, Counter()
        for seed, result in complete.items():
            act.update(result[profile]["source_actions"])
            for row in result[profile]["rows"]:
                split = row["split"]
                count[split + ":" + row["coverage_stratum"]["cell"]] += 1
                if row["coverage_stratum"]["target"]:
                    target[split] += 1
                    groups[split].add(seed)
        counts[profile], targets[profile] = dict(count), dict(target)
        target_groups[profile] = {s: len(g) for s, g in groups.items()}
        actions[profile] = dict(act)
    enough = (targets["dama"].get("train", 0) >= 24 and targets["dama"].get("validation", 0) >= 8
              and target_groups["dama"]["train"] >= 12 and target_groups["dama"]["validation"] >= 6)
    return {"groups_complete": len(complete), "groups_total": config["source_groups"],
        "seed_start": config["seed_start"], "profiles": list(PROFILES), "strata": counts,
        "target_positions": targets, "target_source_groups": target_groups,
        "source_actions": actions, "coverage_threshold_met": enough,
        "target_definition": "closed_nonriichi, public win insufficient for top2, nondealer rank3/4, gap_to_second<=8000, max known-baseline own gain<=3900",
        "source_policy_is_not_a_teacher": True, "raw_sources_public": False,
        "optimizer_updates": 0, "future_labels_generated": 0}
