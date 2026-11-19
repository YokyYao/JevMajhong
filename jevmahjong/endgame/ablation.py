"""Reuse completed development pairs; add only frozen weights with the new path."""
import json
from collections import defaultdict
from pathlib import Path
import sys

from riichienv import GameRule, RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.training.futures import replay_prefix
from .collect import native_match, write
from .evaluate import interval, rollout
from .futures import bundle_identity
from .policy import ModelAgent


def path_ablation(config):
    out = Path(config["out"])
    development = json.loads((out / "development-result-private.json").read_text())
    if development["status"] != "paired_endgame_development_complete":
        raise ValueError("Complete development pairs are required")
    baseline_identity = bundle_identity(config["checkpoint"])
    if development["baseline_identity"] != baseline_identity:
        raise ValueError("The baseline differs from completed development")
    rows = [json.loads(path.read_text()) for path in sorted((out / "development").glob("*.json"))]
    if len(rows) != development["paired_positions"]:
        raise ValueError("Development pair count changed")
    sys.path.insert(0, config["upstream_source"])
    from jevmahjong.policy.routing import load_model
    frozen_new_path = ModelAgent(load_model(config["checkpoint"]), placement_aware=True)
    directory = out / "path-ablation"
    directory.mkdir(exist_ok=True)
    sources, combined = {}, []
    for row in rows:
        if (row["baseline_identity"] != baseline_identity
                or row["candidate_identity"] != development["candidate_identity"]):
            raise ValueError("Completed pair checkpoint identity changed")
        seed, step, seat = row["seed"], row["step"], row["seat"]
        file = directory / ("%s-%s-%s.json" % (seed, step, seat))
        if file.exists():
            result = json.loads(file.read_text())
            if (result["baseline_identity"] != baseline_identity
                    or result["candidate_identity"] != development["candidate_identity"]
                    or result["baseline"] != row["baseline"] or result["candidate"] != row["candidate"]):
                raise ValueError("Cached ablation no longer matches the original pair")
        else:
            if seed not in sources:
                sources[seed] = native_match(seed)
            source = sources[seed]
            if not any(p["step"] == step and p["seat"] == seat for p in source["selected"]):
                raise ValueError("The source window cannot be reproduced")
            env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_mjsoul(),
                            seed=engine_seed(seed, "spaced-v1"))
            replay_prefix(env, source["mjai_log"], step)
            if list(env.scores()) != row["baseline"]["start_scores"] or list(env.scores()) != row["candidate"]["start_scores"]:
                raise ValueError("Replayed initial scores differ")
            result = dict(row, frozen_new_path=rollout(env, seat, seed, frozen_new_path,
                                                       config["terminal_step_cap"]))
            write(file, result)
        combined.append(result)
        write(out / "path-ablation-status-private.json", {"status": "adding_only_missing_frozen_new_path_arm",
            "positions_complete": len(combined), "positions_total": len(rows),
            "original_paired_rollouts_reused": len(rows) * 2})
    differences = {}
    for first, second in (("frozen_new_path", "baseline"), ("candidate", "frozen_new_path")):
        by_group = defaultdict(list)
        for row in combined:
            by_group[row["seed"]].append(row[first]["utility"] - row[second]["utility"])
        values = [sum(v) / len(v) for v in by_group.values()]
        differences[first + "_minus_" + second] = {
            "source_group_mean_terminal_utility_delta": sum(values) / len(values),
            "source_group_bootstrap_ci95": interval(values)}
    rates = {}
    for rank, name, cutoff in ((3, "third_to_top2", 2), (4, "fourth_escape", 3), (0, "hold_top2", 2)):
        subset = [r for r in combined if r["initial_rank"] <= 2] if rank == 0 else [r for r in combined if r["initial_rank"] == rank]
        rates[name] = {"positions": len(subset), **{policy: sum(r[policy]["rank"] <= cutoff for r in subset) / len(subset)
                                                   for policy in ("baseline", "frozen_new_path", "candidate")}}
    result = {"status": "frozen_new_path_ablation_complete", "paired_positions": len(combined),
        "source_groups": len({r["seed"] for r in combined}), "existing_terminal_rollouts_reused": len(combined) * 2,
        "additional_terminal_rollouts": len(combined), "utility_components": differences, "placement_rates": rates,
        "declined_win_counts": {p: sum(r[p]["declined_wins"] for r in combined)
                                for p in ("baseline", "frozen_new_path", "candidate")},
        "same_source_windows_as_development_not_independent_confirmation": True,
        "limits": ["The new path changes input, win/pass access and ordinary action guards together.",
                   "This isolates trained weights from that combined path, not every path component separately.",
                   "Small conditional development sample with rule opponents; no deployment qualification."]}
    write(out / "path-ablation-result-private.json", result)
    write(out / "path-ablation-status-private.json", result)
    return result


def guard_ablation(config, input_only=False):
    """Diagnostic only: keep candidate weights fixed and vary two path controls."""
    out = Path(config["out"])
    development = json.loads((out / "development-result-private.json").read_text())
    fit = json.loads((out / "fit-result-private.json").read_text())
    checkpoint = (out / "training-run" / fit["selection"]["selected_checkpoint"]).resolve()
    if not checkpoint.is_relative_to((out / "training-run/checkpoints").resolve()):
        raise ValueError("Candidate path is outside the registered run")
    identity = bundle_identity(checkpoint)
    if identity != development["candidate_identity"]:
        raise ValueError("Candidate weights differ from the completed development")
    rows = [json.loads(p.read_text()) for p in sorted((out / "path-ablation").glob("*.json"))]
    if len(rows) != development["paired_positions"]:
        raise ValueError("The prior ablation must be complete")
    sys.path.insert(0, config["upstream_source"])
    from jevmahjong.policy.routing import load_model
    model = load_model(checkpoint)
    policies = {"candidate_ordinary_guard": ModelAgent(model, True, ordinary_guard=True),
                "candidate_guard_and_leading_win": ModelAgent(model, True, ordinary_guard=True,
                                                                protect_leading_wins=True)}
    if input_only:
        policies = {"frozen_new_input_original_actions": ModelAgent(load_model(config["checkpoint"]), True,
                         ordinary_guard=True, force_all_wins=True),
                    "candidate_new_input_original_actions": ModelAgent(model, True,
                         ordinary_guard=True, force_all_wins=True)}
    prefix = "input-ablation" if input_only else "guard-ablation"
    directory = out / prefix
    directory.mkdir(exist_ok=True)
    sources, combined = {}, []
    for row in rows:
        seed, step, seat = row["seed"], row["step"], row["seat"]
        file = directory / ("%s-%s-%s.json" % (seed, step, seat))
        if file.exists():
            result = json.loads(file.read_text())
            if any(result[k] != v for k, v in row.items()):
                raise ValueError("Cached controls differ from the original ablation")
        else:
            if seed not in sources:
                sources[seed] = native_match(seed)
            env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_mjsoul(),
                            seed=engine_seed(seed, "spaced-v1"))
            replay_prefix(env, sources[seed]["mjai_log"], step)
            if list(env.scores()) != row["baseline"]["start_scores"]:
                raise ValueError("Guard ablation initial state differs")
            result = dict(row)
            for name, policy in policies.items():
                result[name] = rollout(env.clone(), seat, seed, policy, config["terminal_step_cap"])
            write(file, result)
        combined.append(result)
        write(out / (prefix + "-status-private.json"), {"status": "diagnosing_path_controls_only",
            "positions_complete": len(combined), "positions_total": len(rows)})
    summaries = {}
    for policy in policies:
        by_group = defaultdict(list)
        for row in combined:
            by_group[row["seed"]].append(row[policy]["utility"] - row["baseline"]["utility"])
        values = [sum(v) / len(v) for v in by_group.values()]
        rates = {}
        for rank, name, cutoff in ((3, "third_to_top2", 2), (4, "fourth_escape", 3), (0, "hold_top2", 2)):
            subset = [r for r in combined if r["initial_rank"] <= 2] if rank == 0 else [r for r in combined if r["initial_rank"] == rank]
            rates[name] = {"positions": len(subset),
                "baseline": sum(r["baseline"]["rank"] <= cutoff for r in subset) / len(subset),
                "candidate": sum(r[policy]["rank"] <= cutoff for r in subset) / len(subset)}
        summaries[policy] = {"source_group_mean_terminal_utility_delta": sum(values) / len(values),
            "source_group_bootstrap_ci95": interval(values), "placement_rates": rates,
            "declined_wins": sum(r[policy]["declined_wins"] for r in combined)}
    result = {"status": "input_path_diagnostics_complete" if input_only else "guard_path_diagnostics_complete", "paired_positions": len(rows),
        "source_groups": len({r["seed"] for r in rows}), "existing_terminal_rollouts_reused": len(rows) * 3,
        "additional_terminal_rollouts": len(rows) * len(policies), "candidate_identity": identity,
        "policies": summaries, "not_independent_confirmation": True,
        "deployed": False, "limits": ["Diagnostics on already-used development windows, not release qualification.",
        "Any revised path must be fixed before new sources are evaluated."]}
    write(out / (prefix + "-result-private.json"), result)
    write(out / (prefix + "-status-private.json"), result)
    return result
