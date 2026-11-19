"""Paired fresh endgame development; this is not a human-strength benchmark."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

from riichienv import GameRule, RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import legal_candidates
from jevmahjong.evaluation.agents import EfficiencyAgent, CallingOpponent
from .collect import native_match, write, digest
from .context import placements, placement_objective
from .futures import bundle_identity
from .ledger import hand_score_history
from .policy import ModelAgent
from jevmahjong.training.futures import replay_prefix


def rollout(env, seat, seed, policy, step_cap):
    prefix_size = len(env.mjai_log)
    before = list(env.scores())
    observations = {p: o for p, o in env.get_observations().items() if o.legal_actions()}
    ordinary, caller = EfficiencyAgent(), CallingOpponent()
    declined_wins = win_opportunities = 0
    for step in range(step_cap):
        if env.done():
            break
        selected = {}
        for p, obs in observations.items():
            if p == seat:
                action = policy.act(obs)
                offered = any(json.loads(a.to_mjai())["type"] == "hora" for a in legal_candidates(obs))
                win_opportunities += int(offered)
                declined_wins += int(offered and json.loads(action.to_mjai())["type"] != "hora")
            else:
                action = (caller if p == seed % 4 else ordinary).act(obs)
            selected[p] = action
        observations = env.step(selected)
    if not env.done():
        raise RuntimeError("Truncated endgame evaluation is not a final result")
    history = hand_score_history(env.mjai_log)
    scores, ranks = list(env.scores()), list(env.ranks())
    if history["final_scores"] != scores or placements(scores) != ranks:
        raise ValueError("Evaluation settlement/placement mismatch")
    suffix = env.mjai_log[prefix_size:]
    return {"start_scores": before, "final_scores": scores, "rank": ranks[seat],
            "utility": placement_objective(ranks[seat]), "net_points_from_window": scores[seat] - before[seat],
            "win_opportunities": win_opportunities, "declined_wins": declined_wins,
            "own_wins_after_window": sum(e["type"] == "hora" and e["actor"] == seat for e in suffix),
            "dealins_after_window": sum(e["type"] == "hora" and e["target"] == seat and e["actor"] != seat for e in suffix),
            "hands_ended_after_window": sum(e["type"] == "end_kyoku" for e in suffix),
            "steps": step + 1, "terminal_ledger_passed": True}


def interval(values, seed=2026111907):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return [means[int(.025 * len(means))], means[int(.975 * len(means))]]


def point_and_risk_summary(records):
    if not records:
        raise ValueError("No paired endgame records")
    summary = {}
    for policy in ("baseline", "candidate"):
        hands = sum(r[policy]["hands_ended_after_window"] for r in records)
        dealins = sum(r[policy]["dealins_after_window"] for r in records)
        summary[policy] = {
            "mean_net_points_from_window": sum(r[policy]["net_points_from_window"] for r in records) / len(records),
            "own_win_events": sum(r[policy]["own_wins_after_window"] for r in records),
            "dealin_events": dealins, "completed_hands_after_windows": hands,
            "dealin_events_per_completed_hand": dealins / hands if hands else None,
            "windows_with_any_dealin_fraction": sum(r[policy]["dealins_after_window"] > 0 for r in records) / len(records)}
    by_group = defaultdict(list)
    for row in records:
        by_group[row["seed"]].append(row["candidate"]["net_points_from_window"] - row["baseline"]["net_points_from_window"])
    changes = [sum(v) / len(v) for v in by_group.values()]
    return {"per_policy": summary, "source_group_mean_net_point_gain": sum(changes) / len(changes),
            "source_group_bootstrap_net_point_gain_ci95": interval(changes),
            "scope": "paired late windows; completed hands may be partial at window entry; not full-match-start strength"}


def evaluate(config):
    out = Path(config["out"])
    phase = config.get("evaluation_phase", "development")
    if phase not in ("development", "confirmation"):
        raise ValueError("Unsupported evaluation phase")
    fit = json.loads((out / "fit-result-private.json").read_text())
    if not fit["eligible_for_development"]:
        result = {"status": "development_skipped_zero_step_selected",
                  "reason": "No candidate improved validation terminal utility under the replay guard",
                  "deployed": False}
        write(out / (phase + "-result-private.json"), result)
        return result
    candidate = (out / "training-run" / fit["selection"]["selected_checkpoint"]).resolve()
    if not candidate.is_relative_to((out / "training-run/checkpoints").resolve()):
        raise ValueError("Selected checkpoint is outside this registered run")
    sys.path.insert(0, config["upstream_source"])
    from jevmahjong.policy.routing import load_model
    baseline_identity = bundle_identity(config["checkpoint"])
    candidate_identity = bundle_identity(candidate)
    protocol = None
    if phase == "confirmation":
        development = json.loads((out / "development-result-private.json").read_text())
        if (not development["eligible_for_independent_confirmation"]
                or development["candidate_identity"] != candidate_identity
                or candidate_identity["identity_sha256"] != config["confirmation_candidate_identity"]):
            raise ValueError("Confirmation requires the fixed qualified development candidate")
        from .prepare import checksum
        root = Path(__file__).resolve().parents[2]
        paths = ("jevmahjong/endgame/policy.py", "jevmahjong/endgame/context.py",
                 "jevmahjong/endgame/win_value.py", "jevmahjong/endgame/native_public.py",
                 "jevmahjong/endgame/public_policy.py", "jevmahjong/endgame/collect.py",
                 "jevmahjong/endgame/evaluate.py", "jevmahjong/endgame/ledger.py",
                 "jevmahjong/training/futures.py", "jevmahjong/core/seeds.py",
                 "jevmahjong/policy/routing.py", "desktop/jev_sidecar/play_v2_identity.json",
                 "jevmahjong/core/state.py", "jevmahjong/evaluation/agents.py",
                 "desktop/jev_sidecar/unified_policy.py", "desktop/jev_sidecar/call_features.py",
                 "desktop/jev_sidecar/call_feature_helpers.py")
        protocol = {"candidate_identity": candidate_identity, "baseline_identity": baseline_identity,
                    "input_version": config.get("input_version", "endgame-public-v2"),
                    "terminal_step_cap": config["terminal_step_cap"],
                    "action_controls": {k: config.get(k, False) for k in ("preserve_ordinary_guard", "protect_leading_wins")},
                    "source_code_sha256": {p: checksum(root / p) for p in paths},
                    "groups": config["confirmation_groups"], "seed_start": config["confirmation_seed_start"]}
        protocol_file = out / "confirmation-protocol-private.json"
        if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Confirmation code, candidate or source schedule changed")
        write(protocol_file, protocol)
    baseline = ModelAgent(load_model(config["checkpoint"]), placement_aware=False)
    learner = ModelAgent(load_model(candidate), placement_aware=True,
        input_version=config.get("input_version", "endgame-public-v2"),
        ordinary_guard=config.get("preserve_ordinary_guard", False),
        protect_leading_wins=config.get("protect_leading_wins", False))
    groups = config["confirmation_groups"] if phase == "confirmation" else config.get("development_groups", 8)
    seed_start = config["confirmation_seed_start"] if phase == "confirmation" else config.get("development_seed_start", 2026112900)
    fitted_sources = json.loads((out / "selected-positions-private.json").read_text())
    fitted_seeds = {row["seed"] for row in fitted_sources}
    if fitted_seeds & set(range(seed_start, seed_start + groups)):
        raise ValueError("Development source seeds overlap training/validation")
    if phase == "confirmation" and set(range(seed_start, seed_start + groups)) & set(range(
            config.get("development_seed_start", 2026112900),
            config.get("development_seed_start", 2026112900) + config.get("development_groups", 8))):
        raise ValueError("Confirmation overlaps development")
    directory = out / phase
    directory.mkdir(exist_ok=True)
    records = []
    for group in range(groups):
        seed = seed_start + group
        source = native_match(seed)
        for position in source["selected"]:
            file = directory / ("%s-%s-%s.json" % (seed, position["step"], position["seat"]))
            if file.exists():
                row = json.loads(file.read_text())
                if (row["candidate_identity"] != candidate_identity
                        or row["baseline_identity"] != baseline_identity
                        or (protocol is not None and row.get("protocol_sha256") != digest(protocol))):
                    raise ValueError("Existing development used different checkpoints")
            else:
                env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_mjsoul(),
                               seed=engine_seed(seed, "spaced-v1"))
                replay_prefix(env, source["mjai_log"], position["step"])
                seat = position["seat"]
                initial_rank = placements(list(env.scores()))[seat]
                row = {"seed": seed, "seat": seat, "step": position["step"],
                    "bucket": position["bucket"], "initial_rank": initial_rank,
                    "baseline_identity": baseline_identity, "candidate_identity": candidate_identity,
                    "baseline": rollout(env.clone(), seat, seed, baseline, config["terminal_step_cap"]),
                    "candidate": rollout(env.clone(), seat, seed, learner, config["terminal_step_cap"])}
                if protocol is not None:
                    row["protocol_sha256"] = digest(protocol)
                write(file, row)
            records.append(row)
            write(out / (phase + "-status-private.json"), {"status": "paired_fresh_endgame_" + phase,
                "source_groups_complete": group, "source_groups_total": groups, "paired_positions": len(records)})
    by_group = defaultdict(list)
    for row in records:
        by_group[row["seed"]].append(row["candidate"]["utility"] - row["baseline"]["utility"])
    differences = [sum(v) / len(v) for v in by_group.values()]
    rates = {}
    for initial, name in ((3, "third_to_top2"), (4, "fourth_escape"), (0, "hold_top2")):
        subset = [r for r in records if r["initial_rank"] <= 2] if initial == 0 else [
            r for r in records if r["initial_rank"] == initial]
        if not subset:
            raise ValueError("Development lacks initial placement coverage")
        cutoff = 3 if initial == 4 else 2
        values = {policy: sum(r[policy]["rank"] <= cutoff for r in subset) / len(subset)
                  for policy in ("baseline", "candidate")}
        rates[name] = dict(positions=len(subset), **values,
                          candidate_minus_baseline=values["candidate"] - values["baseline"])
    gain = sum(differences) / len(differences)
    eligible = (gain > 0
        and rates["third_to_top2"]["candidate_minus_baseline"] >= 0
        and rates["fourth_escape"]["candidate_minus_baseline"] >= 0
        and rates["hold_top2"]["candidate_minus_baseline"] >= -.05)
    bounds = interval(differences)
    result = {"status": "paired_endgame_" + phase + "_complete", "independent_source_groups": len(by_group),
        "paired_positions": len(records), "terminal_continuations": len(records) * 2,
        "terminal_utility_gain": gain, "whole_source_group_paired_bootstrap_ci95": bounds,
        "placement_rates": rates, "eligible_for_independent_confirmation": eligible,
        "points_and_dealins": point_and_risk_summary(records),
        "baseline_identity": baseline_identity, "candidate_identity": candidate_identity,
        "declined_win_counts": {p: sum(r[p]["declined_wins"] for r in records) for p in ("baseline", "candidate")},
        "no_independent_confirmation_yet": phase != "confirmation", "deployed": False,
        "limits": ["Rule-policy source trajectories and opponents; sampled late windows, not full-match-start win rate.",
                   f"Requested {groups} source groups; bootstrap draws add no independent information.",
                   "The learner changes the endgame input/action path as well as LoRA weights.",
                   "Raw per-window records remain private."]}
    if phase == "confirmation":
        if any(checksum(root / p) != value for p, value in protocol["source_code_sha256"].items()):
            raise ValueError("Frozen serving code changed during confirmation")
        result["eligible_for_independent_confirmation"] = False
        result["confirmation_passed"] = (bounds[0] > 0
            and rates["third_to_top2"]["candidate_minus_baseline"] > 0
            and rates["fourth_escape"]["candidate_minus_baseline"] >= 0
            and rates["hold_top2"]["candidate_minus_baseline"] >= 0)
        result["protocol_sha256"] = digest(protocol)
    write(out / (phase + "-result-private.json"), result)
    write(out / (phase + "-status-private.json"), result)
    return result

