"""Complete terminal-match labels for all actions in each selected late window."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from riichienv import GameRule, RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request, legal_candidates, signature
from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.evaluation.agents import EfficiencyAgent, CallingOpponent
from jevmahjong.training.futures import replay_prefix, sample_future
from .collect import digest, write
from .context import placements, serving_request, terminal_utility, target_from_utilities
from .ledger import hand_score_history
from .policy import ModelAgent


def bundle_identity(checkpoint):
    names = ("adapter_model.safetensors", "adapter_config.json", "head.pt",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    hashes = {name: hashlib.sha256((Path(checkpoint) / name).read_bytes()).hexdigest()
              for name in names}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    return {"files": hashes, "identity_sha256": identity}


def finish_match(env, initial_actions, seat, seed, frozen, step_cap):
    ordinary, caller = EfficiencyAgent(), CallingOpponent()
    observations = env.step(initial_actions)
    for step in range(step_cap):
        if env.done():
            break
        selected = {}
        for p, obs in observations.items():
            selected[p] = frozen.act(obs) if p == seat else (
                caller if p == seed % 4 else ordinary).act(obs)
        observations = env.step(selected)
    if not env.done():
        raise RuntimeError("Terminal rollout truncated; reject this label")
    ledger = settlement_ledger(env.mjai_log, env.scores())
    scores, ranks = list(env.scores()), list(env.ranks())
    if placements(scores) != ranks:
        raise ValueError("Terminal placement differs from score tie-break")
    history = hand_score_history(env.mjai_log)
    if history["final_scores"] != scores:
        raise ValueError("Per-hand history does not reach terminal scores")
    return {"scores": scores, "ranks": ranks, "utility": terminal_utility(ranks[seat]),
            "ledger_passed": True, "steps": step + 1,
            "final_hand_marker": [env.round_wind, env.kyoku_idx, env.honba],
            "hand_history": history}


def label(config, limit=None):
    out = Path(config["out"])
    selected = json.loads((out / "selected-positions-private.json").read_text())
    if limit is not None:
        selected = selected[:limit]
    sys.path.insert(0, config["upstream_source"])
    from jevmahjong.policy.routing import load_model
    identity = bundle_identity(config["checkpoint"])
    from desktop.jev_sidecar.unified_policy import identity as released_identity
    released = released_identity()
    if identity["identity_sha256"] not in (released["source_identity_sha256"], released["identity_sha256"]):
        raise ValueError("The baseline must be the exact published Play v2 source checkpoint")
    write(out / "baseline-identity-private.json", identity)
    write(out / "label-status-private.json", {"status": "loading_frozen_play_v2",
          "positions_total": len(selected), "optimizer_updates": 0})
    model = load_model(config["checkpoint"])
    frozen = ModelAgent(model, placement_aware=False)
    ordinary, caller = EfficiencyAgent(), CallingOpponent()
    source_cache = {}
    start = time.monotonic()
    done = 0
    for row in selected:
        row_id = digest([row["seed"], row["step"], row["seat"], row["request_sha256"]])
        file = out / "labels" / (row_id + ".json")
        if file.exists():
            result = json.loads(file.read_text())
            if (result.get("status") == "complete_terminal_menu"
                    and result["source_request_sha256"] == row["request_sha256"]
                    and result["baseline_identity_sha256"] == identity["identity_sha256"]
                    and result["worlds_per_action"] == config["future_worlds"]):
                done += 1
                continue
            raise ValueError("Partial or mismatched label file; inspect before resuming")
        source = source_cache.get(row["seed"])
        if source is None:
            source_path = out / "sources" / (str(row["seed"]) + ".json")
            source = json.loads(source_path.read_text())
            history = hand_score_history(source["mjai_log"])
            if history["final_scores"] != source["final_scores"]:
                raise ValueError("Source per-hand scores do not reconcile")
            if "hand_score_history" not in source:
                source["hand_score_history"] = history
                write(source_path, source)
            source_cache[row["seed"]] = source
        env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_mjsoul(),
                       seed=engine_seed(row["seed"], "spaced-v1"))
        observations = replay_prefix(env, source["mjai_log"], row["step"])
        obs = observations[row["seat"]]
        request, actions, _ = build_request(obs)
        request["state"]["hand_number"] = int(obs.kyoku_index) + 1
        payload = serving_request(request)
        if digest(payload) != row["request_sha256"] or payload != row["request"]:
            raise ValueError("Replayed public request differs from collection")
        action_signatures = [signature(a) for a in actions]
        outcomes = [[] for _ in actions]
        for world in range(config["future_worlds"]):
            world_seed = int(digest([row_id, world, "terminal-wall-v1"])[:16], 16)
            sampled = sample_future(env, world_seed, "unseen")
            initial = {p: obs for p, obs in sampled.clone().get_observations().items()
                       if obs.legal_actions()}
            other_actions = {p: (caller if p == row["seed"] % 4 else ordinary).act(o)
                             for p, o in initial.items() if p != row["seat"]}
            for index, action_signature in enumerate(action_signatures):
                branch = sampled.clone()
                branch_obs = {p: obs for p, obs in branch.get_observations().items()
                              if obs.legal_actions()}
                matching = {signature(a): a for a in legal_candidates(branch_obs[row["seat"]])}
                if action_signature not in matching:
                    raise ValueError("Sampled wall changed the current legal action menu")
                initial_actions = {**other_actions, row["seat"]: matching[action_signature]}
                result = finish_match(branch, initial_actions, row["seat"], row["seed"],
                                      frozen, config["terminal_step_cap"])
                outcomes[index].append(result)
                write(out / "label-status-private.json", {
                    "status": "generating_full_terminal_labels", "positions_complete": done,
                    "positions_total": len(selected), "current_bucket": row["bucket"],
                    "current_split": row["split"], "current_world": world + 1,
                    "worlds_per_action": config["future_worlds"],
                    "current_action": index + 1, "current_actions_total": len(actions),
                    "seconds": time.monotonic() - start, "optimizer_updates": 0})
        means = [sum(x["utility"] for x in values) / len(values) for values in outcomes]
        probability = target_from_utilities(means, config["target_temperature"])
        keys = list(payload["questions"]["action"]["criteria"])
        record = json.loads(json.dumps(payload))
        record["questions"]["action"].update(
            target=dict(zip(keys, probability)), label=keys[max(range(len(keys)), key=lambda i: probability[i])],
            src="endgame_terminal")
        record["_meta"] = {"id": "endgame:" + row_id, "group_id": row["group_id"],
            "source": "endgame_terminal", "variant": "clean", "split": row["split"],
            "bucket": row["bucket"], "expected_terminal_utilities": dict(zip(keys, means))}
        result = {"status": "complete_terminal_menu", "source": row,
            "source_request_sha256": row["request_sha256"],
            "baseline_identity_sha256": identity["identity_sha256"],
            "worlds_per_action": config["future_worlds"], "outcomes": outcomes,
            "record": record, "actual_actions": len(actions),
            "sampling_scope": "unseen current-hand wall; fixed current concealed hands and cloned future-hand RNG",
            "continued_until_native_match_done": True}
        write(file, result)
        done += 1
        print(json.dumps({"terminal_positions_complete": done, "total": len(selected),
                          "bucket": row["bucket"], "split": row["split"],
                          "actions": len(actions), "seconds": time.monotonic() - start}), flush=True)
    result = {"status": "all_selected_terminal_labels_complete", "positions": done,
              "worlds_per_action": config["future_worlds"], "optimizer_updates": 0}
    write(out / "label-result-private.json", result)
    write(out / "label-status-private.json", result)
    return result

