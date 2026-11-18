"""Admit complete terminal menus and mix a bounded replay guard set."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .collect import digest, write
from .context import (annotate_record, target_from_utilities, terminal_training_target,
                      placement_objective, placements, terminal_utility)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def loss_recipe(counts, new_mass=.5, balanced=False):
    if not math.isfinite(new_mass) or not 0 < new_mass < 1:
        raise ValueError("New terminal loss mass must be strictly between zero and one")
    mass = {"replay_expert": (1 - new_mass) * .75,
            "replay_masked": (1 - new_mass) * .25}
    if balanced:
        mass.update({"endgame_terminal_" + bucket: new_mass / 4
                     for bucket in ("win", "third", "fourth", "hold")})
    else:
        mass["endgame_terminal"] = new_mass
    if set(counts) != set(mass) or any(n <= 0 for n in counts.values()):
        raise ValueError("Missing or unexpected training loss source")
    total = sum(counts.values())
    weights = {key: mass[key] * total / counts[key] for key in mass}
    if max(weights.values()) > 8:
        raise ValueError("Pilot mixture would over-amplify a small source")
    return mass, weights


def prepare(config):
    if (config.get("training_objective") != "top2_primary_v1"
            or config.get("terminal_utility") != [1.1, 1.0, 0.0, -0.05]):
        raise ValueError("The registered top-two objective and utility must be explicit")
    out = Path(config["out"])
    positions = json.loads((out / "selected-positions-private.json").read_text())
    splits = {"train": [], "validation": []}
    groups = {"train": set(), "validation": set()}
    new_counts = Counter()
    admitted_counts = Counter()
    neutral_counts = Counter()
    for row in positions:
        row_id = digest([row["seed"], row["step"], row["seat"], row["request_sha256"]])
        file = out / "labels" / (row_id + ".json")
        label = json.loads(file.read_text())
        if (label["status"] != "complete_terminal_menu"
                or label["source_request_sha256"] != row["request_sha256"]
                or not label["continued_until_native_match_done"]
                or label["worlds_per_action"] != config["future_worlds"]
                or len(label["outcomes"]) != label["actual_actions"]):
            raise ValueError("Missing or incompatible complete terminal label")
        record = label["record"]
        keys = list(record["questions"]["action"]["criteria"])
        if len(keys) != label["actual_actions"]:
            raise ValueError("Legal menu changed after terminal labeling")
        means = []
        for outcomes in label["outcomes"]:
            if len(outcomes) != config["future_worlds"]:
                raise ValueError("Incomplete future worlds")
            if not all(x["ledger_passed"] for x in outcomes):
                raise ValueError("Unaudited terminal settlement")
            for outcome in outcomes:
                if (placements(outcome["scores"]) != outcome["ranks"]
                        or terminal_utility(outcome["ranks"][row["seat"]]) != outcome["utility"]):
                    raise ValueError("Source terminal rank or original diagnostic utility differs")
            means.append(sum(x["utility"] for x in outcomes) / len(outcomes))
        target = target_from_utilities(means, config["target_temperature"])
        actual = record["questions"]["action"]["target"]
        if any(abs(actual[key] - p) > 1e-12 for key, p in zip(keys, target)):
            raise ValueError("Stored terminal target formula differs")
        if record["_meta"]["expected_terminal_utilities"] != dict(zip(keys, means)):
            raise ValueError("Validation utility annotations differ from raw terminal outcomes")
        # Preserve the original source diagnostic and rederive the new objective
        # exclusively from complete terminal ranks, before any optimizer update.
        record["_meta"]["source_diagnostic_utilities"] = dict(zip(keys, means))
        means = [sum(placement_objective(x["ranks"][row["seat"]]) for x in outcomes) / len(outcomes)
                 for outcomes in label["outcomes"]]
        record["_meta"]["expected_terminal_utilities"] = dict(zip(keys, means))
        record["_meta"]["terminal_objective"] = "top2_primary_v1"
        # Targets and future diagnostics are metadata; they never enter the state.
        probabilities = target_from_utilities(means, config["target_temperature"])
        record["questions"]["action"]["target"] = dict(zip(keys, probabilities))
        record["questions"]["action"]["label"] = keys[max(range(len(keys)), key=lambda i: probabilities[i])]
        record = annotate_record(record)
        split = row["split"]
        new_counts[split + ":" + row["bucket"]] += 1
        admitted_target, basis = terminal_training_target(record["questions"]["action"]["criteria"],
                                                         means, config["target_temperature"])
        record["_meta"]["target_basis"] = basis
        if admitted_target is None:
            neutral_counts[split + ":neutral_nonwin"] += 1
            if split == "train":
                continue
        else:
            if "tie_preserve" in basis:
                neutral_counts[split + ":neutral_win_keep_baseline"] += 1
            record["questions"]["action"]["target"] = admitted_target
            record["questions"]["action"]["label"] = max(admitted_target, key=admitted_target.get)
        groups[split].add(record["_meta"]["group_id"])
        splits[split].append(record)
        admitted_counts[split + ":" + row["bucket"]] += 1
    if groups["train"] & groups["validation"]:
        raise ValueError("New source groups cross training and validation")
    if any(new_counts["train:" + bucket] < minimum for bucket, minimum in
           (("third", 24), ("fourth", 24), ("hold", 24), ("win", 8))):
        raise ValueError("Pilot lacks required pursuit, guard, or win/pass coverage")
    if any(new_counts["validation:" + bucket] < minimum for bucket, minimum in
           (("third", 8), ("fourth", 8), ("hold", 8), ("win", 4))):
        raise ValueError("Validation lacks registered endgame coverage")
    replay_sources = {}
    replay_admission = {}
    for split, key in (("train", "replay_train"), ("validation", "replay_validation")):
        source = Path(config[key])
        old = read_jsonl(source)
        replay_sources[str(source)] = checksum(source)
        pools = {}
        ineligible = Counter()
        for record in old:
            try:
                normalized = annotate_record(record)
            except (ValueError, KeyError, TypeError) as error:
                ineligible[type(error).__name__] += 1
                continue
            phase = record["_meta"].get("phase")
            label_type = record["_meta"].get("label_type")
            pools.setdefault((phase, label_type), []).append(normalized)
        replay_admission[split] = {"source_records": len(old),
            "eligible_records": sum(len(v) for v in pools.values()),
            "ineligible_for_complete_endgame_context": dict(ineligible),
            "eligible_strata": {str(k): len(v) for k, v in pools.items()}}
        phases = ("closed_discard", "open_discard", "call_window", "kan_window")
        quota = {"expert_replay": 96, "paired_future_masked": 32} if split == "train" else {
            "expert_replay": 16}
        for phase in phases:
            for kind, count in quota.items():
                pool = pools.get((phase, kind), [])
                if len(pool) < count:
                    raise ValueError("Insufficient eligible replay coverage: " + str((split, phase, kind, len(pool))))
                chosen = sorted(pool, key=lambda r: digest(["endgame-replay-v1", r["_meta"]["id"]]))[:count]
                for record in chosen:
                    record["questions"]["action"]["src"] = "replay_masked" if kind == "paired_future_masked" else "replay_expert"
                    record["_meta"]["source"] = "replay_guard"
                    record["_meta"]["bucket"] = "replay:" + phase
                    groups[split].add(record["_meta"]["group_id"])
                splits[split].extend(chosen)
    if groups["train"] & groups["validation"]:
        raise ValueError("Combined replay/new groups overlap")
    train_ids = {r["_meta"]["id"] for r in splits["train"]}
    val_ids = {r["_meta"]["id"] for r in splits["validation"]}
    if len(train_ids) != len(splits["train"]) or len(val_ids) != len(splits["validation"]) or train_ids & val_ids:
        raise ValueError("Repeated record identity")
    if config.get("balance_endgame_buckets", False):
        for record in splits["train"]:
            q = record["questions"]["action"]
            if q["src"] == "endgame_terminal":
                q["src"] += "_" + record["_meta"]["bucket"]
    for split, rows in splits.items():
        rows.sort(key=lambda r: digest(["endgame-train-order-v1", r["_meta"]["id"]]))
        jsonl(out / (split + ".jsonl"), rows)
    counts = Counter(r["questions"]["action"]["src"] for r in splits["train"])
    total = len(splits["train"])
    mass, weights = loss_recipe(counts, config.get("new_terminal_loss_mass", .5),
                               config.get("balance_endgame_buckets", False))
    max_steps = min(math.ceil(total / 16), config.get("max_optimizer_steps", math.ceil(total / 16)))
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 3:
        raise ValueError("Short fit budget must be at least three optimizer updates")
    result = {"status": "complete_terminal_dataset_ready", "new_strata": dict(new_counts),
        "objective": "top2_primary_v1", "terminal_utility": [1.1, 1.0, 0.0, -0.05],
        "admitted_new_strata": dict(admitted_counts), "neutral_menu_handling": dict(neutral_counts),
        "records": {s: len(r) for s, r in splits.items()},
        "group_overlap": 0, "loss_source_counts": dict(counts), "loss_source_mass": mass,
        "loss_source_weights": weights, "max_steps": max_steps,
        "replay_inputs": replay_sources, "replay_admission": replay_admission,
        "validation_replay": "16 expert examples per phase; no fictitious call-window MC validation",
        "files": {s: {"path": str(out / (s + ".jsonl")),
                     "sha256": checksum(out / (s + ".jsonl"))} for s in splits},
        "code_sha256": {str(p): checksum(p) for p in Path(__file__).parent.glob("*.py")},
        "optimizer_updates": 0, "training_data_public": False}
    write(out / "preparation-private.json", result)
    return result

