"""Admit complete terminal menus and mix a bounded replay guard set."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .collect import digest, write
from .context import annotate_record, target_from_utilities, terminal_training_target


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


def prepare(config):
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
            means.append(sum(x["utility"] for x in outcomes) / len(outcomes))
        target = target_from_utilities(means, config["target_temperature"])
        actual = record["questions"]["action"]["target"]
        if any(abs(actual[key] - p) > 1e-12 for key, p in zip(keys, target)):
            raise ValueError("Stored terminal target formula differs")
        if record["_meta"]["expected_terminal_utilities"] != dict(zip(keys, means)):
            raise ValueError("Validation utility annotations differ from raw terminal outcomes")
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
    for split, key in (("train", "replay_train"), ("validation", "replay_validation")):
        source = Path(config[key])
        old = read_jsonl(source)
        replay_sources[str(source)] = checksum(source)
        pools = {}
        for record in old:
            try:
                normalized = annotate_record(record)
            except (ValueError, KeyError, TypeError):
                continue
            phase = record["_meta"].get("phase")
            label_type = record["_meta"].get("label_type")
            pools.setdefault((phase, label_type), []).append(normalized)
        phases = ("closed_discard", "open_discard", "call_window", "kan_window")
        quota = {"expert_replay": 96, "paired_future_masked": 32} if split == "train" else {
            "expert_replay": 8, "paired_future_masked": 8}
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
    for split, rows in splits.items():
        rows.sort(key=lambda r: digest(["endgame-train-order-v1", r["_meta"]["id"]]))
        jsonl(out / (split + ".jsonl"), rows)
    counts = Counter(r["questions"]["action"]["src"] for r in splits["train"])
    total = len(splits["train"])
    mass = {"endgame_terminal": .5, "replay_expert": .375, "replay_masked": .125}
    if set(counts) != set(mass):
        raise ValueError("Missing training loss source")
    weights = {key: mass[key] * total / counts[key] for key in mass}
    if max(weights.values()) > 8:
        raise ValueError("Pilot mixture would over-amplify a small source")
    result = {"status": "complete_terminal_dataset_ready", "new_strata": dict(new_counts),
        "admitted_new_strata": dict(admitted_counts), "neutral_menu_handling": dict(neutral_counts),
        "records": {s: len(r) for s, r in splits.items()},
        "group_overlap": 0, "loss_source_counts": dict(counts), "loss_source_mass": mass,
        "loss_source_weights": weights, "max_steps": math.ceil(total / 16),
        "replay_inputs": replay_sources,
        "files": {s: {"path": str(out / (s + ".jsonl")),
                     "sha256": checksum(out / (s + ".jsonl"))} for s in splits},
        "code_sha256": {str(p): checksum(p) for p in Path(__file__).parent.glob("*.py")},
        "optimizer_updates": 0, "training_data_public": False}
    write(out / "preparation-private.json", result)
    return result
