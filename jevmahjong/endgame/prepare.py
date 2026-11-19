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


def loss_recipe(counts, new_mass=.5, balanced=False, reference=False, source_mass=None):
    if not math.isfinite(new_mass) or not 0 < new_mass < 1:
        raise ValueError("New terminal loss mass must be strictly between zero and one")
    mass = {"replay_expert": (1 - new_mass) * .75,
            "replay_masked": (1 - new_mass) * .25}
    if reference:
        mass = {"replay_reference": 1 - new_mass}
    if balanced:
        mass.update({"endgame_terminal_" + bucket: new_mass / 4
                     for bucket in ("win", "third", "fourth", "hold")})
    else:
        mass["endgame_terminal"] = new_mass
    if source_mass is not None:
        mass = dict(source_mass)
        if (any(not math.isfinite(v) or v < 0 or v >= 1 for v in mass.values())
                or abs(sum(mass.values()) - 1) > 1e-12
                or any(v == 0 and key != "replay_fixed_win" for key, v in mass.items())):
            raise ValueError("Explicit source masses must sum to one; only runtime-fixed win controls may have zero loss")
    if set(counts) != set(mass) or any(n <= 0 for n in counts.values()):
        raise ValueError("Missing or unexpected training loss source")
    total = sum(counts.values())
    weights = {key: mass[key] * total / counts[key] for key in mass}
    if max(weights.values()) > 8:
        raise ValueError("Pilot mixture would over-amplify a small source")
    return mass, weights


def label_recipe(config, row, file):
    """Keep heterogeneous original MC counts/temperatures explicit and immutable."""
    if not config.get("label_sets"):
        return config["future_worlds"], config["target_temperature"], "legacy_single_set"
    tag = row.get("_label_set")
    if tag not in config["label_sets"]:
        raise ValueError("Terminal row has no registered label-set provenance")
    spec = config["label_sets"][tag]
    expected = Path(spec["root"]) / "labels" / Path(file).name
    if Path(file).resolve() != expected.resolve():
        raise ValueError("Terminal file is not the registered original label")
    worlds, temperature = spec["future_worlds"], spec["source_target_temperature"]
    if (type(worlds) is not int or not 1 <= worlds <= 32
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Invalid registered original label recipe")
    return worlds, temperature, tag


def balance_training_sources(records, config):
    if not config.get("balance_endgame_buckets", False):
        return
    for record in records:
        q = record["questions"]["action"]
        if q["src"] != "endgame_terminal":
            continue
        bucket = record["_meta"]["bucket"]
        q["src"] += "_" + bucket
        if (config.get("split_win_by_initial_rank", False) and config.get("reference_regularization", False)
                and bucket == "win"):
            rank = placements(record["state"]["scores"])[record["state"]["seat"]]
            if rank not in (3, 4):
                raise ValueError("Only trailing active-win positions may receive rank-balanced loss")
            q["src"] += "_third" if rank == 3 else "_fourth"


def prepare(config):
    if config.get('native_leading_context_root') and not config.get('native_context_root'):
        raise ValueError('Leading contexts cannot replace or silently omit trailing contexts')
    if config.get("native_context_root"):
        from .context import TRAILING_INPUT_VERSION
        if config.get("reference_mode") != "released_original_policy" or config.get("input_version") != TRAILING_INPUT_VERSION:
            raise ValueError("Native context retention requires the registered original-policy/scoped-input format")
        if not config.get("reference_regularization", False):
            from .native_retention import reference_preparation_only
            reference_preparation_only(config, {"replay_native": 1})
    if (config.get("training_objective") != "top2_primary_v1"
            or config.get("terminal_utility") != [1.1, 1.0, 0.0, -0.05]):
        raise ValueError("The registered top-two objective and utility must be explicit")
    out = Path(config["out"])
    positions = json.loads((out / "selected-positions-private.json").read_text())
    input_version = config.get("input_version", "endgame-public-v2")
    public_context_rows = None
    from .context import PUBLIC_SETTLEMENT_VERSIONS
    if input_version in PUBLIC_SETTLEMENT_VERSIONS:
        public_cache = json.loads((out / "public-context-private.json").read_text())
        if (public_cache.get("schema") != "current-public-win-context-v1"
                or not public_cache.get("source_prefix_only") or public_cache.get("future_results_used")):
            raise ValueError("Public-context enrichment provenance differs")
        public_context_rows = public_cache["rows"]
    splits = {"train": [], "validation": []}
    groups = {"train": set(), "validation": set()}
    new_counts = Counter()
    admitted_counts = Counter()
    neutral_counts = Counter()
    label_set_counts = Counter()
    for row in positions:
        row_id = digest([row["seed"], row["step"], row["seat"], row["request_sha256"]])
        file = out / "labels" / (row_id + ".json")
        worlds, source_temperature, label_set = label_recipe(config, row, file)
        label = json.loads(file.read_text())
        if (label["status"] != "complete_terminal_menu"
                or label["source_request_sha256"] != row["request_sha256"]
                or not label["continued_until_native_match_done"]
                or label["worlds_per_action"] != worlds
                or len(label["outcomes"]) != label["actual_actions"]):
            raise ValueError("Missing or incompatible complete terminal label")
        record = label["record"]
        keys = list(record["questions"]["action"]["criteria"])
        if len(keys) != label["actual_actions"]:
            raise ValueError("Legal menu changed after terminal labeling")
        means = []
        for outcomes in label["outcomes"]:
            if len(outcomes) != worlds:
                raise ValueError("Incomplete future worlds")
            if not all(x["ledger_passed"] for x in outcomes):
                raise ValueError("Unaudited terminal settlement")
            for outcome in outcomes:
                if (placements(outcome["scores"]) != outcome["ranks"]
                        or terminal_utility(outcome["ranks"][row["seat"]]) != outcome["utility"]):
                    raise ValueError("Source terminal rank or original diagnostic utility differs")
            means.append(sum(x["utility"] for x in outcomes) / len(outcomes))
        target = target_from_utilities(means, source_temperature)
        actual = record["questions"]["action"]["target"]
        if any(abs(actual[key] - p) > 1e-12 for key, p in zip(keys, target)):
            raise ValueError("Stored terminal target formula differs")
        if record["_meta"]["expected_terminal_utilities"] != dict(zip(keys, means)):
            raise ValueError("Validation utility annotations differ from raw terminal outcomes")
        # Preserve the original source diagnostic and rederive the new objective
        # exclusively from complete terminal ranks, before any optimizer update.
        record["_meta"]["source_diagnostic_utilities"] = dict(zip(keys, means))
        record["_meta"]["terminal_label_provenance"] = {
            "set": label_set, "worlds_per_action": worlds,
            "source_target_temperature": source_temperature, "file_sha256": checksum(file)}
        label_set_counts[row["split"] + ":" + label_set] += 1
        means = [sum(placement_objective(x["ranks"][row["seat"]]) for x in outcomes) / len(outcomes)
                 for outcomes in label["outcomes"]]
        record["_meta"]["expected_terminal_utilities"] = dict(zip(keys, means))
        record["_meta"]["terminal_objective"] = "top2_primary_v1"
        if config.get("fourth_escape_constraint", False):
            if config.get("reference_mode") != "released_original_policy":
                raise ValueError("Escape constraint requires the original effective policy reference")
            record["_meta"]["source_initial_placement"] = placements(record["state"]["scores"])[row["seat"]]
            record["_meta"]["terminal_escape4_probabilities"] = {
                key: sum(x["ranks"][row["seat"]] < 4 for x in outcomes) / len(outcomes)
                for key, outcomes in zip(keys, label["outcomes"])}
        # Targets and future diagnostics are metadata; they never enter the state.
        probabilities = target_from_utilities(means, config["target_temperature"])
        record["questions"]["action"]["target"] = dict(zip(keys, probabilities))
        record["questions"]["action"]["label"] = keys[max(range(len(keys)), key=lambda i: probabilities[i])]
        if public_context_rows is not None:
            public_row = public_context_rows[row_id]
            if public_row["source_request_sha256"] != row["request_sha256"]:
                raise ValueError("Public-context source request differs")
            record["state"].setdefault("screen_observation", {}).update(public_row["screen_observation"])
        record = annotate_record(record, input_version, original_request=label["source"]["request"])
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
                normalized = annotate_record(record, input_version)
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
    native_rows = []
    terminal_native_records = {}
    if config.get("native_context_root"):
        from .native_retention import load_native_pairs, native_record
        native_rows = load_native_pairs(config)
        from .native_terminal import load_terminal_native_records
        terminal_native_records = load_terminal_native_records(config, native_rows)
        for row in native_rows:
            splits[row["split"]].append(terminal_native_records.get(row['id']) or native_record(row))
            groups[row["split"]].add(row["group_id"])
    if groups["train"] & groups["validation"]:
        raise ValueError("Combined replay/new groups overlap")
    train_ids = {r["_meta"]["id"] for r in splits["train"]}
    val_ids = {r["_meta"]["id"] for r in splits["validation"]}
    if len(train_ids) != len(splits["train"]) or len(val_ids) != len(splits["validation"]) or train_ids & val_ids:
        raise ValueError("Repeated record identity")
    reference_receipt = None
    if config.get("reference_regularization", False):
        from .reference import apply_reference
        reference_receipt = apply_reference(config, splits)
    balance_training_sources(splits["train"], config)
    for split, rows in splits.items():
        rows.sort(key=lambda r: digest(["endgame-train-order-v1", r["_meta"]["id"]]))
        jsonl(out / (split + ".jsonl"), rows)
    counts = Counter(r["questions"]["action"]["src"] for r in splits["train"])
    total = len(splits["train"])
    if native_rows and not config.get("reference_regularization", False):
        from .native_retention import reference_preparation_only
        mass, weights = reference_preparation_only(config, counts)
    else:
        mass, weights = loss_recipe(counts, config.get("new_terminal_loss_mass", .5),
                               config.get("balance_endgame_buckets", False),
                               config.get("reference_regularization", False),
                               config.get("loss_source_mass") if config.get("reference_regularization", False) else None)
    epochs = config.get("epochs", 1)
    if type(epochs) is not int or not 1 <= epochs <= 2:
        raise ValueError("Registered short fit must use one or two epochs")
    max_steps = min(epochs * math.ceil(total / 16), config.get("max_optimizer_steps", math.ceil(total / 16)))
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 3:
        raise ValueError("Short fit budget must be at least three optimizer updates")
    result = {"status": "complete_terminal_dataset_ready", "new_strata": dict(new_counts),
        "objective": "top2_primary_v1", "terminal_utility": [1.1, 1.0, 0.0, -0.05],
        "input_version": input_version,
        "admitted_new_strata": dict(admitted_counts), "neutral_menu_handling": dict(neutral_counts),
        "terminal_label_sets": dict(label_set_counts),
        "records": {s: len(r) for s, r in splits.items()},
        "group_overlap": 0, "loss_source_counts": dict(counts), "loss_source_mass": mass,
        "loss_source_weights": weights, "max_steps": max_steps, "epochs": epochs,
        "replay_inputs": replay_sources, "replay_admission": replay_admission,
        "reference_regularization": reference_receipt,
        "native_context_records": dict(Counter(r["split"] for r in native_rows)),
        "native_terminal_supervision_records": dict(Counter(r['_meta']['split'] for r in terminal_native_records.values())),
        "reference_preparation_only": bool(native_rows and not config.get("reference_regularization", False)),
        "validation_replay": "16 expert examples per phase; no fictitious call-window MC validation",
        "files": {s: {"path": str(out / (s + ".jsonl")),
                     "sha256": checksum(out / (s + ".jsonl"))} for s in splits},
        "code_sha256": {str(p): checksum(p) for p in Path(__file__).parent.glob("*.py")},
        "optimizer_updates": 0, "training_data_public": False}
    write(out / "preparation-private.json", result)
    return result

