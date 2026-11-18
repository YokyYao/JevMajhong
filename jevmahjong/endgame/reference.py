"""Frozen-v8 distributions and terminal-utility improvement targets, offline only."""
import copy
import json
import math
from pathlib import Path
import sys

from .collect import digest, write


def inference_request(record):
    """Strip labels and metadata before the same input is sent to a frozen teacher."""
    return {"state": copy.deepcopy(record["state"]), "questions": {
        name: {key: copy.deepcopy(value) for key, value in question.items()
               if key in ("type", "instructions", "criteria")}
        for name, question in record["questions"].items()}}


def distribution(values, keys):
    if set(values) != set(keys) or not keys:
        raise ValueError("Reference menu differs from the current legal menu")
    p = [values[key] for key in keys]
    if any(not math.isfinite(x) or x < 0 for x in p) or abs(sum(p) - 1) > 1e-5:
        raise ValueError("Reference probabilities must form a finite distribution")
    return [x / sum(p) for x in p]


def improvement_target(reference, utilities, temperature=.1, exploration=.1):
    """Tilt a reference distribution by utility, rather than erase its preferences."""
    keys = list(reference)
    p = distribution(reference, keys)
    if (set(utilities) != set(keys) or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(exploration) or not 0 <= exploration < 1):
        raise ValueError("Invalid reference-improvement recipe")
    values = [utilities[key] for key in keys]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Terminal utilities must be finite")
    if max(values) - min(values) <= 1e-12:
        return dict(zip(keys, p))
    advantages = [math.exp((v - max(values)) / temperature) for v in values]
    prior = [(1 - exploration) * x + exploration / len(p) for x in p]
    weights = [a * b for a, b in zip(prior, advantages)]
    q = [x / sum(weights) for x in weights]
    if sum(x * v for x, v in zip(q, values)) < sum(x * v for x, v in zip(p, values)) - 1e-12:
        # A uniform exploration floor must not reduce the known expected utility.
        weights = [a * b for a, b in zip(p, advantages)]
        if sum(weights) == 0:
            return dict(zip(keys, p))
        q = [x / sum(weights) for x in weights]
    return dict(zip(keys, q))


def load_reference(config, records):
    from .futures import bundle_identity
    out = Path(config["out"])
    path = out / "reference-targets-private.json"
    cache = json.loads(path.read_text())
    if (cache.get("status") != "complete_frozen_reference"
            or cache.get("identity_sha256") != bundle_identity(config["checkpoint"])["identity_sha256"]
            or cache.get("input_version") != "endgame-public-v2"):
        raise ValueError("Reference cache identity or format differs")
    if {r["_meta"]["id"] for r in records} != set(cache["rows"]):
        raise ValueError("Reference cache record set differs")
    for record in records:
        reference = cache["rows"][record["_meta"]["id"]]
        if reference["request_sha256"] != digest(inference_request(record)):
            raise ValueError("Reference inference input changed")
        distribution(reference["probabilities"], list(record["questions"]["action"]["criteria"]))
    return cache


def apply_reference(config, splits):
    cache = load_reference(config, splits["train"] + splits["validation"])
    changes, expected_gains, previous_gains = [], [], []
    expert_count = expert_disagreements = 0
    for record in splits["train"]:
        meta, q = record["_meta"], record["questions"]["action"]
        p = cache["rows"][meta["id"]]["probabilities"]
        meta["original_target_source"] = q["src"]
        if q["src"].startswith("replay_"):
            if q["src"] == "replay_expert":
                expert_count += 1
                expert_disagreements += max(p, key=p.get) != q["label"]
            q["src"], q["target"] = "replay_reference", dict(p)
        elif meta.get("target_basis") != "finite_MC_tie_preserve_existing_legal_win":
            values = meta["expected_terminal_utilities"]
            previous_gains.append(sum((q["target"][k] - p[k]) * values[k] for k in p))
            target = improvement_target(p, values, config["reference_temperature"], config["reference_exploration"])
            changes.append(sum(abs(target[k] - p[k]) for k in p) / 2)
            expected_gains.append(sum((target[k] - p[k]) * values[k] for k in p))
            q["target"] = target
            meta["target_basis"] = "frozen_reference_tilted_by_terminal_utility"
        q["label"] = max(q["target"], key=q["target"].get)
    return {"reference_identity_sha256": cache["identity_sha256"],
            "input_version": cache["input_version"], "reference_rows": len(cache["rows"]),
            "mean_training_terminal_target_tv_from_reference": sum(changes) / len(changes),
            "mean_training_target_expected_utility_gain": sum(expected_gains) / len(expected_gains),
            "minimum_training_target_expected_utility_gain": min(expected_gains),
            "previous_absolute_target_mean_expected_utility_delta": sum(previous_gains) / len(previous_gains),
            "previous_absolute_targets_with_negative_expected_delta": sum(x < -1e-12 for x in previous_gains),
            "old_expert_labels": expert_count, "old_expert_reference_argmax_disagreements": expert_disagreements,
            "teacher_is_frozen_weights_on_new_input_not_the_released_guarded_policy": True}


def build_reference(config):
    from .prepare import prepare, read_jsonl
    from .futures import bundle_identity
    from desktop.jev_sidecar.unified_policy import identity
    out = Path(config["out"])
    if (out / "training-run").exists() or (out / "reference-targets-private.json").exists():
        raise ValueError("Existing reference or fit output must be inspected before rebuilding")
    ordinary = dict(config, reference_regularization=False)
    prepare(ordinary)
    records = read_jsonl(out / "train.jsonl") + read_jsonl(out / "validation.jsonl")
    frozen_identity = bundle_identity(config["checkpoint"])["identity_sha256"]
    if frozen_identity not in (identity()["source_identity_sha256"], identity()["identity_sha256"]):
        raise ValueError("The frozen reference must be exactly released v8")
    sys.path.insert(0, config["upstream_source"])
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    model = load_model(config["checkpoint"])
    rows = {}
    for record in records:
        payload = inference_request(record)
        keys = list(payload["questions"]["action"]["criteria"])
        result = model.system_one(**payload)
        probabilities, _ = validate_distribution(result["answers"]["action"]["probabilities"], keys)
        rows[record["_meta"]["id"]] = {"request_sha256": digest(payload),
                                        "probabilities": dict(zip(keys, probabilities))}
        if len(rows) % 50 == 0 or len(rows) == len(records):
            progress = {"status": "building_frozen_reference", "records_complete": len(rows),
                        "records_total": len(records), "optimizer_updates": 0}
            write(out / "reference-status-private.json", progress)
            print(json.dumps(progress), flush=True)
    cache = {"status": "complete_frozen_reference", "identity_sha256": frozen_identity,
             "input_version": "endgame-public-v2", "rows": rows}
    write(out / "reference-targets-private.json", cache)
    result = {k: v for k, v in cache.items() if k != "rows"}
    result.update(records=len(rows), optimizer_updates=0)
    write(out / "reference-status-private.json", result)
    return result
