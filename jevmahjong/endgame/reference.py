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


def restrict_distribution(probabilities, eligible):
    if not eligible or not set(eligible) <= set(probabilities):
        raise ValueError("Policy support differs from the legal menu")
    total = sum(probabilities[k] for k in eligible)
    if total <= 0:
        return {k: float(k == eligible[0]) for k in probabilities}
    return {k: probabilities[k] / total if k in eligible else 0. for k in probabilities}


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
            or cache.get("input_version") != config.get("input_version", "endgame-public-v2")
            or cache.get("reference_mode", "current_input_raw") != config.get("reference_mode", "current_input_raw")):
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
    from .public_policy import win_pass_allowed
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
        elif (config.get("reference_mode") == "released_original_policy" and meta["bucket"] == "win"
              and not win_pass_allowed(record)):
            q["src"], q["target"] = "replay_reference", dict(p)
            meta["target_basis"] = "fixed_public_win_scope_preserve_original_policy"
        elif meta.get("target_basis") != "finite_MC_tie_preserve_existing_legal_win":
            values = meta["expected_terminal_utilities"]
            previous_gains.append(sum((q["target"][k] - p[k]) * values[k] for k in p))
            eligible = list(p)
            if config.get("reference_mode") == "released_original_policy":
                from .public_policy import action_keys
                eligible = action_keys(record)
            supported = restrict_distribution(p, eligible)
            temperature = (config.get("reference_win_temperature", config["reference_temperature"])
                           if meta["bucket"] == "win" else config["reference_temperature"])
            target = improvement_target({k: supported[k] for k in eligible},
                {k: values[k] for k in eligible}, temperature, config["reference_exploration"])
            target = {k: target.get(k, 0.) for k in p}
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
            "reference_mode": config.get("reference_mode", "current_input_raw"),
            "teacher_is_frozen_weights_on_new_input_not_the_released_guarded_policy": config.get("reference_mode") != "released_original_policy"}


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
    mode = config.get("reference_mode", "current_input_raw")
    if mode not in ("current_input_raw", "released_original_policy"):
        raise ValueError("Unsupported reference mode")
    originals = {}
    if mode == "released_original_policy":
        from desktop.jev_sidecar.unified_policy import serving_request
        from .public_policy import original_policy_keys
        needed = {r["_meta"]["id"] for r in records if not r["_meta"]["id"].startswith("endgame:")}
        for field in ("replay_train", "replay_validation"):
            for raw in read_jsonl(config[field]):
                key = raw["_meta"]["id"]
                if key in needed:
                    candidate = inference_request(raw)
                    if key in originals and originals[key] != candidate:
                        raise ValueError("Ambiguous original replay input identity")
                    originals[key] = candidate
        if set(originals) != needed:
            raise ValueError("Original replay input missing")
    rows = {}
    for record in records:
        payload = inference_request(record)
        keys = list(payload["questions"]["action"]["criteria"])
        teacher, eligible = payload, keys
        if mode == "released_original_policy":
            key = record["_meta"]["id"]
            if key.startswith("endgame:"):
                raw = json.loads((out / "labels" / (key.split(":", 1)[1] + ".json")).read_text())["source"]["request"]
                raw = copy.deepcopy(raw); raw["state"].pop("endgame_context", None)
            else:
                raw = originals[key]
            teacher = serving_request(raw)
            if set(teacher["questions"]["action"]["criteria"]) != set(keys):
                raise ValueError("Original and student legal menus differ")
            eligible = original_policy_keys(teacher)
        if mode == "released_original_policy" and len(eligible) == 1:
            probabilities = [float(k == eligible[0]) for k in keys]
        else:
            result = model.system_one(**teacher)
            probabilities, _ = validate_distribution(result["answers"]["action"]["probabilities"], keys)
        prior = dict(zip(keys, probabilities))
        if mode == "released_original_policy":
            prior = restrict_distribution(prior, eligible)
        rows[record["_meta"]["id"]] = {"request_sha256": digest(payload),
                                        "teacher_request_sha256": digest(teacher),
                                        "teacher_eligible_keys": eligible, "probabilities": prior}
        if len(rows) % 50 == 0 or len(rows) == len(records):
            progress = {"status": "building_frozen_reference", "records_complete": len(rows),
                        "records_total": len(records), "optimizer_updates": 0}
            write(out / "reference-status-private.json", progress)
            print(json.dumps(progress), flush=True)
    cache = {"status": "complete_frozen_reference", "identity_sha256": frozen_identity,
             "reference_mode": mode,
             "input_version": config.get("input_version", "endgame-public-v2"), "rows": rows}
    write(out / "reference-targets-private.json", cache)
    result = {k: v for k, v in cache.items() if k != "rows"}
    result.update(records=len(rows), optimizer_updates=0)
    write(out / "reference-status-private.json", result)
    return result
