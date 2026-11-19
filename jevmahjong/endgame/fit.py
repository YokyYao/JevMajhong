"""Short continued LoRA fit selected by held-out terminal utility, with a replay guard."""
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import sys
import time

from .collect import write
from .prepare import checksum, prepare, read_jsonl


def target_ce(question, probabilities):
    import math
    keys = list(question["criteria"])
    target = question.get("target")
    if target is None:
        return -math.log(max(probabilities[question["label"]], 1e-30))
    support = [key for key in keys if target[key] > 0]
    mass = sum(probabilities[key] for key in support) if question["src"] == "replay_masked" else sum(probabilities.values())
    total = sum(target[key] for key in support)
    return -sum(target[key] / total * math.log(max(probabilities[key] / mass, 1e-30)) for key in support)


def terminal_floor_guard(step, current, original, initial):
    """Retain zero as an anchor; nonzero must clear both reference floors."""
    if not current or len(current) != len(original):
        raise ValueError("Required terminal validation coverage is missing")
    mean, baseline = sum(current) / len(current), sum(original) / len(original)
    if initial is None:
        if step != 0:
            raise ValueError("Initial validation anchor is missing")
        initial = mean
    floor = max(initial, baseline)
    return (step == 0 or mean >= floor - 1e-12), {
        "positions": len(current), "candidate": mean, "original_policy": baseline,
        "initial_input_path": initial, "required_nonzero_floor": floor,
        "zero_step_is_anchor": step == 0}


def fit(config):
    if config.get("fit_registered", True) is not True:
        raise ValueError("This data-audit configuration has no registered fit protocol")
    out = Path(config["out"])
    proof = prepare(config)
    train_dir = out / "training-run"
    if train_dir.exists():
        raise ValueError("Training output exists; inspect it instead of silently restarting")
    sys.path.insert(0, config["upstream_source"])
    import torch
    from jevany import train as trainer
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    if not str(Path(trainer.__file__).resolve()).startswith(str(Path(config["upstream_source"]).resolve())):
        raise ValueError("Unexpected upstream trainer")
    records = read_jsonl(out / "validation.jsonl")
    by_id = {r["_meta"]["id"]: r for r in records}
    reference_rows = None
    if config.get("reference_regularization", False):
        from .reference import load_reference
        reference_rows = load_reference(config, read_jsonl(out / "train.jsonl") + records)["rows"]
    expected = {str(out / (s + ".jsonl")): proof["files"][s]["sha256"] for s in ("train", "validation")}
    original_loss = trainer.question_loss
    original_eval = trainer.evaluate_during_training
    original_batch = trainer.batch_loss
    original_warm = trainer.Checkpoint.warm_start
    original_admit = trainer.training_requests
    original_encode = trainer.encode_batch
    original_stop = trainer.early_stop_update
    from .exposure import TrainingExposure, exposure_early_stop_update
    exposure = TrainingExposure(read_jsonl(out / "train.jsonl"), config.get("minimum_unique_exposure"))
    curve, state = [], {}
    weights = proof["loss_source_weights"]
    lr = config.get("learning_rate", 2e-6)
    head_lr = config.get("head_learning_rate", 1e-5)
    evaluation_every = config.get("evaluation_every_steps", 16)
    if (any(not math.isfinite(x) or x <= 0 for x in (lr, head_lr))
            or type(evaluation_every) is not int or not 1 <= evaluation_every <= proof["max_steps"]):
        raise ValueError("Invalid registered learning rate or evaluation cadence")

    def loss(logits, question, device):
        source = question["src"]
        if source not in weights:
            raise ValueError("Unregistered loss source")
        q = dict(question)
        if q.get("target") is not None:
            target = torch.tensor(q["target"], dtype=logits.dtype, device=device)
            if source == "replay_masked":
                mask = target > 0
                target, logits = target[mask], logits[mask]
            target = target / target.sum()
            value = -(target * logits.log_softmax(-1)).sum()
        else:
            value = original_loss(logits, question, device)
        return weights[source] * value

    def warm(checkpoint, model, meta):
        result = original_warm(checkpoint, model, meta)
        actual = get_peft_model_state_dict(model.lm, save_embedding_layers=False)
        previous = load_file(str(Path(config["checkpoint"]) / "adapter_model.safetensors"), device="cpu")
        if set(actual) != set(previous) or any(not torch.equal(actual[k].detach().cpu(), previous[k]) for k in actual):
            raise ValueError("Loaded LoRA differs from the published v2 source")
        head = torch.load(Path(config["checkpoint"]) / "head.pt", map_location="cpu", weights_only=True)["head"]
        if any(not torch.equal(v.detach().cpu(), head[k]) for k, v in model.head.state_dict().items()):
            raise ValueError("Loaded action head differs from baseline")
        state["adapter_initial"] = {k: v.detach().cpu().clone() for k, v in actual.items()}
        state["head_initial"] = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        if any(p.requires_grad for n, p in model.lm.named_parameters() if "lora_" not in n):
            raise ValueError("Backbone must remain frozen")
        write(out / "initialization-private.json", {"parent": config["checkpoint"],
            "exact_parent_adapter_and_head": True, "same_rank16_lora_and_head": True,
            "new_optimizer": True, "external_residual_networks": 0})
        return result

    def admitted(*args, **kwargs):
        requests, report = original_admit(*args, **kwargs)
        if report["rejected_records"] or report["admitted_records"] != proof["records"]["train"]:
            raise ValueError("Prepared records were dropped during token admission")
        write(out / "token-admission-private.json", report)
        return requests, report

    def evaluate(*args, **kwargs):
        step = kwargs.get("step", args[5] if len(args) > 5 else None)
        directory = Path(kwargs.get("out_dir", args[4]))
        summary = original_eval(*args, **kwargs)
        rows = json.loads((directory / "training_eval" / ("step-%06d" % step) / "development/rows.json").read_text())
        if len(rows) != len(records) or {r["id"] for r in rows} != set(by_id):
            raise ValueError("Validation dropped or duplicated records")
        utilities, ce, reference_kl = defaultdict(list), defaultdict(list), []
        fourth_escape, original_fourth_escape, third_pursuit, original_third_pursuit = [], [], [], []
        for prediction in rows:
            record = by_id[prediction["id"]]
            question = record["questions"]["action"]
            probabilities = dict(zip(prediction["keys"], prediction["p"]))
            if set(probabilities) != set(question["criteria"]):
                raise ValueError("Validation legal menu differs")
            source = question["src"]
            ce[source].append(target_ce(question, probabilities))
            if reference_rows is not None and source.startswith("replay_"):
                p = reference_rows[record["_meta"]["id"]]["probabilities"]
                current = probabilities
                if config.get("reference_mode") == "released_original_policy":
                    from .public_policy import action_keys
                    from .reference import restrict_distribution
                    current = restrict_distribution(probabilities, action_keys(record))
                reference_kl.append(sum(v * math.log(max(v, 1e-30) / max(current[k], 1e-30))
                                        for k, v in p.items() if v > 0))
            if source == "endgame_terminal":
                eligible = list(probabilities)
                from .context import PUBLIC_SETTLEMENT_VERSIONS
                if config.get("input_version") in PUBLIC_SETTLEMENT_VERSIONS:
                    from .public_policy import action_keys
                    eligible = action_keys(record)
                chosen = max(eligible, key=probabilities.get)
                utilities[record["_meta"]["bucket"]].append(record["_meta"]["expected_terminal_utilities"][chosen])
                if config.get("fourth_escape_constraint", False) and record["_meta"]["source_initial_placement"] == 4:
                    escape = record["_meta"]["terminal_escape4_probabilities"]
                    baseline = reference_rows[record["_meta"]["id"]]["probabilities"]
                    fourth_escape.append(escape[chosen])
                    original_fourth_escape.append(escape[max(baseline, key=baseline.get)])
                if config.get("fourth_escape_constraint", False) and record["_meta"]["source_initial_placement"] == 3:
                    baseline = reference_rows[record["_meta"]["id"]]["probabilities"]
                    values = record["_meta"]["expected_terminal_utilities"]
                    third_pursuit.append(values[chosen])
                    original_third_pursuit.append(values[max(baseline, key=baseline.get)])
        required = {"win", "third", "fourth", "hold"}
        if set(utilities) != required:
            raise ValueError("Validation missing registered endgame stratum")
        utility = sum(sum(v) / len(v) for v in utilities.values()) / len(utilities)
        replay = [x for k, values in ce.items() if k.startswith("replay_") for x in values]
        replay_ce = sum(replay) / len(replay)
        if step == 0:
            state["initial_replay_ce"] = replay_ce
        strata = {k: sum(v) / len(v) for k, v in utilities.items()}
        if step == 0:
            state["initial_strata"] = strata
        replay_guard = replay_ce <= state["initial_replay_ce"] + 0.10
        placement_guard = (not config.get("enforce_placement_strata_guard", False)
            or (strata["third"] >= state["initial_strata"]["third"] - 1e-12
                and (config.get("fourth_escape_constraint", False)
                     or strata["fourth"] >= state["initial_strata"]["fourth"] - 1e-12)
                and strata["hold"] >= state["initial_strata"]["hold"] - .02 - 1e-12))
        mean_reference_kl = sum(reference_kl) / len(reference_kl) if reference_kl else None
        if step == 0:
            state["initial_reference_kl"] = mean_reference_kl
        reference_limit = config.get("reference_kl_guard")
        if config.get("reference_kl_guard_relative_to_start", False):
            reference_limit += state["initial_reference_kl"]
        reference_guard = mean_reference_kl is None or mean_reference_kl <= reference_limit
        other_guards_passed = replay_guard and placement_guard and reference_guard
        win_guard = (not config.get("enforce_win_stratum_guard", False)
                     or strata["win"] >= state["initial_strata"]["win"] - 1e-12)
        escape_guard = third_guard = True
        escape_audit, third_audit = None, None
        if config.get("fourth_escape_constraint", False):
            # Keep step zero as the valid no-update anchor even when the new
            # input path starts below original policy. This guard determines
            # nonzero eligibility; before full coverage it does not drain
            # patience, just like the active-win guard.
            escape_guard, escape_audit = terminal_floor_guard(step, fourth_escape,
                original_fourth_escape, state.get("initial_fourth_escape"))
            third_guard, third_audit = terminal_floor_guard(step, third_pursuit,
                original_third_pursuit, state.get("initial_third_pursuit"))
            if step == 0:
                state["initial_fourth_escape"] = escape_audit["candidate"]
                state["initial_third_pursuit"] = third_audit["candidate"]
        exposure_snapshot = exposure.snapshot()
        exposure_guard = step == 0 or exposure_snapshot["minimum_exposure_met"]
        guard_passed = other_guards_passed and win_guard and exposure_guard and escape_guard and third_guard
        state["evaluation_step"] = step
        state["other_guards_passed"] = other_guards_passed
        state["exposure_met"] = exposure_snapshot["minimum_exposure_met"]
        selection_metric = -utility if guard_passed else 1e6 + replay_ce
        item = {"step": step, "endgame_validation_terminal_utility": utility,
            "utility_by_stratum": strata,
            "replay_target_ce": replay_ce, "replay_guard_passed": replay_guard,
            "placement_strata_guard_passed": placement_guard, "all_guards_passed": guard_passed,
            "replay_reference_kl": mean_reference_kl, "reference_guard_passed": reference_guard,
            "reference_kl_guard_limit": reference_limit,
            "training_exposure": exposure_snapshot, "exposure_guard_passed": exposure_guard,
            "win_stratum_guard_passed": win_guard,
            "fourth_escape_guard_passed": escape_guard, "fourth_escape_validation": escape_audit,
            "third_pursuit_guard_passed": third_guard, "third_pursuit_validation": third_audit,
            "selection_metric_negative_terminal_utility": selection_metric,
            "target_ce_by_source": {k: sum(v) / len(v) for k, v in ce.items()},
            "validation_is_not_independent_strength_confirmation": True}
        curve.append(item)
        write(out / "validation-progress-private.json", curve)
        write(out / "training-exposure-private.json", dict(exposure_snapshot, step=step,
              seen_record_ids=sorted(exposure.seen), based_on_successful_forward_batches=True))
        write(out / "training-status-private.json", dict(status="validation", **item))
        print(json.dumps(item), flush=True)
        result = copy.deepcopy(summary)
        result["clean"]["nll"] = selection_metric
        return result

    def batch(*args, **kwargs):
        state["micro_batches"] = state.get("micro_batches", 0) + 1
        steps = (state["micro_batches"] - 1) // 16
        if steps >= 2 and not state.get("update_proof"):
            model = args[0]
            actual = get_peft_model_state_dict(model.lm, save_embedding_layers=False)
            changed_adapter = any(not torch.equal(v.detach().cpu(), state["adapter_initial"][k]) for k, v in actual.items())
            changed_head = any(not torch.equal(v.detach().cpu(), state["head_initial"][k]) for k, v in model.head.state_dict().items())
            if not changed_adapter or not changed_head:
                raise ValueError("No actual LoRA and head update")
            state["update_proof"] = True
            write(out / "actual-update-proof-private.json", {
                "completed_optimizer_steps_at_least": steps, "adapter_changed": True, "head_changed": True})
        if state["micro_batches"] % 16 == 1:
            write(out / "training-status-private.json", {"status": "training",
                "completed_optimizer_steps_at_least": steps, "max_steps": proof["max_steps"],
                "actual_update_proof": bool(state.get("update_proof"))})
        result = original_batch(*args, **kwargs)
        exposure.observe(v.endgame_record_id for v in args[2])
        return result

    def encode(model, tok, a, chunk, epoch):
        result = original_encode(model, tok, a, chunk, epoch)
        if len(result) != len(chunk):
            raise ValueError("Exposure ledger requires the registered one-variant recipe")
        for variant, record in zip(result, chunk):
            variant.endgame_record_id = record["_meta"]["id"]
        return result

    def stop(value, best, stale, **kwargs):
        return exposure_early_stop_update(original_stop, value, best, stale,
            step=state["evaluation_step"], exposure_met=state["exposure_met"],
            other_guards_passed=state["other_guards_passed"], **kwargs)

    trainer.question_loss, trainer.evaluate_during_training = loss, evaluate
    trainer.batch_loss, trainer.Checkpoint.warm_start = batch, warm
    trainer.training_requests = admitted
    trainer.encode_batch, trainer.early_stop_update = encode, stop
    args = [
        "--base", "Qwen/Qwen3.5-0.8B", "--base_revision", "2fc06364715b967f1860aea9cf38778875588b17",
        "--base_load_path", config["base_load_path"],
        "--data", str(out / "train.jsonl"), "--eval_data", str(out / "validation.jsonl"),
        "--out", str(train_dir), "--init_from", config["checkpoint"],
        "--device", "cuda", "--weights_dtype", "bf16", "--dtype", "bf16",
        "--lora", "16", "--lora_targets", "all", "--lora_dropout", "0.05",
        "--head_dim", "256", "--decision_mode", "pointer", "--branch_mode", "rows",
        "--backbone_adapter", "text", "--epochs", str(proof["epochs"]), "--max_steps", str(proof["max_steps"]),
        "--lr", str(lr), "--head_lr", str(head_lr), "--weight_decay", "0.001",
        "--batch", "1", "--accum", "16", "--checkpointing", "1",
        "--checkpoint_every_steps", str(evaluation_every), "--eval_before_start",
        "--eval_every_steps", str(evaluation_every), "--early_stop_patience", "2",
        "--early_stop_metric", "development_nll", "--early_stop_min_delta", "0.0001",
        "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--p_none_pair", "0",
        "--max_state", "4096", "--max_branch", "8192", "--max_packed", "8192",
        "--seed", "2026111907", "--wandb_mode", "disabled"]
    write(out / "fit-protocol-private.json", {
        "parent": config["checkpoint"], "args": args, "data_preparation": proof,
        "selection_objective": "maximum equally-stratified held-out terminal utility, replay CE guard +0.10",
        "selection_metric_transport": "upstream development_nll carries negative terminal utility only in memory; raw CE retained separately",
        "stage": "pilot_not_deployed", "raw_records_public": False})
    try:
        trainer.main(args)
        for file, expected_digest in expected.items():
            if checksum(file) != expected_digest:
                raise ValueError("Registered training data changed")
        selection = json.loads((train_dir / "selection.json").read_text())
        result = {"status": "endgame_pilot_training_complete",
            "selection": selection, "actual_updates_verified": bool(state.get("update_proof")),
            "eligible_for_development": selection["step"] > 0,
            "validation": curve, "deployed": False}
        write(out / "fit-result-private.json", result)
        write(out / "training-status-private.json", result)
        return result
    finally:
        trainer.question_loss, trainer.evaluate_during_training = original_loss, original_eval
        trainer.batch_loss, trainer.Checkpoint.warm_start = original_batch, original_warm
        trainer.training_requests = original_admit
        trainer.encode_batch, trainer.early_stop_update = original_encode, original_stop

