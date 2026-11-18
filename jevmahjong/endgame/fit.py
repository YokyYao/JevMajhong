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


def fit(config):
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
                reference_kl.append(sum(v * math.log(max(v, 1e-30) / max(probabilities[k], 1e-30))
                                        for k, v in p.items() if v > 0))
            if source == "endgame_terminal":
                chosen = max(probabilities, key=probabilities.get)
                utilities[record["_meta"]["bucket"]].append(record["_meta"]["expected_terminal_utilities"][chosen])
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
                and strata["fourth"] >= state["initial_strata"]["fourth"] - 1e-12
                and strata["hold"] >= state["initial_strata"]["hold"] - .02 - 1e-12))
        mean_reference_kl = sum(reference_kl) / len(reference_kl) if reference_kl else None
        reference_guard = mean_reference_kl is None or mean_reference_kl <= config["reference_kl_guard"]
        guard_passed = replay_guard and placement_guard and reference_guard
        selection_metric = -utility if guard_passed else 1e6 + replay_ce
        item = {"step": step, "endgame_validation_terminal_utility": utility,
            "utility_by_stratum": strata,
            "replay_target_ce": replay_ce, "replay_guard_passed": replay_guard,
            "placement_strata_guard_passed": placement_guard, "all_guards_passed": guard_passed,
            "replay_reference_kl": mean_reference_kl, "reference_guard_passed": reference_guard,
            "selection_metric_negative_terminal_utility": selection_metric,
            "target_ce_by_source": {k: sum(v) / len(v) for k, v in ce.items()},
            "validation_is_not_independent_strength_confirmation": True}
        curve.append(item)
        write(out / "validation-progress-private.json", curve)
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
        return original_batch(*args, **kwargs)

    trainer.question_loss, trainer.evaluate_during_training = loss, evaluate
    trainer.batch_loss, trainer.Checkpoint.warm_start = batch, warm
    trainer.training_requests = admitted
    args = [
        "--base", "Qwen/Qwen3.5-0.8B", "--base_revision", "2fc06364715b967f1860aea9cf38778875588b17",
        "--base_load_path", config["base_load_path"],
        "--data", str(out / "train.jsonl"), "--eval_data", str(out / "validation.jsonl"),
        "--out", str(train_dir), "--init_from", config["checkpoint"],
        "--device", "cuda", "--weights_dtype", "bf16", "--dtype", "bf16",
        "--lora", "16", "--lora_targets", "all", "--lora_dropout", "0.05",
        "--head_dim", "256", "--decision_mode", "pointer", "--branch_mode", "rows",
        "--backbone_adapter", "text", "--epochs", "1", "--max_steps", str(proof["max_steps"]),
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

