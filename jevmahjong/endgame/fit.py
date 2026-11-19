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
from .native_terminal import is_native_terminal


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


def fourth_cohort_floor_guards(step, current, original, initial, native_registered=False):
    """New ordinary windows must never compensate for a legacy escape regression."""
    expected = {'existing', 'native_fourth'} if native_registered else {'existing'}
    if set(current) != expected or set(original) != expected or set(initial) != expected:
        raise ValueError('Required fourth-place validation cohorts differ')
    if native_registered and (len(current['existing']) != 43 or len(current['native_fourth']) != 8):
        raise ValueError('Keep all43 old and8 new fourth-place validation windows separately')
    passed, audits = {}, {}
    for key in sorted(expected):
        passed[key], audits[key] = terminal_floor_guard(step, current[key], original[key], initial[key])
    return passed, audits


def original_replay_ce(ce):
    values = [x for source, rows in ce.items()
              if source.startswith("replay_") and not source.startswith("replay_native") for x in rows]
    if not values:
        raise ValueError("Original replay validation coverage is missing")
    return sum(values) / len(values)


def reference_kl_group(source,meta=None):
    if is_native_terminal(meta or {}):return 'trailing'
    if source=='replay_native_leading':return 'leading'
    if source=='replay_native':return 'trailing'
    return 'original'


def training_compute_options(config):
    scope=config.get('trainable_scope','lora_and_head')
    dtype=config.get('training_compute_dtype','bf16')
    disable=config.get('disable_frozen_encoder_input_grads',False)
    if dtype not in ('bf16','fp32') or type(disable) is not bool:
        raise ValueError('Unsupported registered compute precision or frozen-input setting')
    if (dtype=='fp32' or disable) and scope!='head_only':
        raise ValueError('Frozen-encoder optimization requires head-only training')
    if disable and dtype!='fp32':
        raise ValueError('Frozen-input optimization requires the registered FP32 readout')
    return {'dtype':dtype,'checkpointing':0 if disable else 1,
            'disable_frozen_encoder_input_grads':disable}


def configure_parameter_scope(model,scope,disable_frozen_encoder_input_grads=False):
    if scope not in ('lora_and_head','head_only'):
        raise ValueError('Unsupported registered trainable parameter scope')
    if disable_frozen_encoder_input_grads and scope!='head_only':
        raise ValueError('Input gradient hooks may only be disabled for a frozen encoder')
    if scope=='head_only':
        model.requires_grad_(False)
        model.head.requires_grad_(True)
        model.lm.eval()
        if disable_frozen_encoder_input_grads:
            # PEFT can install an embedding-output gradient hook even though every
            # encoder parameter is frozen. Remove it through the backbone API.
            disable=getattr(model.lm,'disable_input_require_grads',None)
            checkpointing_disable=getattr(model.lm,'gradient_checkpointing_disable',None)
            if not callable(disable) or not callable(checkpointing_disable):
                raise ValueError('Frozen encoder does not expose the required gradient controls')
            checkpointing_disable()
            disable()
    head_ids={id(p) for p in model.head.parameters()}
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not trainable or not all(p.requires_grad for p in model.head.parameters()):
        raise ValueError('The existing action head must remain trainable')
    if scope=='head_only' and any(id(p) not in head_ids for _,p in trainable):
        raise ValueError('Head-only fit must freeze every encoder parameter')
    return {'scope':scope,'trainable_parameters':sum(p.numel() for _,p in trainable),
        'head_parameters':sum(p.numel() for p in model.head.parameters()),
        'trainable_encoder_parameters':sum(p.numel() for _,p in trainable if id(p) not in head_ids),
        'frozen_encoder_input_grads_disabled':disable_frozen_encoder_input_grads}


def verify_frozen_readout(model,features):
    """Check real head inputs, not just the requested training flags."""
    import torch
    if not torch.is_grad_enabled() or not model.head.training or model.lm.training:
        raise ValueError('Readout audit requires training with an eval-mode frozen encoder')
    if len(features)!=2 or any(x.requires_grad or x.dtype!=torch.float32 for x in features):
        raise ValueError('Frozen readout features must be detached FP32 tensors')
    if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
        raise ValueError('FP32 head optimization must not run inside autocast')
    if any(p.dtype!=torch.float32 or not p.requires_grad for p in model.head.parameters()):
        raise ValueError('Existing head must have trainable FP32 parameters')
    return {'encoder_features_require_grad':False,'readout_dtype':'fp32',
            'autocast_enabled':False,'encoder_training':False,'head_training':True}


def validate_parameter_updates(scope,adapter_changed,head_changed):
    if scope=='head_only':
        if adapter_changed or not head_changed:
            raise ValueError('Head-only fit requires unchanged LoRA and an actual head update')
    elif scope=='lora_and_head':
        if not adapter_changed or not head_changed:
            raise ValueError('Joint fit requires actual LoRA and head updates')
    else:raise ValueError('Unsupported registered trainable parameter scope')


def selection_value(step,utility,replay_ce,guards_passed,zero_feasible,feasibility_first=False):
    if not feasibility_first:
        return -utility if guards_passed else 1e6+replay_ce
    if step==0:
        return -utility if zero_feasible else 1e6
    # All rejected nonzero steps are strictly worse than the no-update anchor,
    # while every feasible nonzero step outranks an infeasible anchor.
    return -utility if guards_passed else 1e6+1+replay_ce


def fit(config):
    if config.get("fit_registered", True) is not True:
        raise ValueError("This data-audit configuration has no registered fit protocol")
    parameter_scope=config.get('trainable_scope','lora_and_head')
    compute=training_compute_options(config)
    if config.get('native_terminal_label_root') and config.get('enforce_native_terminal_escape_guard') is not True:
        raise ValueError('Native terminal validation requires a separate registered escape guard')
    if parameter_scope not in ('lora_and_head','head_only'):
        raise ValueError('Unsupported registered trainable parameter scope')
    if config.get("native_context_root"):
        native_limit = config.get("native_reference_kl_guard")
        if type(native_limit) not in (float, int) or not math.isfinite(native_limit) or native_limit <= 0:
            raise ValueError("Native policy retention requires an explicit positive validation KL limit")
    if config.get('native_leading_context_root'):
        leading_limit=config.get('leading_native_reference_kl_guard')
        if type(leading_limit) not in (float,int) or not math.isfinite(leading_limit) or leading_limit<=0:
            raise ValueError('Leading native retention requires its own positive KL limit')
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
        scope_proof=configure_parameter_scope(model,parameter_scope,compute['disable_frozen_encoder_input_grads'])
        if compute['disable_frozen_encoder_input_grads']:
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            def audit_readout(head,features):
                if not torch.is_grad_enabled() or not head.training:return
                state['readout_proof']=verify_frozen_readout(model,features)
                write(out / 'frozen-readout-proof-private.json',state['readout_proof'])
                state['readout_hook'].remove()
            state['readout_hook']=model.head.register_forward_pre_hook(audit_readout)
        state['model']=model
        state['parameter_scope']=scope_proof
        write(out / "initialization-private.json", {"parent": config["checkpoint"],
            "exact_parent_adapter_and_head": True, "same_rank16_lora_and_head": True,
            "new_optimizer": True, "external_residual_networks": 0,'parameter_scope':scope_proof,
            'encoder_dropout_disabled':parameter_scope=='head_only','compute':compute})
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
        utilities, ce, reference_kl, native_reference_kl, leading_reference_kl = defaultdict(list), defaultdict(list), [], [], []
        fourth_escape, original_fourth_escape, third_pursuit, original_third_pursuit = [], [], [], []
        native_fourth_escape, original_native_fourth_escape = [], []
        for prediction in rows:
            record = by_id[prediction["id"]]
            question = record["questions"]["action"]
            probabilities = dict(zip(prediction["keys"], prediction["p"]))
            if set(probabilities) != set(question["criteria"]):
                raise ValueError("Validation legal menu differs")
            source = question["src"]
            ce[source].append(target_ce(question, probabilities))
            if reference_rows is not None and (source.startswith("replay_") or is_native_terminal(record['_meta'])):
                p = reference_rows[record["_meta"]["id"]]["probabilities"]
                current = probabilities
                if config.get("reference_mode") == "released_original_policy":
                    from .public_policy import action_keys
                    from .reference import restrict_distribution
                    current = restrict_distribution(probabilities, action_keys(record))
                destination = {'leading':leading_reference_kl,'trailing':native_reference_kl,
                               'original':reference_kl}[reference_kl_group(source, record['_meta'])]
                destination.append(sum(v * math.log(max(v, 1e-30) / max(current[k], 1e-30))
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
                    current_values, original_values = ((native_fourth_escape, original_native_fourth_escape)
                        if is_native_terminal(record['_meta']) else (fourth_escape, original_fourth_escape))
                    current_values.append(escape[chosen])
                    original_values.append(escape[max(baseline, key=baseline.get)])
                if config.get("fourth_escape_constraint", False) and record["_meta"]["source_initial_placement"] == 3:
                    baseline = reference_rows[record["_meta"]["id"]]["probabilities"]
                    values = record["_meta"]["expected_terminal_utilities"]
                    third_pursuit.append(values[chosen])
                    original_third_pursuit.append(values[max(baseline, key=baseline.get)])
        required = {"win", "third", "fourth", "hold"}
        if set(utilities) != required:
            raise ValueError("Validation missing registered endgame stratum")
        utility = sum(sum(v) / len(v) for v in utilities.values()) / len(utilities)
        replay_ce = original_replay_ce(ce)
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
        native_kl = sum(native_reference_kl) / len(native_reference_kl) if native_reference_kl else None
        native_guard = True
        if config.get("native_context_root"):
            if len(native_reference_kl) != 16:
                raise ValueError("Native context validation must retain all16 frozen-policy examples")
            # Like the pursuit/escape guard, native alignment determines
            # nonzero eligibility after exposure; zero stays a valid anchor.
            native_guard = step == 0 or native_kl <= config["native_reference_kl_guard"]
        leading_kl=sum(leading_reference_kl)/len(leading_reference_kl) if leading_reference_kl else None
        leading_guard=True
        if config.get('native_leading_context_root'):
            if len(leading_reference_kl)!=16:
                raise ValueError('Leading native validation must retain all16 frozen-policy examples')
            leading_guard=step==0 or leading_kl<=config['leading_native_reference_kl_guard']
        win_guard = (not config.get("enforce_win_stratum_guard", False)
                     or strata["win"] >= state["initial_strata"]["win"] - 1e-12)
        escape_guard = third_guard = True
        escape_audit, third_audit = None, None
        native_escape_guard, native_escape_audit = True, None
        if config.get("fourth_escape_constraint", False):
            # Keep step zero as the valid no-update anchor even when the new
            # input path starts below original policy. This guard determines
            # nonzero eligibility; before full coverage it does not drain
            # patience, just like the active-win guard.
            current_cohorts, original_cohorts, initial_cohorts = ({'existing': fourth_escape},
                {'existing': original_fourth_escape}, {'existing': state.get('initial_fourth_escape')})
            native_registered = config.get('enforce_native_terminal_escape_guard', False)
            if native_registered:
                current_cohorts['native_fourth'] = native_fourth_escape
                original_cohorts['native_fourth'] = original_native_fourth_escape
                initial_cohorts['native_fourth'] = state.get('initial_native_fourth_escape')
            elif native_fourth_escape or original_native_fourth_escape:
                raise ValueError('Native terminal windows cannot enter the old aggregate escape guard')
            cohort_passed, cohort_audits = fourth_cohort_floor_guards(step, current_cohorts,
                original_cohorts, initial_cohorts, native_registered)
            escape_guard, escape_audit = cohort_passed['existing'], cohort_audits['existing']
            native_escape_guard = cohort_passed.get('native_fourth', True)
            native_escape_audit = cohort_audits.get('native_fourth')
            third_guard, third_audit = terminal_floor_guard(step, third_pursuit,
                original_third_pursuit, state.get("initial_third_pursuit"))
            if step == 0:
                state["initial_fourth_escape"] = escape_audit["candidate"]
                if native_escape_audit is not None:
                    state['initial_native_fourth_escape'] = native_escape_audit['candidate']
                state["initial_third_pursuit"] = third_audit["candidate"]
        exposure_snapshot = exposure.snapshot()
        exposure_guard = step == 0 or exposure_snapshot["minimum_exposure_met"]
        guard_passed = (other_guards_passed and win_guard and exposure_guard and escape_guard
            and third_guard and native_guard and leading_guard and native_escape_guard)
        if step == 0:
            raw_floors = all(audit is None or audit['candidate'] >= audit['required_nonzero_floor']-1e-12
                             for audit in (escape_audit, third_audit, native_escape_audit))
            raw_native = native_kl is None or native_kl <= config['native_reference_kl_guard']
            raw_leading=leading_kl is None or leading_kl<=config['leading_native_reference_kl_guard']
            state['zero_feasible'] = other_guards_passed and win_guard and raw_floors and raw_native and raw_leading
        state["evaluation_step"] = step
        state["other_guards_passed"] = other_guards_passed
        state["exposure_met"] = exposure_snapshot["minimum_exposure_met"]
        selection_metric = selection_value(step,utility,replay_ce,guard_passed,state['zero_feasible'],
                                           config.get('feasibility_first_selection',False))
        item = {"step": step, "endgame_validation_terminal_utility": utility,
            "utility_by_stratum": strata,
            "replay_target_ce": replay_ce, "replay_guard_passed": replay_guard,
            "placement_strata_guard_passed": placement_guard, "all_guards_passed": guard_passed,
            "replay_reference_kl": mean_reference_kl, "reference_guard_passed": reference_guard,
            "reference_kl_guard_limit": reference_limit,
            "native_reference_kl": native_kl, "native_reference_guard_passed": native_guard,
            "native_reference_kl_limit": config.get("native_reference_kl_guard"),
            'leading_native_reference_kl':leading_kl,'leading_native_reference_guard_passed':leading_guard,
            'leading_native_reference_kl_limit':config.get('leading_native_reference_kl_guard'),
            "zero_anchor_raw_constraints_met": state['zero_feasible'],
            "feasibility_first_selection": config.get('feasibility_first_selection',False),
            "training_exposure": exposure_snapshot, "exposure_guard_passed": exposure_guard,
            "win_stratum_guard_passed": win_guard,
            "fourth_escape_guard_passed": escape_guard, "fourth_escape_validation": escape_audit,
            "native_fourth_escape_guard_passed": native_escape_guard,
            "native_fourth_escape_validation": native_escape_audit,
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
        if parameter_scope=='head_only':args[0].lm.eval()
        state["micro_batches"] = state.get("micro_batches", 0) + 1
        steps = (state["micro_batches"] - 1) // 16
        if steps >= 2 and not state.get("update_proof"):
            model = args[0]
            actual = get_peft_model_state_dict(model.lm, save_embedding_layers=False)
            changed_adapter = any(not torch.equal(v.detach().cpu(), state["adapter_initial"][k]) for k, v in actual.items())
            changed_head = any(not torch.equal(v.detach().cpu(), state["head_initial"][k]) for k, v in model.head.state_dict().items())
            validate_parameter_updates(parameter_scope,changed_adapter,changed_head)
            state["update_proof"] = True
            write(out / "actual-update-proof-private.json", {
                "completed_optimizer_steps_at_least": steps, "adapter_changed": changed_adapter,
                "head_changed": changed_head,'trainable_scope':parameter_scope})
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
        "--device", "cuda", "--weights_dtype", "bf16", "--dtype", compute['dtype'],
        "--lora", "16", "--lora_targets", "all", "--lora_dropout", "0.05",
        "--head_dim", "256", "--decision_mode", "pointer", "--branch_mode", "rows",
        "--backbone_adapter", "text", "--epochs", str(proof["epochs"]), "--max_steps", str(proof["max_steps"]),
        "--lr", str(lr), "--head_lr", str(head_lr), "--weight_decay", "0.001",
        "--batch", "1", "--accum", "16", "--checkpointing", str(compute['checkpointing']),
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
        if compute['disable_frozen_encoder_input_grads'] and not state.get('readout_proof'):
            raise ValueError('No real training readout was audited')
        if parameter_scope=='head_only':
            current=get_peft_model_state_dict(state['model'].lm,save_embedding_layers=False)
            if (set(current)!=set(state['adapter_initial']) or any(
                    not torch.equal(value.detach().cpu(),state['adapter_initial'][name])
                    for name,value in current.items())):
                raise ValueError('Frozen LoRA changed by the end of head-only fitting')
        for file, expected_digest in expected.items():
            if checksum(file) != expected_digest:
                raise ValueError("Registered training data changed")
        selection = json.loads((train_dir / "selection.json").read_text())
        result = {"status": "endgame_pilot_training_complete",
            "selection": selection, "actual_updates_verified": bool(state.get("update_proof")),
            "eligible_for_development": selection["step"] > 0,
            "validation": curve, "deployed": False,'trainable_scope':parameter_scope,
            'parameter_scope':state.get('parameter_scope'),
            'frozen_encoder_preserved':parameter_scope=='head_only',
            'compute':compute,'readout_proof':state.get('readout_proof')}
        write(out / "fit-result-private.json", result)
        write(out / "training-status-private.json", result)
        return result
    finally:
        if state.get('readout_hook') is not None:state['readout_hook'].remove()
        trainer.question_loss, trainer.evaluate_during_training = original_loss, original_eval
        trainer.batch_loss, trainer.Checkpoint.warm_start = original_batch, original_warm
        trainer.training_requests = original_admit
        trainer.encode_batch, trainer.early_stop_update = original_encode, original_stop

