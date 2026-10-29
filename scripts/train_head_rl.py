"""Bounded head-only RL pilot, with complete-game gates and explicit stop analysis."""
import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from head_rl import HeadPolicy, PPO, deployment_greedy_index, development_gate, fingerprint, mixed_probabilities, paired_primary_comparison, plateau_checks, primary_metric, save_checkpoint, summarize, terminal_reward, write


class PilotDeadline(RuntimeError):
    pass


def evaluate(policy, directory, config, deadline, frozen=False):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    path = directory/"games.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    keys = {(r["seed"], r["seat"]) for r in rows}
    expected = {(s, seat) for s in range(config["development_seed_start"],
                                         config["development_seed_start"]+config["development_seeds"]) for seat in range(4)}
    if len(keys) != len(rows) or not keys <= expected:
        raise ValueError("Development game coverage differs from frozen protocol")
    if any(r.get("game_mode", "4p-red-half") != config.get("game_mode", "4p-red-half")
           or r.get("reward_mode", "first") != config.get("reward_mode", "first") for r in rows):
        raise ValueError("Development match format or reward differs from frozen protocol")
    with path.open("a") as output:
        for seed, seat in sorted(expected-keys):
            if time.monotonic() >= deadline:
                raise PilotDeadline("Pilot wall-time limit reached")
            row, _ = policy.game(seed, seat, frozen=frozen)
            rows.append(row)
            output.write(json.dumps(row)+"\n"); output.flush()
            print(json.dumps({"phase": directory.name, "games": len(rows),
                              "planned": len(expected), "rank": row["rank"], "score": row["score"]}), flush=True)
    report = summarize(rows)
    write(directory/"summary.json", report)
    return report, rows


@torch.no_grad()
def rollout_diagnostics(policy, rows):
    flips, values = 0, []
    for row in rows:
        decide, options = row["decide"].to("cuda"), row["options"].to("cuda")
        logits, reference = policy.head(decide, options), policy.reference(decide, options)
        p = mixed_probabilities(logits, row["eligible"], policy.config["exploration_uniform"])
        flips += int(deployment_greedy_index(logits, row["eligible"]) !=
                     deployment_greedy_index(reference, row["eligible"]))
        values.append(float(-(p*p.log()).sum()))
    return {"sampled_states": len(rows), "greedy_flips_vs_reference": flips,
            "greedy_flip_fraction": flips/len(rows), "mean_entropy": sum(values)/len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"head-rl-pilot.json")
    parser.add_argument("--resume", type=Path, help="Resume a complete stage-boundary RL checkpoint")
    parser.add_argument("--seed-protocol-migration", action="store_true",
                        help="Explicitly migrate a saved stage boundary to independent wall streams")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("reward_mode", "first") not in ("first", "top2", "net_points") or config.get("game_mode", "4p-red-half") not in ("4p-red-east", "4p-red-half"):
        raise ValueError("Unsupported reward or match format")
    terminal_reward(1, config.get("reward_mode", "first"), score=25000, scale=config.get("reward_scale", 1000.0))
    if config.get("seed_scheme") != "spaced-v1":
        raise ValueError("Raw contiguous episode seeds are retired; use the reviewed spaced-v1 protocol")
    out = ROOT/config["out"]
    config_hash = __import__('hashlib').sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (out/"status.json").exists():
        existing = json.loads((out/"status.json").read_text()).get("status", "")
        if existing in {"stopped", "awaiting_confirmation", "confirmed", "stopped_confirmation_not_qualified"}:
            raise ValueError("Run is terminal; preserve its frozen evidence and create a new experiment")
    if out.exists() and not args.resume:
        raise ValueError("Run already exists; never overwrite an experiment")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text())["config"] != config:
        raise ValueError("Cannot change a resumed pilot protocol")
    write(out/"protocol.json", {"config": config, "sha256": config_hash,
                              "code_tree_sha": subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip(),
                              "code_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in
                                              [ROOT/'head_rl.py', Path(__file__), ROOT/'arena.py', ROOT/'scripts/confirm_head_rl.py']},
                              "reference_adapter_sha256": fingerprint(ROOT/config["reference"]/"adapter_model.safetensors"),
                              "reference_head_sha256": fingerprint(ROOT/config["reference"]/"head.pt")})
    random.seed(config["seed"]); torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
    deadline = time.monotonic()+config["pilot_wall_minutes"]*60
    write(out/"status.json", {"status": "loading_model", "updates": 0})
    policy = HeadPolicy(config, args.resume)
    trainer = PPO(policy)
    update, start_stage, pending_stage = 0, 1, None
    if args.resume:
        state = torch.load(args.resume/"rl_state.pt", map_location="cpu", weights_only=False)
        if state["config_hash"] != config_hash and args.seed_protocol_migration:
            previous = json.loads((args.resume.parent.parent/"protocol.json").read_text())
            allowed = {"out", "seed_scheme", "development_seed_start", "confirmation_seed_start"}
            if (previous["sha256"] != state["config_hash"]
                    or previous["config"].get("seed_scheme", "raw") != "raw"
                    or any(previous["config"].get(k) != config.get(k) for k in set(previous["config"]) | set(config) if k not in allowed)
                    or config.get("seed_scheme") != "spaced-v1"):
                raise ValueError("Only the reviewed seed-isolation migration may change a resume protocol")
            write(out/"seed-migration.json", {"source": str(args.resume), "source_protocol": previous,
                                             "reason": "Raw adjacent seeds share shifted hand-wall streams",
                                             "preserved_update_budget": config["max_stages"]*config["updates_per_stage"],
                                             "inherited_updates": state["update"]})
            state["config_hash"] = config_hash
        if state["config_hash"] != config_hash or state["stage_update"] not in (0, config["updates_per_stage"]):
            raise ValueError("Resume requires an unchanged protocol and complete stage boundary")
        update = state["update"]
        policy.critic.load_state_dict(state["critic"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"]); torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"]); policy.generator.set_state(state["sampling_rng"])
        if update:
            pending_stage = state["stage"]
            start_stage = pending_stage
        if args.seed_protocol_migration:
            save_checkpoint(policy, trainer, out, update, state["stage"], state["stage_update"], config_hash)
    actor_parameters = sum(p.numel() for p in policy.head.parameters())
    print(json.dumps({"phase": "initialized", "actor_trainable_parameters": actor_parameters,
                      "backbone_trainable_parameters": sum(p.numel() for p in policy.model.lm.parameters() if p.requires_grad),
                      "reward": config["reward"], "max_updates": config["max_stages"]*config["updates_per_stage"]}), flush=True)
    checkpoints = out/"checkpoints"
    checkpoints.mkdir(exist_ok=True)
    results = json.loads((out/"development-results.json").read_text()) if (out/"development-results.json").exists() else []
    baseline, baseline_rows = None, []
    metric = primary_metric(config)
    reason, last_rollout = "stop_target_not_reached", None

    def record_stage(stage):
        nonlocal reason
        write(out/"status.json", {"status": "development_validation", "stage": stage, "updates": update})
        report, rows = evaluate(policy, out/f"development-update-{update:06d}", config, deadline)
        decision, changes = development_gate(stage, report, baseline, config)
        item = {"stage": stage, "update": update, "metrics": report, "decision": decision,
                "changes": changes, "paired_exploratory": paired_primary_comparison(rows, baseline_rows, config),
                "checkpoint": str((checkpoints/f"update-{update:06d}").relative_to(ROOT))}
        if config.get("plateau_patience") and decision == "continue":
            history = [r["metrics"][metric] for r in results if r["update"] < update]
            stale = plateau_checks(baseline[metric], history, report[metric])
            if stale >= config["plateau_patience"]:
                item["decision"] = decision = "stop_plateau"
                changes["consecutive_checks_without_new_best"] = stale
        results[:] = [r for r in results if r["update"] != update] + [item]
        write(out/"development-results.json", results)
        print("development_gate "+json.dumps(item), flush=True)
        reason = decision
        return decision

    try:
        if not update:
            save_checkpoint(policy, trainer, out, 0, 0, 0, config_hash)
        write(out/"status.json", {"status": "baseline_validation", "updates": update})
        baseline, baseline_rows = evaluate(policy, out/"development-baseline", config, deadline, frozen=True)
        write(out/"baseline.json", baseline)
        print("baseline "+json.dumps(baseline), flush=True)
        for stage in range(start_stage, config["max_stages"]+1):
            if pending_stage == stage:
                pending_stage = None
                if record_stage(stage) != "continue":
                    break
                continue
            rows, games = [], []
            write(out/"status.json", {"status": "collecting_rollouts", "stage": stage, "updates": update})
            training_start = config["training_seed_start"]+(stage-1)*config["training_seeds_per_stage"]
            for seed in range(training_start, training_start+config["training_seeds_per_stage"]):
                for seat in range(4):
                    if time.monotonic() >= deadline:
                        raise PilotDeadline("Pilot wall-time limit reached")
                    result, trace = policy.game(seed, seat, training=True, old_opponent=(seed % 2 == 1))
                    games.append(result); rows.extend(trace)
                    print(json.dumps({"phase": "rollout", "stage": stage, "games": len(games),
                                      "rank": result["rank"], "reward": result["reward"], "states": len(trace)}), flush=True)
            write(out/f"training-games-stage-{stage:02d}.json", games)
            torch.save(rows, out/f"rollout-stage-{stage:02d}.pt")
            if not rows or len({g["reward"] for g in games}) < 2 or len({r["return"] for r in rows}) < 2:
                reason = "stop_no_reward_variation"
                break
            advantages = torch.tensor([r["return"]-r["old_value"] for r in rows])
            advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
            for row, advantage in zip(rows, advantages.tolist()):
                row["advantage"] = advantage
            stage_state = copy.deepcopy(policy.head.state_dict())
            critic_state = copy.deepcopy(policy.critic.state_dict())
            optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
            reason = "continue"
            order, cursor = list(range(len(rows))), 0
            random.shuffle(order)
            for local_update in range(1, config["updates_per_stage"]+1):
                if time.monotonic() >= deadline:
                    raise PilotDeadline("Pilot wall-time limit reached")
                if cursor >= len(order):
                    random.shuffle(order); cursor = 0
                indices = order[cursor:cursor+config["minibatch"]]
                cursor += len(indices)
                losses = trainer.update([rows[i] for i in indices])
                update += 1
                print("ppo_update "+json.dumps({"update": update, "stage": stage, **losses}), flush=True)
                if update % config["checkpoint_every"] == 0:
                    kl = trainer.reference_kl(rows)
                    if kl > config["max_reference_kl"]:
                        reason = "stop_reference_kl_limit"
                        policy.head.load_state_dict(stage_state)
                        policy.critic.load_state_dict(critic_state)
                        trainer.optimizer.load_state_dict(optimizer_state)
                        write(out/"kl-stop.json", {"update": update, "reference_kl": kl,
                                                  "limit": config["max_reference_kl"], "rolled_back_stage": stage})
                        break
                    save_checkpoint(policy, trainer, out, update, stage, local_update, config_hash)
                    write(out/"status.json", {"status": "training", "updates": update,
                                              "stage": stage, "reference_kl": kl})
            last_rollout = {"stage": stage, "games": summarize(games),
                            "diagnostics": rollout_diagnostics(policy, rows)}
            write(out/f"rollout-diagnostics-stage-{stage:02d}.json", last_rollout)
            if reason != "continue":
                break
            if record_stage(stage) != "continue":
                break
    except PilotDeadline:
        reason = "stop_pilot_wall_limit"
    except Exception as error:
        write(out/"failure.json", {"type": type(error).__name__, "message": str(error), "updates": update})
        write(out/"status.json", {"status": "failed", "updates": update})
        raise

    # Never promote from repeatedly inspected development results.
    passed = [r for r in results if r["decision"] == "target_reached"]
    selected = max(passed, key=lambda r: (r["metrics"][metric], r["metrics"]["mean_net_points"])) if passed else None
    reference = ROOT/config["reference"]
    protocol = json.loads((out/"protocol.json").read_text())
    if fingerprint(reference/"adapter_model.safetensors") != protocol["reference_adapter_sha256"] or fingerprint(reference/"head.pt") != protocol["reference_head_sha256"]:
        raise ValueError("Original reference weights changed")
    analysis = {"status": "pilot_complete", "stop_reason": reason, "updates": update,
                "baseline": baseline, "development": results, "last_rollout": last_rollout,
                "primary_metric": metric, "game_mode": config.get("game_mode", "4p-red-half"),
                "selected_for_confirmation": selected, "default_policy_changed": False,
                "inference_feature_checks_passed": policy.checks,
                "actor_trainable_parameters": actor_parameters,
                "backbone_trainable_parameters": 0,
                "interpretation": "Repeated 64-game development checks are exploratory; thresholds stop wasted training, "
                                  "not a claim of statistical improvement. Frozen reference remains available."}
    write(out/"pilot-analysis.json", analysis)
    lines = ["# 评分头强化学习阶段结果", "", f"停止原因：`{reason}`；完成 {update} 次小批更新。", "",
             f"对局格式：{config.get('game_mode', '4p-red-half')}；奖励：{config['reward']}；底座和LoRA未更新。默认策略未改变。", "",
             "| 检查点 | 前二率 | 一位率 | 平均净得点 | 末位率 | 阶段结论 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    if baseline:
        lines.append(f"| 旧模型 | {baseline['top2_rate']:.2%} | {baseline['first_rate']:.2%} | {baseline['mean_net_points']:.1f} | {baseline['fourth_rate']:.2%} | 参考 |")
    for r in results:
        m = r["metrics"]
        lines.append(f"| 更新{r['update']} | {m['top2_rate']:.2%} | {m['first_rate']:.2%} | {m['mean_net_points']:.1f} | {m['fourth_rate']:.2%} | {r['decision']} |")
    lines += ["", "若目标未达到，停止此配方，保留全部检查点和轨迹。结合策略KL、实际动作翻转比例、训练回报变化和独立开发对局分析，不能把训练损失下降当作胜率提升。"]
    if last_rollout:
        d = last_rollout["diagnostics"]
        lines += ["", f"最近训练轨迹中，贪心动作相对旧策略改变 {d['greedy_flips_vs_reference']}/{d['sampled_states']} 次。"]
        if d["greedy_flip_fraction"] < .01:
            lines.append("观察到的部署动作变化很少，短程终局奖励尚未形成足够明显的策略改进；不能用延长同一训练替代信用分配和奖励方差分析。")
    if selected:
        lines += ["", "开发门槛达到，冻结该检查点，进入新牌山确认；开发数据不作为最终收益证明。"]
        write(out/"confirmation-selection.json", {"checkpoint": selected["checkpoint"],
                                                   "development_update": selected["update"], "config": config})
    (out/"analysis.md").write_text("\n".join(lines)+"\n")
    write(out/"status.json", {"status": "awaiting_confirmation" if selected else "stopped",
                              "stop_reason": reason, "updates": update})
    print("pilot_result "+json.dumps({"stop_reason": reason, "updates": update,
                                      "confirmation_required": bool(selected)}), flush=True)


if __name__ == "__main__":
    main()
