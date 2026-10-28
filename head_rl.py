"""Small on-policy pointer-head PPO using terminal hanchan first-place rewards."""
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from riichienv import ActionType, GameRule, RiichiEnv
from arena import EfficiencyAgent, guarded_indices
from semantic import build_request, legal_candidates


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def fingerprint(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def mixed_probabilities(logits, eligible, epsilon):
    if not 0 <= epsilon < 1 or len(eligible) < 1:
        raise ValueError("Invalid exploration or empty action set")
    probabilities = torch.softmax(logits[eligible].float(), -1)
    return (1-epsilon)*probabilities + epsilon/len(eligible)


def terminal_reward(rank):
    if rank not in (1, 2, 3, 4):
        raise ValueError("Invalid final rank")
    return float(rank == 1)


def deployment_greedy_index(logits, eligible):
    # Match ModelAgent: choose from full probabilities AFTER applying the guard.
    # Renormalizing masked logits first can change full-softmax underflow ties.
    full = torch.softmax(logits.float(), -1)
    return max(range(len(eligible)), key=lambda i: float(full[eligible[i]]))


def ppo_surrogate(logp, old_logp, advantage, clip):
    ratio = (logp-old_logp).exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)


def summarize(rows):
    if not rows:
        raise ValueError("No complete games")
    n = len(rows)
    return {"games": n, "independent_seeds": len({r["seed"] for r in rows}),
            "firsts": sum(r["rank"] == 1 for r in rows),
            "first_rate": sum(r["rank"] == 1 for r in rows)/n,
            "mean_rank": sum(r["rank"] for r in rows)/n,
            "fourth_rate": sum(r["rank"] == 4 for r in rows)/n,
            "mean_final_points": sum(r["score"] for r in rows)/n,
            "mean_net_points": sum(r["score"]-25000 for r in rows)/n}


def development_gate(stage, candidate, baseline, config):
    first = candidate["first_rate"] - baseline["first_rate"]
    points = candidate["mean_net_points"] - baseline["mean_net_points"]
    fourth = candidate["fourth_rate"] - baseline["fourth_rate"]
    details = {"first_gain": first, "net_points_gain": points, "fourth_increase": fourth,
               "required_first_rate": max(config["target_first_rate"],
                                          baseline["first_rate"]+config["target_first_gain"])}
    if first <= -config["first_drop_stop"] or points <= -config["net_points_drop_stop"] or fourth >= config["fourth_increase_stop"]:
        return "stop_regression", details
    reached = (candidate["first_rate"] >= details["required_first_rate"] and points >= 0
               and fourth <= config["target_fourth_tolerance"])
    if reached:
        return "target_reached", details
    if stage >= 2 and first < config["stage_two_min_first_gain"]:
        return "stop_no_early_progress", details
    if stage >= config["max_stages"]:
        return "stop_target_not_reached", details
    return "continue", details


class HeadPolicy:
    def __init__(self, config, checkpoint=None):
        from jevany import JevModel
        from jevany.checkpoint import Checkpoint, LoadOptions
        from jevany.inference import InferenceOptions
        self.config = config
        self.client = JevModel.from_pretrained(checkpoint or config["reference"], device="cuda", dtype="bf16",
            options=LoadOptions(merge=False, temperature=1.0),
            inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                                               max_packed_tokens=8192))
        self.model = self.client.runtime.model
        self.model.requires_grad_(False)
        self.head = self.model.head
        self.head.requires_grad_(True)
        self.head.eval()
        self.reference = copy.deepcopy(self.head).requires_grad_(False).eval()
        self.reference.load_state_dict(Checkpoint(config["reference"]).meta.head)
        self.reference.temperature = 1.0
        self.reference_meta = Checkpoint(config["reference"]).meta
        dimension = self.head.q.in_features
        self.critic = nn.Linear(dimension, 1).to("cuda")
        nn.init.zeros_(self.critic.weight); nn.init.zeros_(self.critic.bias)
        self.capture = None
        self.head.register_forward_pre_hook(self._capture)
        self.checks = 0
        self.generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        if any(p.requires_grad for p in self.model.lm.parameters()):
            raise ValueError("Backbone and LoRA must be frozen")

    def _capture(self, module, inputs):
        self.capture = tuple(t.detach() for t in inputs)

    def act(self, obs, frozen=False, stochastic=False, trace=None):
        actions = legal_candidates(obs)
        if len(actions) == 1:
            return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO):
                return action
        request, actions, feats = build_request(obs)
        self.capture = None
        response = self.client.system_one(**request)
        if self.capture is None:
            raise ValueError("The native inference path did not expose pointer features")
        decide, options = self.capture
        self.capture = None
        if len(options) != len(actions):
            raise ValueError("Feature/action alignment mismatch")
        eligible = guarded_indices(obs, actions, feats)
        with torch.no_grad():
            # Clone outside the runtime call so replay features are ordinary detached tensors.
            decide, options = decide.clone(), options.clone()
            logits = self.head(decide, options)
            if self.checks < 32:
                from jevany.api import validate_distribution
                ordered, _ = validate_distribution(response["answers"]["action"]["probabilities"],
                                                    list(request["questions"]["action"]["criteria"]))
                native = torch.tensor(ordered, device="cuda")
                replayed = torch.softmax(logits, -1)
                if not torch.allclose(native, replayed, atol=2e-5, rtol=2e-4):
                    raise ValueError("Cached head features do not reproduce deployed probabilities")
                expected = max(eligible, key=lambda i: float(native[i]))
                actual = max(eligible, key=lambda i: float(replayed[i]))
                if expected != actual:
                    raise ValueError("Deployed guarded action differs from feature replay")
                self.checks += 1
            if frozen:
                logits = self.reference(decide, options)
            epsilon = self.config["exploration_uniform"] if stochastic else 0.0
            probabilities = mixed_probabilities(logits, eligible, epsilon)
            if stochastic:
                index = int(torch.multinomial(probabilities.cpu(), 1, generator=self.generator))
            else:
                index = deployment_greedy_index(logits, eligible)
            if trace is not None and len(eligible) > 1:
                reference = mixed_probabilities(self.reference(decide, options), eligible, epsilon)
                value = float(torch.sigmoid(self.critic(decide)).squeeze())
                trace.append({"decide": decide.cpu(), "options": options.cpu(), "eligible": eligible,
                              "action": index, "old_logp": float(probabilities[index].log()),
                              "reference": reference.cpu(), "old_value": value})
        return actions[eligible[index]]

    def game(self, seed, seat, training=False, frozen=False, old_opponent=False):
        env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul(), seed=seed)
        observations = env.reset()
        baseline, trace, steps = EfficiencyAgent(), [], 0
        while not env.done():
            selected = {}
            for player, observation in observations.items():
                if player == seat:
                    selected[player] = self.act(observation, frozen=frozen, stochastic=training,
                                                trace=trace if training else None)
                elif old_opponent and player == (seat+1) % 4:
                    selected[player] = self.act(observation, frozen=True)
                else:
                    selected[player] = baseline.act(observation)
            observations = env.step(selected)
            steps += 1
            if steps > 100000:
                raise RuntimeError("Game failed to terminate")
        row = {"seed": seed, "seat": seat, "rank": int(env.ranks()[seat]),
               "score": int(env.scores()[seat]), "environment_steps": steps,
               "reward": terminal_reward(int(env.ranks()[seat])), "trainable_decisions": len(trace),
               "old_opponent": old_opponent}
        for item in trace:
            item["return"] = float(row["reward"])
        return row, trace


class PPO:
    def __init__(self, policy):
        self.policy, self.config = policy, policy.config
        self.optimizer = torch.optim.AdamW([
            {"params": policy.head.parameters(), "lr": self.config["actor_lr"]},
            {"params": policy.critic.parameters(), "lr": self.config["critic_lr"]}], weight_decay=0.0)

    def update(self, batch):
        head, critic, config = self.policy.head, self.policy.critic, self.config
        losses, policy_losses, values, kls, entropies = [], [], [], [], []
        advantages = torch.tensor([r["advantage"] for r in batch], device="cuda")
        for index, row in enumerate(batch):
            decide, options = row["decide"].to("cuda"), row["options"].to("cuda")
            probability = mixed_probabilities(head(decide, options), row["eligible"], config["exploration_uniform"])
            logp = probability[row["action"]].log()
            advantage = advantages[index]
            actor = ppo_surrogate(logp, row["old_logp"], advantage, config["clip_ratio"])
            reference = row["reference"].to("cuda")
            kl = (probability*(probability.log()-reference.log())).sum()
            entropy = -(probability*probability.log()).sum()
            value = torch.sigmoid(critic(decide)).squeeze()
            value_loss = .5*(value-row["return"]).square()
            losses.append(actor + config["reference_kl_beta"]*kl - config["entropy_beta"]*entropy + value_loss)
            policy_losses.append(actor.detach()); values.append(value_loss.detach())
            kls.append(kl.detach()); entropies.append(entropy.detach())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite PPO objective")
        self.optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(head.parameters(), .5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite head gradient")
        self.optimizer.step()
        if any(p.grad is not None for p in self.policy.model.lm.parameters()):
            raise ValueError("Frozen backbone received a gradient")
        return {"loss": float(loss.detach()), "policy_loss": float(torch.stack(policy_losses).mean()),
                "value_loss": float(torch.stack(values).mean()), "reference_kl_batch": float(torch.stack(kls).mean()),
                "entropy": float(torch.stack(entropies).mean()), "head_grad_norm": float(grad)}

    @torch.no_grad()
    def reference_kl(self, rows):
        values = []
        for row in rows:
            p = mixed_probabilities(self.policy.head(row["decide"].to("cuda"), row["options"].to("cuda")),
                                    row["eligible"], self.config["exploration_uniform"])
            reference = row["reference"].to("cuda")
            values.append(float((p*(p.log()-reference.log())).sum()))
        return sum(values)/len(values)


def save_checkpoint(policy, optimizer, out, update, stage, stage_update, config_hash):
    path = Path(out)/"checkpoints"/f"update-{update:06d}"
    if path.exists():
        if (path/"complete.json").exists():
            return path
        path.rename(path.with_name(path.name+f"-incomplete-{int(time.time())}"))
    path.mkdir(parents=True)
    source = Path(policy.config["reference"])
    for name in ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
                 "tokenizer.json", "tokenizer_config.json"]:
        if (source/name).exists():
            try:
                os.link(source/name, path/name)
            except OSError:
                shutil.copy2(source/name, path/name)
    from jevany.checkpoint import write_meta
    meta = replace(policy.reference_meta,
                   head={k: v.detach().cpu().clone() for k, v in policy.head.state_dict().items()}, temperature=1.0,
                   extra={"args": {k: v for k, v in policy.reference_meta.extra.get("args", {}).items()
                                   if k in ("lora_targets", "lora_target_modules", "lora_dropout")},
                          "training_method": "pointer-head PPO with terminal first-place reward",
                          "rl_update": update, "parent": str(source), "protocol_sha256": config_hash,
                          "rl_config": policy.config})
    write_meta(path, meta)
    torch.save({"format": "head-ppo-v1", "update": update, "stage": stage, "stage_update": stage_update,
                "config_hash": config_hash, "critic": policy.critic.state_dict(),
                "optimizer": optimizer.optimizer.state_dict(), "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "sampling_rng": policy.generator.get_state()}, path/"rl_state.pt")
    write(path/"complete.json", {"update": update, "stage": stage, "stage_update": stage_update,
                                "head_sha256": fingerprint(path/"head.pt"), "protocol_sha256": config_hash})
    return path
