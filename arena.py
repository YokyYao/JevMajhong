"""Fixed-wall, seat-rotated evaluation against a conservative efficiency agent."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from riichienv import ActionType, GameRule, RiichiEnv
from semantic import build_request, features, describe, heuristic_index, legal_candidates
from seed_control import engine_seed


class EfficiencyAgent:
    def act(self, obs):
        actions = legal_candidates(obs)
        if len(actions) == 1:
            return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO):
                return action
        _, visible, d = describe(obs)
        feats = features(obs, actions, visible, d)
        return actions[heuristic_index(obs, actions, feats)]


def guarded_indices(obs, actions, feats):
    """The common admissible action set for deployment and head-only RL."""
    eligible = list(range(len(actions)))
    if any(feats):
        ref = heuristic_index(obs, actions, feats)
        if feats[ref]:
            d = obs.to_dict()
            threats = {p for p in range(4) if p != obs.player_id and d["riichi_declared"][p]}
            fold = bool(threats) and min(f["sh"] for f in feats if f) >= 1
            ref_risk = len(threats - set(feats[ref]["safe"]))
            eligible = [i for i, f in enumerate(feats) if f and f["sh"] <= feats[ref]["sh"]
                        and (not fold or len(threats - set(f["safe"])) <= ref_risk)]
    if not eligible:
        raise ValueError("Guarded action set is empty")
    return eligible


class ModelAgent:
    def __init__(self, checkpoint, guarded=False):
        from jevany import JevModel
        from jevany.inference import InferenceOptions
        self.model = JevModel.from_pretrained(checkpoint, device="cuda", dtype="bf16",
            inference_options=InferenceOptions(max_state_tokens=4096,
                                              max_branch_tokens=8192, max_packed_tokens=8192))
        self.guarded = guarded
        self.calls = 0

    def act(self, obs):
        from jevany.api import validate_distribution
        actions = legal_candidates(obs)
        if len(actions) == 1:
            return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO):
                return action
        request, actions, feats = build_request(obs)
        result = self.model.system_one(**request)
        keys = list(request["questions"]["action"]["criteria"])
        probabilities, _ = validate_distribution(result["answers"]["action"]["probabilities"], keys)
        self.calls += 1
        eligible = guarded_indices(obs, actions, feats) if self.guarded else list(range(len(actions)))
        return actions[max(eligible, key=lambda i: probabilities[i])]


def play(agent, seat, seed, seed_scheme="spaced-v1"):
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul(), seed=engine_seed(seed, seed_scheme))
    baseline = EfficiencyAgent()
    obs = env.reset()
    steps = 0
    while not env.done():
        selected = {p: (agent if p == seat else baseline).act(o) for p, o in obs.items()}
        obs = env.step(selected)
        steps += 1
        if steps > 100000:
            raise RuntimeError("Arena game did not terminate")
    log = env.mjai_log
    starts = sum(e["type"] == "start_kyoku" for e in log)
    wins = sum(e["type"] == "hora" and e.get("actor") == seat for e in log)
    dealins = sum(e["type"] == "hora" and e.get("target") == seat and e.get("actor") != seat for e in log)
    return {"seed": seed, "engine_seed": engine_seed(seed, seed_scheme), "seed_scheme": seed_scheme,
            "seat": seat, "rank": env.ranks()[seat],
            "score": env.scores()[seat], "hands": starts, "hand_wins": wins,
            "dealins": dealins, "steps": steps}


def ci(values, confidence=0.95):
    if len(values) < 2:
        return None
    from scipy.stats import t
    mean = statistics.mean(values)
    half = float(t.ppf((1+confidence)/2, len(values)-1)) * statistics.stdev(values) / math.sqrt(len(values))
    return [mean-half, mean+half]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=510000)
    parser.add_argument("--seed-scheme", choices=("spaced-v1", "raw"), default="spaced-v1",
                        help="spaced-v1 isolates wall streams; raw only reproduces legacy results")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    config = {"checkpoint": args.checkpoint, "guarded": args.guarded,
              "seeds": args.seeds, "seed_start": args.seed_start, "seed_scheme": args.seed_scheme}
    config_path = args.out / "arena_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        saved.setdefault("seed_scheme", "raw")
        if saved != config:
            raise ValueError("Arena resume configuration differs from original run")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    if args.resume and (args.out / "summary.json").exists():
        print((args.out / "summary.json").read_text())
        return
    candidate = ModelAgent(args.checkpoint, args.guarded) if args.checkpoint else EfficiencyAgent()
    rows_path = args.out / "games.jsonl"
    rows = ([json.loads(line) for line in rows_path.read_text().splitlines()]
            if args.resume and rows_path.exists() else [])
    completed = {(x["candidate"]["seed"], x["candidate"]["seat"]) for x in rows}
    started = time.monotonic()
    with rows_path.open("a" if args.resume else "w") as stream:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for seat in range(4):
                if (seed, seat) in completed:
                    continue
                calls_before = getattr(candidate, "calls", 0)
                base = play(EfficiencyAgent(), seat, seed, args.seed_scheme)
                result = play(candidate, seat, seed, args.seed_scheme) if args.checkpoint else base.copy()
                row = {"baseline": base, "candidate": result,
                       "model_calls": getattr(candidate, "calls", 0) - calls_before}
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps({"games": len(rows), "seed": seed, "seat": seat,
                                  "baseline_rank": base["rank"], "candidate_rank": result["rank"]}), flush=True)
    candidate_rows = [x["candidate"] for x in rows]
    ranks = [x["rank"] for x in candidate_rows]
    deltas = [statistics.mean(x["candidate"]["rank"] - x["baseline"]["rank"]
                             for x in rows if x["candidate"]["seed"] == seed)
              for seed in range(args.seed_start, args.seed_start + args.seeds)]
    report = {"games": len(rows), "independent_seeds": args.seeds if args.seed_scheme == "spaced-v1" else None,
        "nominal_seed_groups": args.seeds, "seed_scheme": args.seed_scheme,
        "mean_rank": statistics.mean(ranks), "first_rate": ranks.count(1)/len(ranks),
        "fourth_rate": ranks.count(4)/len(ranks),
        "hand_win_rate": sum(x["hand_wins"] for x in candidate_rows)/sum(x["hands"] for x in candidate_rows),
        "dealin_rate": sum(x["dealins"] for x in candidate_rows)/sum(x["hands"] for x in candidate_rows),
        "paired_rank_delta": statistics.mean(deltas),
        "paired_rank_delta_ci95": ci(deltas) if args.seed_scheme == "spaced-v1" else None,
        "checkpoint": args.checkpoint, "guarded": args.guarded,
        "model_calls": sum(x.get("model_calls", 0) for x in rows),
        "wall_seconds_this_invocation": time.monotonic()-started,
        "note": "Negative rank delta is better; few seeds are a smoke test, not strength evidence."}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
