"""Explore actual placement and points for every predeclared trained checkpoint."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_round2 import paired_comparison, write


def main():
    run = ROOT / "runs/semantic-2026-round2"
    output = run / "strength-screen"
    output.mkdir(exist_ok=True)
    policies = [("previous-guarded", "runs/semantic-2026-freegpu/checkpoints/step-002000")]
    policies += [(f"step-{step:06d}-guarded", f"runs/semantic-2026-round2/checkpoints/step-{step:06d}")
                 for step in (400, 800, 1200)]
    protocol = {"purpose": "exploratory strength comparison; no automatic promotion",
                "seed_start": 820000, "independent_seeds": 32, "games_per_policy": 128,
                "guarded": True, "policies": [{"name": name, "checkpoint": checkpoint,
                    "adapter_sha256": hashlib.sha256((ROOT/checkpoint/"adapter_model.safetensors").read_bytes()).hexdigest(),
                    "head_sha256": hashlib.sha256((ROOT/checkpoint/"head.pt").read_bytes()).hexdigest()}
                    for name, checkpoint in policies],
                "primary_metric": "first_rate", "secondary_metrics": ["mean_net_points", "mean_rank", "fourth_rate"],
                "confirmation_seed_start": 910000, "confirmation_seeds": 128,
                "note": "NLL does not choose a model here. All three trained checkpoints are reported. "
                        "Screening intervals are exploratory and do not establish playing-strength gains."}
    path = output/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Cannot change policies, weights or seeds within an existing strength screen")
    write(path, protocol)
    previous = None
    results = []
    for name, checkpoint in policies:
        write(output/"status.json", {"status": "running", "current": name, "completed_policies": len(results)})
        directory = output/name
        command = [sys.executable, str(ROOT/"arena.py"), "--checkpoint", checkpoint, "--guarded",
                   "--seeds", "32", "--seed-start", "820000", "--out", str(directory)]
        if directory.exists():
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, check=True)
        rows = [json.loads(s) for s in (directory/"games.jsonl").read_text().splitlines()]
        candidate = [r["candidate"] for r in rows]
        if previous is None:
            previous = candidate
        report = json.loads((directory/"summary.json").read_text())
        report.update(name=name, mean_final_points=statistics.mean(r["score"] for r in candidate),
                      mean_net_points=statistics.mean(r["score"]-25000 for r in candidate),
                      vs_rule=paired_comparison(candidate, [r["baseline"] for r in rows]),
                      vs_previous_guarded=paired_comparison(candidate, previous))
        results.append(report)
        write(output/"results.json", {"status": "in_progress", "policies": results,
                                      "note": "Exploratory screen; no default-policy change."})
        print("strength_result " + json.dumps(report), flush=True)
    best = max(results, key=lambda r: (r["first_rate"], r["mean_net_points"], -r["mean_rank"]))
    write(output/"results.json", {"status": "complete", "policies": results,
                                 "screen_best": best["name"], "default_policy_changed": False,
                                 "note": "First rate breaks ties using mean net points then mean rank. "
                                         "A trained screen winner requires independent confirmation on fresh seeds; "
                                         "screening alone never enables a model."})
    write(output/"status.json", {"status": "complete", "completed_policies": len(results)})


if __name__ == "__main__":
    main()
