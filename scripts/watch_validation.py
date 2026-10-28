"""Evaluate immutable saved checkpoints during an existing Slurm training allocation."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--start", type=int, default=600)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--per-match", type=int, default=4)
    args = parser.parse_args()
    out = args.run/"periodic-validation"
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for line in args.validation.read_text().splitlines():
        record = json.loads(line)
        if record["_meta"]["split"] != "validation":
            raise ValueError("Only the validation split can be used here")
        groups[record["_meta"]["group_id"]].append(record)
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: hashlib.sha256(r["_meta"]["id"].encode()).hexdigest())
        selected.extend(group[:args.per_match])
    subset = out/"fixed-subset.jsonl"
    content = "".join(json.dumps(r, separators=(",", ":"))+"\n" for r in selected)
    if subset.exists() and subset.read_text() != content:
        raise ValueError("Validation subset changed")
    subset.write_text(content)
    print(json.dumps({"validation_records": len(selected), "matches":len(groups),
                      "every_steps":args.every}), flush=True)
    env = {**os.environ, "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2"}
    while True:
        checkpoints = sorted((args.run/"checkpoints").glob("step-*"))
        for checkpoint in checkpoints:
            step = int(checkpoint.name.split("-")[-1])
            report = out/f"step-{step:06d}.json"
            if step < args.start or step % args.every or report.exists():
                continue
            required = ["trainer_state.pt", "trainer_state.json", "adapter_model.safetensors", "head.pt"]
            if not all((checkpoint/name).exists() for name in required):
                continue
            # Saving completes before the trainer advances the log's checkpoint marker.
            if time.time() - max((checkpoint/name).stat().st_mtime for name in required) < 20:
                continue
            subprocess.run([sys.executable, "evaluate.py", str(checkpoint), str(subset),
                            "--out", str(report)], env=env, check=True)
            data = json.loads(report.read_text())
            result = {"step":step, "n":data["n"], "accuracy":data["accuracy"], "nll":data["nll"]}
            print(json.dumps(result), flush=True)
        reports = sorted(out.glob("step-*.json"))
        curve = [{"step":int(p.stem.split("-")[-1]),
                  **{k:v for k,v in json.loads(p.read_text()).items() if k in ("n","accuracy","nll")}}
                 for p in reports]
        (out/"curve.json").write_text(json.dumps(curve, indent=2))
        if curve:
            (out/"best-validation.json").write_text(json.dumps(min(curve, key=lambda r:r["nll"]), indent=2))
        if (args.run/"training_metrics.json").exists() and reports and curve[-1]["step"] == 2000:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
