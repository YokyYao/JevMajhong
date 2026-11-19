"""Run a bounded endgame training stage on a scheduler-allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "coverage", "target-manifest", "reuse-fourth-manifest", "label", "enrich", "assemble", "prepare", "reference", "fit", "evaluate", "path-ablation", "path-guards", "input-ablation"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_GPUS"):
        raise RuntimeError("This training pipeline requires a scheduled GPU allocation")
    if socket.gethostname().startswith(("lo-", "login")):
        raise RuntimeError("Computation on login nodes is prohibited")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    config = json.loads(args.config.read_text())
    from jevmahjong.endgame.collect import write
    try:
        if args.stage == "coverage":
            from jevmahjong.endgame.coverage import collect_coverage
            result = collect_coverage(config)
        elif args.stage == "reuse-fourth-manifest":
            from jevmahjong.endgame.reuse import make_reused_manifest
            result = make_reused_manifest(config)
        elif args.stage == "target-manifest":
            from jevmahjong.endgame.coverage import make_target_manifest
            result = make_target_manifest(config)
        elif args.stage == "assemble":
            from jevmahjong.endgame.mix import assemble
            result = assemble(config)
        elif args.stage == "collect":
            from jevmahjong.endgame.collect import collect
            result = collect(config)
        elif args.stage == "label":
            from jevmahjong.endgame.futures import label
            result = label(config, args.limit)
        elif args.stage == "prepare":
            from jevmahjong.endgame.prepare import prepare
            result = prepare(config)
        elif args.stage == "enrich":
            from jevmahjong.endgame.enrich import enrich_positions
            result = enrich_positions(config)
        elif args.stage == "fit":
            from jevmahjong.endgame.fit import fit
            result = fit(config)
        elif args.stage == "reference":
            from jevmahjong.endgame.reference import build_reference
            result = build_reference(config)
        elif args.stage == "path-ablation":
            from jevmahjong.endgame.ablation import path_ablation
            result = path_ablation(config)
        elif args.stage == "path-guards":
            from jevmahjong.endgame.ablation import guard_ablation
            result = guard_ablation(config)
        elif args.stage == "input-ablation":
            from jevmahjong.endgame.ablation import guard_ablation
            result = guard_ablation(config, input_only=True)
        else:
            from jevmahjong.endgame.evaluate import evaluate
            result = evaluate(config)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        write(Path(config["out"]) / (args.stage + "-error-private.json"),
              {"error": str(error), "traceback": traceback.format_exc(),
               "job_id": os.environ.get("SLURM_JOB_ID")})
        raise


if __name__ == "__main__":
    main()

