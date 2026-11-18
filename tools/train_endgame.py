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
    parser.add_argument("stage", choices=("collect", "label", "prepare", "fit", "evaluate"))
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
        if args.stage == "collect":
            from jevmahjong.endgame.collect import collect
            result = collect(config)
        elif args.stage == "label":
            from jevmahjong.endgame.futures import label
            result = label(config, args.limit)
        elif args.stage == "prepare":
            from jevmahjong.endgame.prepare import prepare
            result = prepare(config)
        elif args.stage == "fit":
            from jevmahjong.endgame.fit import fit
            result = fit(config)
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
