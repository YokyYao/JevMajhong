"""Select a practical default only when paired arena evidence supports improvement."""
import argparse
import json
from pathlib import Path
import statistics

from arena import ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arenas", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decisions = []
    for directory in args.arenas:
        report = json.loads((directory/"summary.json").read_text())
        rows = [json.loads(line) for line in (directory/"games.jsonl").read_text().splitlines()]
        seeds = sorted({x["candidate"]["seed"] for x in rows})
        first_deltas = [statistics.mean(int(x["candidate"]["rank"] == 1) - int(x["baseline"]["rank"] == 1)
                                       for x in rows if x["candidate"]["seed"] == seed) for seed in seeds]
        first_ci = ci(first_deltas, confidence=0.975)  # two predeclared policy comparisons
        rank_ci = report["paired_rank_delta_ci95"]
        qualified = (len(seeds) >= 32 and first_ci is not None and first_ci[0] > 0
                     and rank_ci is not None and rank_ci[1] <= 0 and report["fourth_rate"] <= 0.25)
        decisions.append({"arena": str(directory), "checkpoint": report["checkpoint"],
                          "guarded": report["guarded"], "first_rate": report["first_rate"],
                          "first_delta_ci97_5": first_ci, "mean_rank": report["mean_rank"],
                          "qualified": qualified})
    eligible = [d for d in decisions if d["qualified"]]
    selected = max(eligible, key=lambda x: (x["first_rate"], -x["mean_rank"])) if eligible else None
    result = {"active_policy": "model" if selected else "efficiency", "selected": selected,
              "comparisons": decisions,
              "note": "Selection is versus this fixed efficiency baseline only; not a Mahjong Soul rank claim."}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
