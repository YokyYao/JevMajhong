"""Execute the frozen second-round arenas and paired promotion comparisons."""
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena import ci


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def paired_comparison(candidate, reference):
    by_key = lambda rows: {(r["seed"], r["seat"]): r for r in rows}
    a, b = by_key(candidate), by_key(reference)
    if len(a) != len(candidate) or len(b) != len(reference) or a.keys() != b.keys():
        raise ValueError("Paired arenas must have identical, unique seed/seat coverage")
    seeds = sorted({s for s, _ in a})
    if set(a) != {(s, seat) for s in seeds for seat in range(4)}:
        raise ValueError("All four seats are required for every seed")
    result = {"independent_seeds": len(seeds), "games": len(a)}
    for name, fn, confidence in [("first", lambda r: int(r["rank"] == 1), .975),
                                 ("rank", lambda r: r["rank"], .95),
                                 ("points", lambda r: r["score"], .95)]:
        values = [statistics.mean(fn(a[s, seat])-fn(b[s, seat]) for seat in range(4)) for s in seeds]
        result[name + "_delta"] = statistics.mean(values)
        result[name + "_delta_ci"] = ci(values, confidence=confidence)
    return result


def main():
    run = ROOT / "runs/semantic-2026-round2"
    if (run / "strength-revision.json").exists():
        raise SystemExit("NLL-only selection was superseded; run scripts/evaluate_round2_checkpoints.py instead")
    selected = json.loads((run / "selected-validation.json").read_text())
    previous = "runs/semantic-2026-freegpu/checkpoints/step-002000"
    candidate = selected["checkpoint"]
    manifest = {"selected": selected, "previous": previous, "screen_seed_start": 810000,
                "confirmation_seed_start": 910000, "status": "screening"}
    write(run / "evaluation-status.json", manifest)

    def arena(checkpoint, guarded, seeds, start, name):
        directory = run / name
        command = [sys.executable, str(ROOT / "arena.py"), "--checkpoint", checkpoint,
                   "--seeds", str(seeds), "--seed-start", str(start), "--out", str(directory)]
        if guarded:
            command.append("--guarded")
        if directory.exists():
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, check=True)
        return [json.loads(s) for s in (directory / "games.jsonl").read_text().splitlines()]

    for name, checkpoint, guarded in [("previous-model", previous, False),
                                      ("previous-guarded", previous, True),
                                      ("new-model", candidate, False),
                                      ("new-guarded", candidate, True)]:
        arena(checkpoint, guarded, 32, 810000, "screen-"+name)
    manifest["status"] = "confirmation"
    write(run / "evaluation-status.json", manifest)
    old = arena(previous, True, 128, 910000, "confirm-previous-guarded")
    new = arena(candidate, True, 128, 910000, "confirm-new-guarded")
    vs_rule = paired_comparison([r["candidate"] for r in new], [r["baseline"] for r in new])
    vs_previous = paired_comparison([r["candidate"] for r in new], [r["candidate"] for r in old])
    fourth = sum(r["candidate"]["rank"] == 4 for r in new)/len(new)
    qualified = (vs_rule["independent_seeds"] == 128 and vs_previous["independent_seeds"] == 128
                 and vs_rule["first_delta_ci"][0] > 0 and vs_previous["first_delta_ci"][0] > 0
                 and vs_rule["rank_delta_ci"][1] <= 0 and fourth <= .25)
    report = {"active_policy": "model" if qualified else "efficiency",
              "selected": {"checkpoint": candidate, "guarded": True} if qualified else None,
              "candidate": candidate, "qualified": qualified, "fourth_rate": fourth,
              "vs_rule": vs_rule, "vs_previous_guarded": vs_previous,
              "mean_final_points": statistics.mean(r["candidate"]["score"] for r in new),
              "mean_net_points": statistics.mean(r["candidate"]["score"]-25000 for r in new),
              "note": "Two predeclared first-rate comparisons use 97.5% seed-clustered t intervals; "
                      "raw point comparisons are secondary. Not a Mahjong Soul rank claim."}
    write(run / "champion.json", report)
    if qualified:
        current = ROOT / "runs/champion.json"
        if current.exists() and not (run / "previous-champion.json").exists():
            shutil.copyfile(current, run / "previous-champion.json")
        write(current, report)
    manifest.update(status="complete", qualified=qualified)
    write(run / "evaluation-status.json", manifest)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
