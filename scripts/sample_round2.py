"""Prepare a new, nonduplicated training mixture without opening test decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_shard import split_for_match


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select(records, total=64000, discard_fraction=0.85, seed=4243):
    discard, other = [], []
    for record in records:
        question = record["questions"]["action"]
        target_type = question["criteria"][question["label"]]["type"]
        (discard if target_type == "dahai" else other).append(record)
    needed = round(total * discard_fraction)
    if len(discard) < needed or len(other) < total - needed:
        raise ValueError(f"Insufficient unique pool: discard={len(discard)}, other={len(other)}; "
                         f"need {needed}/{total-needed}. Expand pool; never duplicate records.")
    priority = lambda r: hashlib.sha256(f"{seed}:{r['_meta']['id']}".encode()).digest()
    chosen = sorted(discard, key=priority)[:needed] + sorted(other, key=priority)[:total-needed]
    random.Random(seed).shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    parser.add_argument("--previous", type=Path, default=Path("data/semantic-2026-v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, default=64000)
    parser.add_argument("--discard-fraction", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=4243)
    args = parser.parse_args()
    if args.total < 1 or not 0 < args.discard_fraction < 1:
        parser.error("Invalid total or discard fraction")
    prior = json.loads((args.previous / "provenance.json").read_text())
    excluded = {m for matches in prior["matches"].values() for m in matches}
    validation_path = args.previous / "validation.jsonl"
    validation = [json.loads(s) for s in validation_path.read_text().splitlines()]
    validation_hashes = {r["_meta"]["text_sha256"] for r in validation}
    seen_ids, seen_hashes, records = set(), set(), []
    for line in (args.pool / "train.jsonl").read_text().splitlines():
        record = json.loads(line)
        meta = record["_meta"]
        if meta["group_id"] in excluded or split_for_match(meta["group_id"]) != "train":
            raise ValueError("Pool contains a held-out or previously used match")
        if meta["id"] in seen_ids or meta["text_sha256"] in seen_hashes:
            raise ValueError("Duplicate pool record")
        if meta["text_sha256"] in validation_hashes:
            raise ValueError("Training pool overlaps validation content")
        if meta["split"] != "train":
            raise ValueError("Non-training record in pool")
        seen_ids.add(meta["id"]); seen_hashes.add(meta["text_sha256"]); records.append(record)
    chosen = select(records, args.total, args.discard_fraction, args.seed)
    types = Counter(r["questions"]["action"]["criteria"][r["questions"]["action"]["label"]]["type"]
                    for r in chosen)
    groups = Counter(r["_meta"]["group_id"] for r in chosen)
    if max(groups.values()) > 32:
        raise ValueError("More than 32 decisions from one match")
    args.out.mkdir(parents=True, exist_ok=False)
    train_path = args.out / "train.jsonl"
    train_path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in chosen))
    shutil.copyfile(validation_path, args.out / "validation.jsonl")
    provenance = {"source": json.loads((args.pool / "provenance.json").read_text()),
                  "counts": {"train": len(chosen), "validation": len(validation), "test": 0},
                  "matches": {"train": sorted(groups),
                              "validation": sorted({r["_meta"]["group_id"] for r in validation}), "test": []},
                  "seed": args.seed, "discard_fraction": args.discard_fraction,
                  "action_types": dict(types), "sealed_test_reference": str(args.previous / "test.jsonl")}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    audit = {"complete": True, "pool_records": len(records), "train_records": len(chosen),
             "train_matches": len(groups), "validation_records": len(validation),
             "excluded_prior_matches": len(excluded), "max_decisions_per_match": max(groups.values()),
             "action_types": dict(types), "train_sha256": sha(train_path),
             "validation_sha256": sha(validation_path), "test_opened": False,
             "split_and_content_overlap": False}
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
