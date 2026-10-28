"""Reconstruct expert observations under Mahjong Soul rules; emit shared semantic schema."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tarfile
import tempfile

from riichienv import MjaiReplay
import zstandard
from audit_shard import split_for_match
from semantic import build_request, signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=16000)
    parser.add_argument("--validation", type=int, default=512)
    parser.add_argument("--test", type=int, default=512)
    parser.add_argument("--per-match", type=int, default=64)
    parser.add_argument("--exclude-provenance", type=Path, action="append", default=[],
                        help="Exclude every match listed in a previous dataset provenance")
    args = parser.parse_args()
    limits = {k: getattr(args, k) for k in ("train", "validation", "test")}
    if min(limits.values()) < 0 or args.train < 1 or args.per_match < 1:
        parser.error("train and per-match must be positive; validation/test may be zero")
    excluded = set()
    for path in args.exclude_provenance:
        provenance = json.loads(path.read_text())
        excluded.update(match for matches in provenance["matches"].values() for match in matches)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, failures, groups = Counter(), [], {k: set() for k in limits}
    types = Counter()
    seen = set()
    for key in limits:
        (args.out / f"{key}.jsonl").touch()
    with tarfile.open(args.archive, "r|*") as archive, tempfile.TemporaryDirectory() as tmp:
        replay_path = Path(tmp) / "replay.jsonl"
        for member in archive:
            if not member.isfile() or not member.name.endswith(".zst"):
                continue
            match = Path(member.name).name.removesuffix(".mjai.json.zst")
            if match in excluded:
                continue
            split = split_for_match(match)
            if counts[split] >= limits[split]:
                continue
            try:
                if member.size > 10_000_000:
                    raise ValueError("Oversized compressed match")
                with archive.extractfile(member) as compressed:
                    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                        raw = reader.read(10_000_001)
                if len(raw) > 10_000_000:
                    raise ValueError("Oversized decompressed match")
                events = [json.loads(line) for line in raw.splitlines() if line.strip()]
                start = next(e for e in events if e["type"] == "start_kyoku")
                if len(start["tehais"]) != 4:
                    continue
                replay_path.write_bytes(raw)
                replay = MjaiReplay.from_jsonl(str(replay_path), rule="mjsoul")
                samples = []
                for hand in replay.take_kyokus():
                    for pid, obs, expert in hand.steps(seat=None, skip_single_action=True):
                        actions = obs.legal_actions()
                        if len({signature(a) for a in actions}) < 2:
                            continue
                        samples.append((obs, expert))
                rng = random.Random(int(hashlib.sha256(match.encode()).hexdigest()[:16], 16))
                rng.shuffle(samples)
                converted = []
                for obs, expert in samples[:args.per_match]:
                    request, actions, _ = build_request(obs)
                    key = signature(expert)
                    signatures = [signature(a) for a in actions]
                    if key not in signatures:
                        raise ValueError("Expert action absent from legal candidates")
                    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                    if digest in seen:
                        continue
                    request["questions"]["action"]["label"] = f"a{signatures.index(key)}"
                    request["_meta"] = {"id": f"{match}/{len(converted)}", "group_id": match,
                        "source": args.archive.stem.replace("-", "_"), "split": split, "license": "CC-BY-4.0",
                        "text_sha256": digest}
                    converted.append((request, digest, json.loads(expert.to_mjai())["type"]))
            except Exception as exc:
                failures.append({"match": match, "error": str(exc)[:300]})
                continue
            with (args.out / f"{split}.jsonl").open("a") as stream:
                for request, digest, typ in converted:
                    if counts[split] >= limits[split]:
                        break
                    stream.write(json.dumps(request, separators=(",", ":")) + "\n")
                    counts[split] += 1
                    seen.add(digest)
                    groups[split].add(match)
                    types[typ] += 1
            if sum(counts.values()) % 1024 < args.per_match:
                print(json.dumps({"counts": dict(counts), "failed_matches": len(failures)}), flush=True)
            if all(counts[k] >= limits[k] for k in limits):
                break
    provenance = {"counts": dict(counts), "matches": {k: sorted(v) for k,v in groups.items()},
        "action_types": dict(types), "failures": failures,
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "source": "https://huggingface.co/datasets/nikketryhard/tenhou-houou-mjai",
        "attribution": ["NikkeTryHard/tenhou-to-mjai", "Mahjong Soul"],
        "schema": "semantic-v1", "rule": "mjsoul", "requested_counts": limits,
        "excluded_prior_matches": len(excluded),
        "exclude_provenance": [str(p) for p in args.exclude_provenance]}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if any(counts[k] < limits[k] for k in limits):
        raise ValueError(f"Incomplete requested split: {dict(counts)}; see provenance failures")
    print(json.dumps({"counts": dict(counts), "failed_matches": len(failures), "types": dict(types)}))


if __name__ == "__main__":
    main()
