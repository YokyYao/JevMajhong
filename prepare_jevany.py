"""Convert a small real-log shard into labelled JevAny decision requests."""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from audit_shard import audit, split_for_match

FIELDS = ("round_wind", "kyoku", "dealer", "seat_wind", "honba", "kyotaku",
          "scores", "ranks", "dora", "tiles_left", "hand_counts", "hand_red_counts",
          "trigger_type", "trigger_actor", "trigger_tile", "player")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shard", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train", type=int, default=2048)
    p.add_argument("--validation", type=int, default=64)
    p.add_argument("--test", type=int, default=64)
    args = p.parse_args()
    limits = {k: getattr(args, k) for k in ("train", "validation", "test")}
    if min(limits.values()) < 1:
        p.error("split limits must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(limits, 0)
    groups = {k: set() for k in limits}
    seen = set()
    with ExitStack() as stack:
        files = {k: stack.enter_context((args.out / f"{k}.jsonl").open("w")) for k in limits}
        archive = stack.enter_context(tarfile.open(args.shard, "r|*"))
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            if member.size > 256 * 1024 * 1024:
                raise ValueError("Oversized archive member")
            match = Path(member.name).stem
            partition = split_for_match(match)
            if counts[partition] >= limits[partition]:
                continue
            with archive.extractfile(member) as stream:
                blob = stream.read()
            audit(blob, match)
            with np.load(io.BytesIO(blob), allow_pickle=False) as d:
                stream_owners = {}
                for row in range(len(d["expert_index"])):
                    stream = int(d["history_ref"][row])
                    owner = (int(d["kyoku_index"][row]), int(d["player"][row]))
                    if stream in stream_owners and stream_owners[stream] != owner:
                        raise ValueError("History stream shared across players or hands")
                    stream_owners[stream] = owner
                # Spread the pilot over different hands/stages, not just opening discards.
                indices = np.linspace(0, len(d["expert_index"]) - 1,
                                      min(16, len(d["expert_index"])), dtype=int)
                for i in indices:
                    lo, hi = (int(x) for x in d["legal_offsets"][i:i+2])
                    if hi - lo < 2:
                        continue
                    state = {key: d[key][i].tolist() for key in FIELDS}
                    ref, length = int(d["history_ref"][i]), int(d["history_len"][i])
                    if not 0 <= ref < len(d["history_offsets"]) - 1:
                        raise ValueError("Invalid history reference")
                    start, end = (int(x) for x in d["history_offsets"][ref:ref+2])
                    if not 0 <= length <= end - start:
                        raise ValueError("Invalid historical prefix")
                    # Read only the observed prefix, never the future remainder of the stream.
                    state["public_history"] = np.column_stack([
                        d["history_type"][start:start+length],
                        d["history_actor"][start:start+length],
                        d["history_tiles"][start:start+length],
                        d["history_tsumogiri"][start:start+length],
                    ]).tolist()
                    criteria = {f"a{j-lo}": {"type_code": int(d["legal_type"][j]),
                                            "tiles_code": d["legal_tiles"][j].tolist()}
                                for j in range(lo, hi)}
                    question = {"type": "choice", "instructions":
                        "Choose the legal action matching strong four-player riichi Mahjong play. "
                        "Tiles and event/action types use the source NPZ numeric encoding. "
                        "hand_counts uses 34 tile types; public_history rows are "
                        "[event_type,actor,tile0,tile1,tile2,tile3,tsumogiri]. "
                        "Use only information in this observation.", "criteria": criteria}
                    content = {"state": state, "questions": {"action": question}}
                    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    question["label"] = f"a{int(d['expert_index'][i])}"
                    content["_meta"] = {
                        "source": "tenhou_mjai_pilot", "id": f"{match}/{i}",
                        "group_id": match, "split": partition, "license": "CC-BY-4.0",
                        "text_sha256": digest,
                    }
                    files[partition].write(json.dumps(content, separators=(",", ":")) + "\n")
                    counts[partition] += 1
                    groups[partition].add(match)
                    if counts[partition] >= limits[partition]:
                        break
            if all(counts[k] >= limits[k] for k in limits):
                break
    if any(counts[k] == 0 for k in limits):
        raise ValueError(f"Empty split: {counts}")
    manifest = {"counts": counts, "matches": {k: sorted(v) for k,v in groups.items()},
                "shard_sha256": hashlib.sha256(args.shard.read_bytes()).hexdigest(),
                "source": "https://huggingface.co/datasets/1nfrared218/tenhou-mjai-riichi-training",
                "attribution": ["1nfrared218", "NikkeTryHard/tenhou-to-mjai", "Tenhou"],
                "note": "Pilot only; numeric encodings retained. Test split is not used in training."}
    (args.out / "provenance.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
