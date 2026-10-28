"""Audit one local NPZ/WebDataset shard without extracting archive paths."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def split_for_match(match_id):
    # All seats, rounds and decisions of a match must remain in one split.
    bucket = int.from_bytes(hashlib.sha256(match_id.encode()).digest()[:8], "big") % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def audit(blob, match_id):
    import numpy as np
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        offsets = data["legal_offsets"].astype("int64")
        expert = data["expert_index"].astype("int64").reshape(-1)
        if len(offsets) != len(expert) + 1:
            raise ValueError(f"{match_id}: inconsistent offsets")
        sizes = np.diff(offsets)
        if offsets[0] != 0 or offsets[-1] != len(data["legal_type"]):
            raise ValueError(f"{match_id}: invalid action boundaries")
        if not np.all((sizes > 0) & (expert >= 0) & (expert < sizes)):
            raise ValueError(f"{match_id}: expert action outside legal actions")
        hands = data["hand_counts"]
        if hands.shape != (len(expert), 34) or not np.all((hands >= 0) & (hands <= 4)):
            raise ValueError(f"{match_id}: invalid hand counts")
        return {"match_id": match_id, "split": split_for_match(match_id),
                "decisions": len(expert)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.path.suffix == ".npz":
        print(json.dumps(audit(args.path.read_bytes(), args.path.stem)))
        return
    with tarfile.open(args.path, "r|*") as archive:
        count = 0
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            if member.size > 256 * 1024 * 1024:
                raise ValueError("NPZ member exceeds audit size limit")
            with archive.extractfile(member) as stream:
                print(json.dumps(audit(stream.read(), Path(member.name).stem)))
            count += 1
            if count >= args.limit:
                break
        if not count:
            raise ValueError("No NPZ matches found")


if __name__ == "__main__":
    main()
