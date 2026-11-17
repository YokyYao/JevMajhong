"""Restore and verify the public Jev Play v2 training/validation JSONL files."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def restore(source: Path, output: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "jevmahjong-jsonl-gzip-parts-v1":
        raise ValueError("Unsupported training data archive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    counts = {}
    with tempfile.TemporaryDirectory(prefix=".jev-data-", dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        for split, spec in manifest["splits"].items():
            if split not in {"train", "validation"}:
                raise ValueError("Unexpected split name")
            packed = bytearray()
            for part in spec["parts"]:
                name = part["name"]
                if Path(name).name != name:
                    raise ValueError("Invalid archive path")
                data = (source / name).read_bytes()
                if len(data) != part["bytes"] or digest(data) != part["sha256"]:
                    raise ValueError(f"Part checksum failed: {name}")
                packed.extend(data)
            if len(packed) != spec["archive_bytes"] or digest(packed) != spec["archive_sha256"]:
                raise ValueError(f"Archive checksum failed: {split}")
            data = gzip.decompress(packed)
            if len(data) != spec["bytes"] or digest(data) != spec["sha256"]:
                raise ValueError(f"JSONL checksum failed: {split}")
            rows = [json.loads(line) for line in data.splitlines()]
            if len(rows) != spec["records"]:
                raise ValueError(f"Record count failed: {split}")
            groups[split] = {row["_meta"]["group_id"] for row in rows}
            counts[split] = len(rows)
            (stage / f"{split}.jsonl").write_bytes(data)
        if groups["train"] & groups["validation"]:
            raise ValueError("Training/validation source groups overlap")
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets" / "jev-play-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "jev-play-v2")
    args = parser.parse_args()
    print(json.dumps({"output": str(args.output), "records": restore(args.source, args.output)}, ensure_ascii=False))
