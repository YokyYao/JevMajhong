"""Create a deterministic, checksummed checkpoint archive split for GitHub transport."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part-mib", type=int, default=4)
    args = parser.parse_args()
    if args.part_mib < 1:
        parser.error("--part-mib must be positive")
    state = json.loads((args.checkpoint/"trainer_state.json").read_text())
    files = sorted(args.checkpoint.iterdir())
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("Checkpoint must contain regular top-level files only")
    args.out.mkdir(parents=True, exist_ok=False)
    archive = args.out/"checkpoint.tar.gz"
    with archive.open("wb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0, compresslevel=3) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as tar:
            for path in files:
                info = tar.gettarinfo(str(path), arcname=path.name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
    parts = []
    with archive.open("rb") as stream:
        index = 0
        while block := stream.read(args.part_mib*1024*1024):
            name = f"checkpoint.tar.gz.part-{index:03d}"
            (args.out/name).write_bytes(block)
            parts.append({"name": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
            index += 1
    manifest = {"format": "jevmahjong-checkpoint-parts-v1", "optimizer_step": state["optimizer_step"],
        "world_size": state["world_size"], "base": state["args"]["base"],
        "base_revision": state["args"]["base_revision"], "dataset_sha256": state["dataset_sha256"],
        "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in files],
        "parts": parts}
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"step": manifest["optimizer_step"], "archive_bytes": manifest["archive_bytes"], "parts":len(parts)}))


if __name__ == "__main__":
    main()
