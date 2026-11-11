"""Verify and restore a checkpoint published as archive parts; no model dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def safe_name(name):
    if not name or Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("Manifest contains a non-local filename")
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a new directory")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("format") != "jevmahjong-checkpoint-parts-v1":
        raise ValueError("Unsupported manifest format")
    expected = {safe_name(f["name"]): f for f in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate checkpoint files")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        archive_sha = hashlib.sha256()
        size = 0
        for part in manifest["parts"]:
            path = args.manifest.parent / safe_name(part["name"])
            part_sha = hashlib.sha256()
            count = 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    archive.write(block); archive_sha.update(block); part_sha.update(block)
                    count += len(block)
            if count != part["bytes"] or part_sha.hexdigest() != part["sha256"]:
                raise ValueError(f"Part checksum failed: {part['name']}")
            size += count
        if size != manifest["archive_bytes"] or archive_sha.hexdigest() != manifest["archive_sha256"]:
            raise ValueError("Archive checksum failed")
        archive.seek(0)
        with tempfile.TemporaryDirectory(dir=args.out.parent, prefix="checkpoint-restore-") as tmp:
            stage = Path(tmp)/"model"
            stage.mkdir()
            seen = set()
            with tarfile.open(fileobj=archive, mode="r:gz") as tar:
                for member in tar:
                    name = safe_name(member.name)
                    if not member.isfile() or name not in expected or name in seen:
                        raise ValueError("Unexpected archive entry")
                    spec = expected[name]
                    if member.size != spec["bytes"]:
                        raise ValueError(f"File size mismatch: {name}")
                    sha = hashlib.sha256()
                    with tar.extractfile(member) as stream, (stage/name).open("xb") as output:
                        for block in iter(lambda: stream.read(1024*1024), b""):
                            output.write(block); sha.update(block)
                    if sha.hexdigest() != spec["sha256"]:
                        raise ValueError(f"File checksum failed: {name}")
                    seen.add(name)
            if seen != set(expected):
                raise ValueError("Checkpoint archive is incomplete")
            shutil.move(str(stage), str(args.out))
    print(json.dumps({"restored": str(args.out), "optimizer_step":manifest["optimizer_step"],
                      "files_verified":len(expected)}))


if __name__ == "__main__":
    main()
