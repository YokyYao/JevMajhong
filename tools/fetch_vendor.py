"""Fetch the pinned upstream trainer; never modify an existing checkout."""
from pathlib import Path
import subprocess

REPO = "https://github.com/SimpleJev/JevAny.git"
REVISION = "c740121a4532592b09b6272104d5f29a94c232e5"


def main():
    target = Path(__file__).resolve().parents[1] / "vendor" / "JevAny"
    if target.exists():
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if actual != REVISION or dirty:
            raise RuntimeError("Existing JevAny checkout differs; inspect it before proceeding")
        print(f"Already available: {target} @ {REVISION}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "remote", "add", "origin", REPO], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", REVISION], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)


if __name__ == "__main__":
    main()
