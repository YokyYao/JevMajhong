"""Download one pinned source archive; do not redistribute it with the code."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = "6450bb584cef43ede3b514df48e301a832c6818b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=(2019, 2026), required=True)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "data" / "raw"
    print(hf_hub_download("nikketryhard/tenhou-houou-mjai", repo_type="dataset",
                          revision=REVISION, filename=f"majsoul-throne-{args.year}.tar",
                          local_dir=str(target)))


if __name__ == "__main__":
    main()
