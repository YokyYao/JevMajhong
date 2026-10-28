#!/bin/bash
#SBATCH --job-name=mahjong-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python -m unittest test_semantic.py
.venv/bin/jevany train --config semantic-full.toml
.venv/bin/python evaluate.py runs/semantic-2026-full data/semantic-2026-v1/validation.jsonl --out runs/semantic-2026-full/validation-full.json
.venv/bin/python arena.py --checkpoint runs/semantic-2026-full --seeds 32 --out runs/arena-2026-model
.venv/bin/python arena.py --checkpoint runs/semantic-2026-full --guarded --seeds 32 --out runs/arena-2026-guarded
.venv/bin/python promote.py runs/arena-2026-model runs/arena-2026-guarded --out runs/champion.json
