#!/bin/bash
#SBATCH --job-name=mahjong-restart
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/restart-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python -m unittest test_semantic.py
.venv/bin/jevany train --config semantic-restart.toml
.venv/bin/python evaluate.py runs/semantic-2026-restart data/semantic-2026-v1/validation.jsonl --out runs/semantic-2026-restart/validation-full.json
# New evaluation seeds; do not reuse the warmup inspection's 510000/510001.
.venv/bin/python arena.py --checkpoint runs/semantic-2026-restart --seeds 32 --seed-start 710000 --out runs/arena-restart-model
.venv/bin/python arena.py --checkpoint runs/semantic-2026-restart --guarded --seeds 32 --seed-start 710000 --out runs/arena-restart-guarded
.venv/bin/python promote.py runs/arena-restart-model runs/arena-restart-guarded --out runs/champion.json
