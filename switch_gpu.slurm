#!/bin/bash
#SBATCH --job-name=mahjong-freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/freegpu-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
checkpoint=${1:?Pass a complete same-world-size checkpoint}
.venv/bin/python scripts/check_gpu.py
.venv/bin/jevany train --resume "$checkpoint" --out runs/semantic-2026-freegpu
.venv/bin/python evaluate.py runs/semantic-2026-freegpu data/semantic-2026-v1/validation.jsonl --out runs/semantic-2026-freegpu/validation-full.json
.venv/bin/python arena.py --checkpoint runs/semantic-2026-freegpu --seeds 32 --seed-start 710000 --out runs/arena-freegpu-model
.venv/bin/python arena.py --checkpoint runs/semantic-2026-freegpu --guarded --seeds 32 --seed-start 710000 --out runs/arena-freegpu-guarded
.venv/bin/python promote.py runs/arena-freegpu-model runs/arena-freegpu-guarded --out runs/champion.json
