#!/bin/bash
#SBATCH --job-name=mahjong-continue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/continue-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
checkpoint=${1:?Pass a complete same-world-size training checkpoint}
.venv/bin/jevany train --resume "$checkpoint" --out runs/semantic-2026-continued
.venv/bin/python evaluate.py runs/semantic-2026-continued data/semantic-2026-v1/validation.jsonl --out runs/semantic-2026-continued/validation-full.json
.venv/bin/python arena.py --checkpoint runs/semantic-2026-continued --seeds 32 --seed-start 710000 --out runs/arena-continued-model
.venv/bin/python arena.py --checkpoint runs/semantic-2026-continued --guarded --seeds 32 --seed-start 710000 --out runs/arena-continued-guarded
.venv/bin/python promote.py runs/arena-continued-model runs/arena-continued-guarded --out runs/champion.json
