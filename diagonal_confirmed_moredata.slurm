#!/bin/bash
#SBATCH --job-name=mahjong-confirmed-moredata
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diagonal-confirmed-moredata-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/collect_confirmed_diagonal_data.py --config diagonal-value-confirmed-data.json
.venv/bin/python scripts/audit_single_decision.py --run runs/diagonal-value-confirmed-data-20261030 \
  --out runs/diagonal-value-confirmed-data-20261030/training-audit.json
.venv/bin/python scripts/train_diagonal_value.py --config head-rl-diagonal-confirmed-moredata.json
if .venv/bin/python -c 'import json,sys; sys.exit(0 if json.load(open("runs/head-rl-diagonal-confirmed-moredata-20261030/training-result.json"))["status"]=="fit_complete" else 1)'; then
  .venv/bin/python scripts/audit_diagonal_value.py --run runs/head-rl-diagonal-confirmed-moredata-20261030 \
    --out runs/head-rl-diagonal-confirmed-moredata-20261030/fit-audit.json
fi
