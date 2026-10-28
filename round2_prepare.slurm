#!/bin/bash
#SBATCH --job-name=mahjong-r2-data
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/round2-data-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv/bin/python prepare_raw.py data/raw/majsoul-throne-2026.tar \
  --out data/semantic-2026-round2-pool --train 80000 --validation 0 --test 0 \
  --per-match 32 --exclude-provenance data/semantic-2026-v1/provenance.json
.venv/bin/python scripts/sample_round2.py data/semantic-2026-round2-pool \
  --out data/semantic-2026-round2
.venv/bin/jevany train --config semantic-round2.toml --dry-run
# No GPU is requested until CPU preparation and all separation audits succeed.
sbatch --parsable --partition=ws-ia round2.slurm | tee runs/round2-gpu-job-id.txt
