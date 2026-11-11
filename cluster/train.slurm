#!/bin/bash
#SBATCH --job-name=jev-tempo-fit
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/jev-tempo-fit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
.venv/bin/python tools/run_experiment.py "${2:-train}" --config "${1:?configuration JSON}"
