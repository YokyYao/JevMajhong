#!/bin/bash
#SBATCH --job-name=mahjong-sampling-prior
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/call-sampling-prior-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/diagnose_call_sampling_prior.py --config "${1:?Diagnostic config required}"
