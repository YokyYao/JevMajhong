#!/usr/bin/env bash
#SBATCH --job-name=jev-qualified-final-source-proof
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/curriculum-final-source-proof-v1-%j.log
#SBATCH --error=logs/curriculum-final-source-proof-v1-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "${SLURM_JOB_ID:?Scheduler GPU allocation required}"
test -n "${SLURM_JOB_GPUS:?Scheduler GPU allocation required}"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'jev_final_source_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%s}\n" "$SLURM_JOB_ID" "$jev_final_source_rc" > "logs/curriculum-final-source-proof-v1-exit-$SLURM_JOB_ID.json"; exit "$jev_final_source_rc"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout --signal=TERM --kill-after=30s 55m .venv/bin/python tools/prove_curriculum_final_source_catalog_v1.py
