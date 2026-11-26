#!/usr/bin/env bash
#SBATCH --job-name=jev-final-source-dispatch
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=logs/curriculum-final-source-dispatch-v1-%j.log
#SBATCH --error=logs/curriculum-final-source-dispatch-v1-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "${SLURM_JOB_ID:?}"
test -n "${SLURM_JOB_GPUS:?}"
test -n "${JEV_FINAL_SOURCE_PHASE:?}"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'jev_dispatch_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%s}\n" "$SLURM_JOB_ID" "$JEV_FINAL_SOURCE_PHASE" "$jev_dispatch_rc" > "logs/curriculum-final-source-dispatch-v1-exit-$SLURM_JOB_ID.json"; exit "$jev_dispatch_rc"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout --signal=TERM --kill-after=20s 4m .venv/bin/python tools/launch_curriculum_final_source_proof_v1.py --phase "$JEV_FINAL_SOURCE_PHASE"
