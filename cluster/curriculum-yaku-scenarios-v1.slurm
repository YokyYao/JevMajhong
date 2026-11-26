#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-yaku-source
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004,ws-l5-002
#SBATCH --output=logs/curriculum-yaku-scenarios-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-yaku-scenarios-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=10s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=15s 12m .venv/bin/python tools/generate_curriculum_yaku_scenarios_v1.py --config configs/diagnostics/curriculum-yaku-scenarios-v1.json
