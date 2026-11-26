#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-paired-terminal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-paired-terminal-v2-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-paired-terminal-v2-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 3m .venv/bin/python tools/capture_curriculum_paired_terminal_v2.py --phase capture
