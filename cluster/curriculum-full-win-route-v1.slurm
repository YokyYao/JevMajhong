#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-full-win-route
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-full-win-route-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_placement_full_win_policy_v1.py -v
timeout --signal=TERM --kill-after=30s 16m .venv/bin/python tools/verify_curriculum_full_win_route_v1.py

