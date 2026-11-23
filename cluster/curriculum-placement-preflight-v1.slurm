#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-placement-preflight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l2-010,ws-l1-018,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-placement-preflight-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-placement-preflight-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_placement_integration_v1.py -v
timeout --signal=TERM --kill-after=30s 32m .venv/bin/python tools/preflight_curriculum_placement_v1.py \
  --registered runs/curriculum-placement-course-v1/fit-plan/registered.json \
  --out runs/curriculum-placement-course-v1/upstream-integration-v1
