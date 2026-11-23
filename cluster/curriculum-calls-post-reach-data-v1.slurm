#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-calls-postreach-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018
#SBATCH --output=logs/curriculum-calls-postreach-data-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_CALLS_DATA_PHASE:?Set prepare or audit}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-calls-postreach-data-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_CALLS_DATA_PHASE" in
  prepare)
    timeout --signal=TERM --kill-after=15s 3m .venv/bin/python -m unittest tests.core.test_curriculum_calls_post_reach_data_v1 -v
    timeout --signal=TERM --kill-after=15s 12m .venv/bin/python tools/curriculum_calls_post_reach_data_v1.py --config configs/training/curriculum-calls-post-reach-data-v1.json --phase prepare ;;
  audit)
    timeout --signal=TERM --kill-after=15s 16m .venv/bin/python tools/curriculum_calls_post_reach_data_v1.py --config configs/training/curriculum-calls-post-reach-data-v1.json --phase audit ;;
  *) exit 2 ;;
esac
