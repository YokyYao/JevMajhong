#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=jev-targeted-shard-terminal
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/curriculum-targeted-shard-terminal-v3-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "$JEV_SHARD_TERM_KIND"
test -n "$JEV_SHARD_TERM_SHARD"
test -n "$JEV_SHARD_TERM_JOB"
test -n "$JEV_SHARD_TERM_ATTEMPT"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"capture\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-targeted-shard-terminal-v3-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
extra=()
if [ "$JEV_SHARD_TERM_KIND" = original ]; then
  test -n "$JEV_SHARD_TERM_ARRAY"
  extra=(--array-job "$JEV_SHARD_TERM_ARRAY")
fi
timeout --signal=TERM --kill-after=15s 4m .venv/bin/python tools/capture_curriculum_targeted_shard_terminal_v3.py --phase capture --kind "$JEV_SHARD_TERM_KIND" --shard "$JEV_SHARD_TERM_SHARD" --job "$JEV_SHARD_TERM_JOB" --attempt "$JEV_SHARD_TERM_ATTEMPT" "${extra[@]}"
