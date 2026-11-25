#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-placement-canonical
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-placement-canonical-course-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
# Retain the original submitter's environment key; the bound command and
# receipt prefix identify this canonical route unambiguously.
test -n "$JEV_PLACEMENT_BALANCED_PHASE"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_PLACEMENT_BALANCED_PHASE" "$task_rc" > "logs/curriculum-placement-canonical-course-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
case "$JEV_PLACEMENT_BALANCED_PHASE" in
  register) task_limit=30m ;;
  integration) task_limit=32m ;;
  prepare) task_limit=15m ;;
  fit) task_limit=1420m ;;
  audit) task_limit=110m ;;
  bind|submit|capture|enqueue-parent-terminal|parent-terminal) task_limit=4m ;;
  *) exit 64 ;;
esac
case "$JEV_PLACEMENT_BALANCED_PHASE" in
  integration|fit|audit) timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py ;;
esac
timeout --signal=TERM --kill-after=30s "$task_limit" .venv/bin/python tools/launch_curriculum_placement_canonical_v2.py --phase "$JEV_PLACEMENT_BALANCED_PHASE"
