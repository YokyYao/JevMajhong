#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-recovered-exposures
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-recovered-exposures-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_RECOVERED_EXPOSURE_PHASE:?Set mix or exposures}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_RECOVERED_EXPOSURE_PHASE" "$task_rc" > "logs/curriculum-recovered-exposures-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_RECOVERED_EXPOSURE_PHASE" in
  mix|exposures) timeout --signal=TERM --kill-after=30s 55m .venv/bin/python tools/prepare_curriculum_recovered_exposures_v1.py --phase "$JEV_RECOVERED_EXPOSURE_PHASE" ;;
  *) exit 2 ;;
esac
