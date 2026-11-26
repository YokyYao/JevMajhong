#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-riichi-paired
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:30:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-progression-paired-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "$JEV_PAIRED_PHASE"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_PAIRED_PHASE" "$task_rc" > "logs/curriculum-progression-paired-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_PAIRED_PHASE" in
  run)
    JEV_PAIRED_PHASE=prepare
    timeout --signal=TERM --kill-after=15s 12m .venv/bin/python tools/curriculum_progression_paired_v1.py --phase prepare
    JEV_PAIRED_PHASE=fit
    timeout --signal=TERM --kill-after=30s 830m .venv/bin/python tools/curriculum_progression_paired_v1.py --phase fit ;;
  audit)
    test -n "$JEV_PAIRED_FIT_JOB"
    timeout --signal=TERM --kill-after=30s 55m .venv/bin/python tools/curriculum_progression_paired_v1.py --phase audit --fit-job "$JEV_PAIRED_FIT_JOB" ;;
  *) exit 2 ;;
esac
