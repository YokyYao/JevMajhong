#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-yaku-paired
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-yaku-paired-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_YAKU_PAIRED_PHASE:?Set prepare, fit or audit}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_YAKU_PAIRED_PHASE" "$task_rc" > "logs/curriculum-yaku-paired-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_YAKU_PAIRED_PHASE" in
  prepare)
    timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_paired_yaku_launch_v1.py -v
    timeout --signal=TERM --kill-after=30s 25m .venv/bin/python tools/curriculum_paired_yaku_launch_v1.py --phase prepare ;;
  fit)
    timeout --signal=TERM --kill-after=30s 1190m .venv/bin/python tools/curriculum_paired_yaku_launch_v1.py --phase fit ;;
  audit)
    test -n "$JEV_YAKU_PAIRED_FIT_JOB"
    timeout --signal=TERM --kill-after=30s 85m .venv/bin/python tools/curriculum_paired_yaku_launch_v1.py --phase audit --fit-job "$JEV_YAKU_PAIRED_FIT_JOB" ;;
  *) exit 2 ;;
esac
