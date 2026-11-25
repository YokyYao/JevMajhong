#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-best3a-direct3b
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-yaku-best3a-v1-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
trap 'phase_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_BEST3A_PHASE" "$phase_rc" > "logs/curriculum-yaku-best3a-v1-exit-$SLURM_JOB_ID.json"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_BEST3A_PHASE" in
  prepare) phase_limit=42m ;;
  fit) phase_limit=1410m ;;
  audit) phase_limit=110m ;;
  capture) phase_limit=4m ;;
  *) exit 2 ;;
esac
timeout --signal=TERM --kill-after=30s "$phase_limit" .venv/bin/python tools/curriculum_yaku_best3a_v1.py --phase "$JEV_BEST3A_PHASE"
