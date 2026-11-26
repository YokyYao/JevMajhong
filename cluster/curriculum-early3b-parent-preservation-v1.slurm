#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-3b-parent-terminal
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:05:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-early3b-parent-preservation-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"capture\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-early3b-parent-preservation-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=20s 4m .venv/bin/python tools/preserve_curriculum_early3b_parent_v1.py --phase capture
