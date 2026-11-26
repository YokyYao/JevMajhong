#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-ri-anchor-activation-v3
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-ri-anchor-activation-v3-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
trap 'task_rc=$?; printf "{\"exit_code\":%d,\"job_id\":\"%s\",\"phase\":\"activate\"}\n" "$task_rc" "$SLURM_JOB_ID" > "logs/curriculum-ri-anchor-activation-v3-exit-$SLURM_JOB_ID.json"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
hostname
timeout --signal=TERM --kill-after=15s 8m .venv/bin/python tools/curriculum_ri_anchor_launch_binding_v3.py --phase activate
