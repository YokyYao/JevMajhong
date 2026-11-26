#!/usr/bin/env bash
#SBATCH --job-name=jev-targeted-case-terminal
#SBATCH --partition=gpu,ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=logs/curriculum-targeted-case-capture-v1-%j.log
#SBATCH --error=logs/curriculum-targeted-case-capture-v1-%j.log
set -euo pipefail
cd /home/junchi.yao/mahjong-discard-defense
trap 'jev_case_capture_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%s}\n" "$SLURM_JOB_ID" "$jev_case_capture_rc" > "logs/curriculum-targeted-case-capture-v1-exit-$SLURM_JOB_ID.json"; exit "$jev_case_capture_rc"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python tools/curriculum_targeted_parallel_capture_v1.py "$@"
