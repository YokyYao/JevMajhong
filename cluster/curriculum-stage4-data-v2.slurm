#!/bin/bash
# Reusable public data-only recipe; run after the stage1 source registry exists.
# Submit with --export=ALL,JEV_STAGE4_SOURCE=rule or JEV_STAGE4_SOURCE=mortal.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-stage4-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=ws-l1-010
#SBATCH --output=logs/curriculum-stage4-data-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-stage4-data-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$SLURM_SUBMIT_DIR"
case "${JEV_STAGE4_SOURCE:?Choose rule or mortal}" in
  rule)
    case_builder=tools/collect_curriculum_endgame_cases.py
    case_config=configs/training/curriculum-stage4-cases-v1.json
    case_root=runs/curriculum-stage4-placement-data-v1
    ;;
  mortal)
    case_builder=tools/collect_curriculum_mortal_endgame_cases_v4.py
    case_config=configs/training/curriculum-stage4-mortal-cases-v4.json
    case_root=runs/curriculum-stage4-placement-mortal-data-v4
    ;;
  *) exit 2 ;;
esac
hostname
timeout --signal=TERM --kill-after=15s 2m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python -m unittest discover -s tests/core -p 'test_curriculum*.py'
timeout --signal=TERM --kill-after=15s 20m .venv/bin/python "$case_builder" --config "$case_config"
timeout --signal=TERM --kill-after=15s 3m .venv/bin/python tools/audit_curriculum_endgame_cases.py \
  --data-root "$case_root/data" --out "$case_root/input-audit-public-v1"
