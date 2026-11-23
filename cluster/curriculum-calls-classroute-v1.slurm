#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-calls-classroute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018
#SBATCH --output=logs/curriculum-calls-classroute-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_CALLS_PHASE:?Set preflight, prepare, fit or audit}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-calls-classroute-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_CALLS_PHASE" in
  preflight)
    timeout --signal=TERM --kill-after=15s 3m .venv/bin/python -m unittest tests.core.test_curriculum_calls_classroute_v1 -v
    timeout --signal=TERM --kill-after=15s 20m .venv/bin/python tools/preflight_curriculum_calls_classroute_v1.py --out runs/curriculum-calls-classroute-v1/preflight-v1 ;;
  prepare)
    timeout --signal=TERM --kill-after=15s 12m .venv/bin/python tools/curriculum_calls_classroute_parent_v1.py --out runs/curriculum-calls-classroute-v1/parent
    timeout --signal=TERM --kill-after=15s 12m .venv/bin/python tools/register_curriculum_calls_classroute_v1.py --config configs/training/curriculum-calls-classroute-v1.json --parent-seal runs/curriculum-calls-classroute-v1/parent/parent-seal-private.json ;;
  fit)
    timeout --signal=TERM --kill-after=30s 710m .venv/bin/python tools/train_curriculum_calls_classroute_v1.py --registered runs/curriculum-calls-classroute-v1/fit-plan/registered.json ;;
  audit)
    timeout --signal=TERM --kill-after=30s 55m .venv/bin/python tools/audit_curriculum_calls_classroute_v1.py --registered runs/curriculum-calls-classroute-v1/fit-plan/registered.json --out runs/curriculum-calls-classroute-v1/final-evidence-private.json ;;
  *) exit 2 ;;
esac
