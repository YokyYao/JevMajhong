#!/bin/bash
#SBATCH --job-name=mahjong-call-confirm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/call-prior-confirm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/watch_call_repair.py --config call-prior-confirmation-20261110.json --eval-job "$SLURM_JOB_ID" --max-seconds 25200 &
call_observer_pid=$!
stop_call_observer() {
    local call_worker_exit=$?
    trap - EXIT
    kill "$call_observer_pid" 2>/dev/null || true
    wait "$call_observer_pid" 2>/dev/null || true
    exit "$call_worker_exit"
}
trap stop_call_observer EXIT
.venv/bin/python scripts/evaluate_call_repair.py --config call-prior-confirmation-20261110.json
.venv/bin/python scripts/audit_call_development.py --config call-prior-confirmation-20261110.json
.venv/bin/python scripts/qualify_call_confirmation.py --config call-prior-confirmation-20261110.json
wait "$call_observer_pid"
