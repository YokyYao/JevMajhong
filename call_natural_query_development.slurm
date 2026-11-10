#!/bin/bash
#SBATCH --job-name=mahjong-natural-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/call-natural-dev-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/watch_call_repair.py --config call-natural-query-20261110.json --eval-job "$SLURM_JOB_ID" --max-seconds 14400 &
call_natural_observer_pid=$!
stop_call_natural_observer() {
    local call_natural_exit=$?
    trap - EXIT
    kill "$call_natural_observer_pid" 2>/dev/null || true
    wait "$call_natural_observer_pid" 2>/dev/null || true
    exit "$call_natural_exit"
}
trap stop_call_natural_observer EXIT
.venv/bin/python scripts/evaluate_call_repair.py --config call-natural-query-20261110.json
if [[ ! -f runs/call-natural-query-20261110/development-skipped.json ]]; then
    .venv/bin/python scripts/audit_call_development.py --config call-natural-query-20261110.json
fi
wait "$call_natural_observer_pid"
