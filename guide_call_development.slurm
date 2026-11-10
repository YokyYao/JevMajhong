#!/bin/bash
#SBATCH --job-name=mahjong-guide-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/guide-call-dev-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
guide_dev_config="${1:-guide-call-development-20261110.json}"
.venv/bin/python scripts/watch_guide_call_points.py --config "$guide_dev_config" --job "$SLURM_JOB_ID" &
guide_dev_observer_pid=$!
stop_guide_dev_observer() {
    local guide_dev_exit=$?
    trap - EXIT
    kill "$guide_dev_observer_pid" 2>/dev/null || true
    wait "$guide_dev_observer_pid" 2>/dev/null || true
    exit "$guide_dev_exit"
}
trap stop_guide_dev_observer EXIT
.venv/bin/python scripts/evaluate_guide_calls.py --config "$guide_dev_config"
guide_dev_run=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["out"])' "$guide_dev_config")
if [[ ! -f "$guide_dev_run/development-skipped.json" ]]; then
    .venv/bin/python scripts/audit_guide_calls.py --config "$guide_dev_config"
fi
