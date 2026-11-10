#!/bin/bash
#SBATCH --job-name=mahjong-confirm-parent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/confirmation-parent-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
config="${1:-feature-unseen-confirmation-parent-20261111.json}"
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/evaluate_feature_confirmation_parent.py --config "$config" "${@:2}"
# Nonpositive completed confirmation vsB skips parent games in the shared main.
.venv/bin/python - "$config" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
config = json.loads(Path(sys.argv[1]).read_text())
if (Path(config['paired_parent_out']) / 'summary.json').exists():
    subprocess.run([sys.executable, 'scripts/evaluate_feature_confirmation_parent.py',
                    '--config', sys.argv[1], '--audit-only'], check=True)
PY
