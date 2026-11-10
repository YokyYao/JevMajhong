#!/bin/bash
#SBATCH --job-name=mahjong-unseen-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/clean-strength-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
evaluation_config="${1:-feature-unseen-development-20261110.json}"
bash feature_policy_evaluate.slurm "$evaluation_config" "${@:2}"
development_complete="$(.venv/bin/python - "$evaluation_config" <<'PY'
import json
from pathlib import Path
import sys
config = json.loads(Path(sys.argv[1]).read_text())
print('yes' if (Path(config['out']) / 'evaluation-audit.json').exists() else 'no')
PY
)"
if [ "$development_complete" = yes ]; then
    .venv/bin/python scripts/evaluate_feature_parent.py --config "$evaluation_config" "${@:2}"
    parent_complete="$(.venv/bin/python - "$evaluation_config" <<'PY'
import json
from pathlib import Path
import sys
config = json.loads(Path(sys.argv[1]).read_text())
print('yes' if (Path(config['paired_parent_out']) / 'summary.json').exists() else 'no')
PY
)"
    if [ "$parent_complete" = yes ]; then
        .venv/bin/python scripts/evaluate_feature_parent.py --config "$evaluation_config" --audit-only
    fi
fi
