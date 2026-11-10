#!/bin/bash
#SBATCH --job-name=mahjong-feature-confirm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/feature-confirmation-gate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
confirmation_config="${1:-clean-feature-confirmation-20261110.json}"

# Submit with afterok on the complete development evaluator, including its audit.
# The existing evaluator separately enforces identical frozen candidate weights.
gate_decision="$(.venv/bin/python - "$confirmation_config" <<'PY'
import json
import math
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
if config['stage'] != 'independent_confirmation':
    raise ValueError('This job only admits independent confirmation')
development = Path(config['development_out'])
audit = json.loads((development / 'evaluation-audit.json').read_text())
protocol = json.loads((development / 'protocol.json').read_text())
if audit['status'] != 'passed_complete' or audit['stage'] != 'development':
    raise ValueError('Completed development audit required')
if protocol['config']['source_config'] != config['source_config']:
    raise ValueError('Confirmation and development training sources differ')
gain = audit['overall_paired_gain']['gain']
if not isinstance(gain, (int, float)) or not math.isfinite(gain):
    raise ValueError('Finite paired mean gain required')
print('confirm' if gain > 0 else 'skip_nonpositive')
PY
)"
case "$gate_decision" in
    confirm)
        exec bash feature_policy_evaluate.slurm "$confirmation_config"
        ;;
    skip_nonpositive)
        echo 'Completed development gain is nonpositive; confirmation skipped before model loading.'
        ;;
    *)
        echo 'Unexpected confirmation decision' >&2
        exit 1
        ;;
esac
