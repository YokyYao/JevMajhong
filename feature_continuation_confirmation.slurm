#!/bin/bash
#SBATCH --job-name=mahjong-continue-confirm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/continuation-confirmation-gate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
confirmation_config="${1:-feature-unseen-confirmation-20261110.json}"
# afterok must depend on development plus the matched-parent audit, not TRAIN.
gate_decision="$(.venv/bin/python scripts/check_continuation_confirmation.py \
    --config "$confirmation_config" \
    --report "logs/continuation-confirmation-${SLURM_JOB_ID:?}.json")"
case "$gate_decision" in
    confirm)
        exec bash feature_policy_evaluate.slurm "$confirmation_config"
        ;;
    skip_nonpositive_oldB|skip_nonpositive_parent)
        echo "Complete audited comparison is nonpositive: $gate_decision; preserve confirmed parent."
        ;;
    *)
        echo 'Unexpected continuation confirmation decision' >&2
        exit 1
        ;;
esac
