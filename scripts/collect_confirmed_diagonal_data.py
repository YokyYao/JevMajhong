"""Expand independent training data under the confirmed preserved policy."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from head_rl import fingerprint
from scripts.audit_score_confirmation import audit


def validate_confirmation(config, evidence, checkpoint_sha):
    if evidence['audit_status'] != 'passed' or evidence['qualified'] is not True:
        raise ValueError('Use a completed, independently audited positive source')
    if evidence['primary_delta_ci95'][0] <= 0 or evidence['candidate']['mean_net_points'] < 0:
        raise ValueError('Source does not meet its declared point-improvement criterion')
    if not evidence['training_and_evaluation_streams_disjoint']:
        raise ValueError('Source confirmation is not independent of training')
    if checkpoint_sha != evidence['candidate_head_sha256'] or checkpoint_sha != config['frozen_teacher_head_sha256']:
        raise ValueError('Collection teacher differs from confirmed source')
    if config['actor_lr'] != 0 or config['updates_per_stage'] != 0:
        raise ValueError('Collect the whole batch with a frozen teacher')


def check(config):
    source = ROOT / config['source_confirmation_run']
    evidence = audit(source, config['frozen_teacher_head_sha256'])
    checkpoint_sha = fingerprint(ROOT / config['reference'] / 'head.pt')
    validate_confirmation(config, evidence, checkpoint_sha)
    return {'status': 'confirmed_source_verified', 'source': str(source),
            'head_sha256': checkpoint_sha, 'source_point_gain': evidence['primary_delta'],
            'source_ci95': evidence['primary_delta_ci95'], 'actor_updates_during_collection': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    print(json.dumps(check(config)), flush=True)
    if not args.check_only:
        subprocess.run([sys.executable, str(ROOT / 'scripts/collect_diagonal_data.py'),
                        '--config', str(args.config.resolve())], cwd=ROOT, check=True)
