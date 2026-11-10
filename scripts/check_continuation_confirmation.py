"""Check both frozen comparisons before loading a continuation for confirmation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from continuation_confirmation_gate import confirmation_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    decision = confirmation_decision(ROOT, config)
    report = {**decision, 'confirmation_config': config,
        'gate_code_sha256': {p: digest(ROOT / p) for p in (
            'continuation_confirmation_gate.py', 'scripts/check_continuation_confirmation.py',
            'feature_continuation_confirmation.slurm')},
        'test_opened': False, 'deployed': False}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(decision['decision'])


if __name__ == '__main__':
    main()
