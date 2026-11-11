"""Recompute complete three-arm ablation statistics from private full-match rows."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from guide_development_audit import aggregate
from head_rl import write
from scripts.evaluate_stage2_ablation import verify_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    protocol = json.loads((out / 'protocol.json').read_text())
    if cfg != protocol['config']: raise ValueError('Ablation config changed')
    verify_protocol(protocol)
    parity = json.loads((out / 'native-parity.json').read_text())
    if (set(parity['native_counts'].values()) != {6} or parity['maximum_probability_error'] > 2e-5
            or not parity['full_stage2_decisions_equal_to_saved_validation']
            or not parity['hybrid_call_probabilities_equal_full_stage2']
            or parity['hybrid_ordinary_probabilities_equal_oldB'] != 12):
        raise ValueError('Native parity incomplete')
    raw = out / 'games-private.jsonl'; rows = [json.loads(l) for l in raw.read_text().splitlines()]
    actual = aggregate(rows, cfg); saved = json.loads((out / 'summary.json').read_text())
    for key in ('games', 'suites', 'overall'):
        if actual[key] != saved[key]: raise ValueError('Recomputed full-game statistics differ')
    result = {'status': 'passed_complete', 'games': len(rows), 'whole_wall_streams': cfg['development_seeds'],
        'primary_full_minus_hybrid': actual['overall']['candidate_minus_previous'],
        'full_minus_oldB': actual['overall']['candidate_minus_reference'],
        'raw_sha256': digest(raw), 'ledger_and_complete_grid_recomputed': True,
        'policy_source_and_weights_frozen': True, 'human_test_opened': False, 'deployed': False}
    write(out / 'evaluation-audit.json', result); write(out / 'status.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
