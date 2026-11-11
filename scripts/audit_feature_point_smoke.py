"""Mechanism gate for stage2 counterfactual collection; no score-based selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest
from feature_point_collector import SUITES
from head_rl import write
from scripts.collect_feature_points import assert_frozen, validate_wall


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    if cfg['purpose_role'] != 'mechanism_smoke' or cfg['training_seeds_per_stage'] != 3:
        raise ValueError('Expected the registered three-suite smoke')
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['config'] != cfg: raise ValueError('Smoke config changed')
    assert_frozen(protocol, cfg)
    result = json.loads((out / 'collection-result.json').read_text())
    if result['status'] != 'collection_complete' or result['walls'] != 3:
        raise ValueError('Smoke collection incomplete')
    parity = json.loads((out / 'teacher-native-parity.json').read_text())
    if (parity['checked'] != {'reaction': 6, 'ordinary': 6}
            or not parity['native_decisions_equal'] or not parity['ordinary_probabilities_unchanged']
            or parity['maximum_probability_error'] > 2e-5):
        raise ValueError('Stage2 native/ordinary parity missing')
    suite_rows = {}; rows = branches = controls = 0
    for seed in range(cfg['training_seed_start'], cfg['training_seed_start'] + 3):
        shard = out / 'walls-private' / f'{seed}.pt'
        audit = json.loads(shard.with_suffix('.audit.json').read_text())
        if digest(shard) != audit['sha256'] or not audit['ledger_passed']:
            raise ValueError('Smoke shard integrity/ledger failed')
        payload = torch.load(shard, map_location='cpu', weights_only=False)
        validate_wall(payload, seed, cfg)
        if audit['native_probability_checks_cumulative'] <= 0 or not payload['rows']:
            raise ValueError('Missing live call capture in a smoke suite')
        rows += len(payload['rows']); controls += len(payload['controls'])
        branches += sum(not g['reused_control_outcome'] for g in payload['games'])
        suite_rows[audit['suite']] = len(payload['rows'])
    if set(suite_rows) != set(SUITES) or rows != result['rows'] or branches != result['actual_branches']:
        raise ValueError('Smoke coverage differs from completed collection')
    audit = {'status': 'passed_mechanism_smoke', 'controls': controls, 'changed_action_rows': rows,
        'actual_branches': branches, 'suite_rows': suite_rows, 'teacher_native_parity': parity,
        'all_branch_ledger_and_exact_prefix_checks_passed': True,
        'reward_or_strength_selected': False, 'strength_improvement_proven': False}
    write(out / 'mechanism-smoke-audit.json', audit); print(json.dumps(audit), flush=True)


if __name__ == '__main__': main()
