"""Recompute checkpoint ranking, shared parent games and fresh confirmation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from fresh_score_curve import summarize
from head_rl import write
from scripts.evaluate_fresh_score_curve import registration, verify_protocol


def check_parity(path):
    d = json.loads(path.read_text())
    if (set(d['native_counts'].values()) != {6} or d['maximum_probability_error'] > 2e-5
            or not d['full_stage2_decisions_equal_to_saved_validation']
            or not d['hybrid_call_probabilities_equal_full_stage2']
            or d['hybrid_ordinary_probabilities_equal_oldB'] != 12):
        raise ValueError('Score-curve native parity incomplete')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    if (out / 'evaluation-skipped.json').exists():
        registration(cfg)
        if cfg['stage'] != 'independent_confirmation': raise ValueError('Only conditional confirmation may skip')
        audit = json.loads((ROOT / cfg['development_out'] / 'evaluation-audit.json').read_text())
        if audit['status'] != 'passed_complete' or audit['confirmation_step'] is not None:
            raise ValueError('Invalid confirmation skip')
        write(out / 'status.json', {'status': 'evaluation_skipped', 'reason': 'no_positive_parent_gain', 'deployed': False}); return
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['config'] != cfg: raise ValueError('Score-curve config changed')
    verify_protocol(protocol)
    steps = ([protocol['selected_confirmation_step']] if cfg['stage'] == 'independent_confirmation'
        else sorted(int(s) for s in protocol['checkpoints']))
    check_parity(out / 'parent-native-parity.json')
    for step in steps: check_parity(out / f'native-parity-{step:06d}.json')
    raw = out / 'games-private.jsonl'; rows = [json.loads(l) for l in raw.read_text().splitlines()]
    actual = summarize(rows, cfg, steps); saved = json.loads((out / 'summary.json').read_text())
    if actual != saved: raise ValueError('Full score-curve grid/selection differs from recomputation')
    best = actual['best_step']
    result = {'status': 'passed_complete', 'stage': cfg['stage'], 'games': len(rows),
        'independent_wall_streams': cfg['seeds'], 'scored_steps': steps,
        'parent_games_computed_once': actual['parent_games_computed_once'],
        'best_step': best, 'confirmation_step': actual['confirmation_step'],
        'evaluated_confirmation_step': actual['evaluated_confirmation_step'],
        'overall_paired_gain': actual['comparisons'][str(best)]['overall']['candidate_minus_reference'],
        'checkpoint_gains': {str(s): actual['comparisons'][str(s)]['overall']['candidate_minus_reference'] for s in steps},
        'raw_games_sha256': digest(raw), 'complete_grid_ledger_and_selection_recomputed': True,
        'candidate_and_parent_files_frozen': True, 'native_parity_passed': True,
        'human_test_opened': False, 'deployed': False}
    write(out / 'evaluation-audit.json', result); write(out / 'status.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
