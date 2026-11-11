"""Independently recompute the complete point-value versus frozen-stage2 grid."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from clean_policy_evaluation import aggregate
from head_rl import write
from scripts.evaluate_feature_point_value import verify_protocol
from scripts.collect_feature_points import assert_frozen


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    skipped = out / 'evaluation-skipped.json'
    if skipped.exists():
        reason = json.loads(skipped.read_text())['reason']
        if reason not in ('point_predictor_not_improved', 'insufficient_wall_coverage',
                'development_skipped', 'nonpositive_parent_development'):
            raise ValueError('Unknown skip reason')
        source_cfg = json.loads((ROOT / cfg['source_config']).read_text())
        run = ROOT / source_cfg['out']; source = json.loads((run / 'protocol.json').read_text())
        if source['config'] != source_cfg: raise ValueError('Skipped training source changed')
        assert_frozen(source, source_cfg)
        fit = json.loads((run / 'fit-result.json').read_text())
        if reason in ('point_predictor_not_improved', 'insufficient_wall_coverage'):
            if fit['status'] != reason: raise ValueError('Skip does not match actual fit result')
        else:
            if cfg['stage'] != 'independent_confirmation' or fit['status'] != 'point_predictor_fit':
                raise ValueError('Invalid conditional confirmation skip')
            dev = ROOT / cfg['development_out']
            if reason == 'development_skipped':
                if not (dev / 'evaluation-skipped.json').exists(): raise ValueError('Development was not skipped')
            else:
                audit = json.loads((dev / 'evaluation-audit.json').read_text())
                if audit['status'] != 'passed_complete' or audit['overall_paired_gain']['gain'] > 0:
                    raise ValueError('Development was not completed nonpositive')
        write(out / 'status.json', {'status': 'evaluation_skipped', 'reason': reason, 'deployed': False}); return
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['config'] != cfg: raise ValueError('Point evaluation config changed')
    verify_protocol(protocol)
    parity = json.loads((out / 'native-parity.json').read_text())
    if (parity['distinct_train_calls'] != cfg['native_parity_requests']
            or parity['ordinary_requests_unchanged'] != 6
            or not parity['cpu_native_decisions_equal'] or not parity['teacher_relative_value_exactly_zero']
            or parity['maximum_teacher_probability_error'] > 2e-5
            or parity['maximum_cpu_native_value_error'] > cfg['point_value_parity_atol']):
        raise ValueError('Point native parity incomplete')
    raw = out / 'games-private.jsonl'; rows = [json.loads(l) for l in raw.read_text().splitlines()]
    recomputed = aggregate(rows, cfg); saved = json.loads((out / 'summary.json').read_text())
    if recomputed != saved: raise ValueError('Point complete-grid aggregate differs')
    audit = {'status': 'passed_complete', 'games': len(rows), 'stage': cfg['stage'],
        'independent_wall_streams': cfg['seeds'],
        'overall_paired_gain': recomputed['overall']['candidate_minus_reference'],
        'raw_games_sha256': digest(raw), 'candidate_frozen': True,
        'source_identity_passed': True, 'native_parity_passed': True,
        'summary_recomputed': True, 'human_test_opened': False, 'deployed': False}
    write(out / 'evaluation-audit.json', audit); write(out / 'status.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__': main()
