"""Qualification uses only complete, audited, independently paired match scores."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest


def qualify(audit, summaries):
    if audit.get('status') != 'passed_complete' or not audit.get('complete'):
        raise ValueError('Incomplete or failed audit')
    if set(summaries) != {'closed', 'mixed'}: raise ValueError('Both opponent suites required')
    result = {}
    for name, summary in summaries.items():
        if not summary.get('is_final_confirmation'): raise ValueError('Development cannot qualify confirmation')
        primary = audit['suites'][name]['comparisons']['candidate_minus_reference']
        mean = summary['policies']['candidate']['mean_net_points']
        result[name] = {'paired_gain': primary['gain'], 'paired_ci95': primary['ci95'],
                        'mean_net_points': mean, 'qualified': primary['ci95'][0] > 0 and mean >= 0}
    return {'qualified': all(r['qualified'] for r in result.values()), 'suites': result,
            'criterion': 'Both fixed opponent suites: paired CI95 lower>0 and candidate mean net points>=0',
            'scope': 'Fixed four-player East simulation; not a Mahjong Soul ranked or screenshot-strength claim',
            'deployed': False, 'fresh_human_test_and_desktop_regressions_pending': True}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); run = ROOT / cfg['out']
    if cfg['evaluation_stage'] != 'independent_confirmation': raise ValueError('Confirmation config required')
    audit = json.loads((run / 'development-audit.json').read_text())
    if audit['candidate_sha256'] != cfg['source_candidate_sha256']: raise ValueError('Candidate identity changed')
    summaries = {name: json.loads((run / 'native-development' / f'{name}-summary.json').read_text()) for name in ('closed', 'mixed')}
    result = qualify(audit, summaries)
    result.update(candidate_sha256=audit['candidate_sha256'], audit_sha256=digest(run / 'development-audit.json'))
    path = run / 'qualification.json'; temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n'); temp.replace(path)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
