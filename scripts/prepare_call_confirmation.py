"""Freeze a validation-selected call candidate for untouched-wall confirmation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    source_cfg = json.loads((ROOT / cfg['source_config']).read_text())
    source = ROOT / cfg['source_run']; out = ROOT / cfg['out']
    if source_cfg['out'] != cfg['source_run']: raise ValueError('Source run mismatch')
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_call_development.py'),
                    '--config', str(ROOT / cfg['source_config'])], cwd=ROOT, check=True)
    audit = json.loads((source / 'development-audit.json').read_text())
    if not audit['complete'] or audit['status'] != 'passed_complete': raise ValueError('Full source audit required')
    for suite in ('closed', 'mixed'):
        if audit['suites'][suite]['comparisons']['candidate_minus_reference']['gain'] <= 0:
            raise ValueError('Full development does not support the predeclared positive-mean replication route')
    selection = json.loads((source / 'selected.json').read_text()); candidate = ROOT / selection['checkpoint']
    if digest(candidate) != cfg['source_candidate_sha256'] or selection['checkpoint_sha256'] != cfg['source_candidate_sha256']:
        raise ValueError('Cannot change candidate using development outcomes')
    for k in ('reference', 'reference_head_sha256', 'previous_call_checkpoint', 'warm_start', 'game_mode', 'seed_scheme'):
        if cfg[k] != source_cfg[k]: raise ValueError(f'Protocol arm changed: {k}')
    if (cfg['evaluation_stage'] != 'independent_confirmation' or cfg['require_refit_improvement']
        or cfg['development_seed_start'] != source_cfg['confirmation_seed_start']
        or cfg['development_seeds'] != source_cfg['confirmation_seeds']):
        raise ValueError('Confirmation differs from reserved protocol')
    old = set(range(source_cfg['development_seed_start'], source_cfg['development_seed_start'] + source_cfg['development_seeds']))
    new = set(range(cfg['development_seed_start'], cfg['development_seed_start'] + cfg['development_seeds']))
    if old & new: raise ValueError('Development and confirmation walls overlap')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('selected.json', 'protocol.json'):
        shutil.copyfile(source / name, out / name)
    manifest = {'status': 'frozen_for_independent_confirmation', 'source_run': cfg['source_run'],
        'candidate_sha256': digest(candidate), 'source_development_audit_sha256': digest(source / 'development-audit.json'),
        'config_sha256': digest(args.config), 'source_config_sha256': digest(ROOT / cfg['source_config']),
        'training_protocol_copied_for_provenance_only': True, 'parameter_updates_in_confirmation': 0,
        'new_wall_clusters': len(new), 'both_suites_share_same_new_wall_clusters': True,
        'expected_games': len(new) * 4 * 3 * 2, 'human_test_opened': False,
        'primary': 'per-suite paired net points against confirmed incumbent; previous call150 secondary',
        'success': cfg['confirmation_success'], 'deployed': False}
    (out / 'confirmation-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__': main()
