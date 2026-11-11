"""Prepare a fresh coverage stage and a frozen reaction-only selection suite.

The unchanged preparer checks parent origin, exact exposure and whole-match
separation first. This entrypoint finalizes its outputs before training is
submitted. Any incomplete finalization explicitly invalidates the new run.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from feature_continuation import verify_preparation
from head_rl import write
from reaction_selection import reaction_subset, validate_recipe
from scripts.prepare_feature_continuation import main as prepare_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    recipe = tomllib.loads((ROOT / cfg['trainer_config']).read_text())
    validate_recipe(cfg, recipe)
    data, out = ROOT / cfg['data'], ROOT / cfg['out']
    evidence_files = []
    for field, metric in [('parent_confirmation_out', 'overall_paired_gain'),
                          ('parent_increment_out', 'overall_parent_paired_gain')]:
        directory = ROOT / cfg[field]
        audit = json.loads((directory / 'evaluation-audit.json').read_text())
        protocol = json.loads((directory / 'protocol.json').read_text())
        gain = audit[metric]['gain']
        if (audit['status'] != 'passed_complete' or not math.isfinite(gain) or gain <= 0
                or protocol['candidate_checkpoint'] != cfg['warm_start']):
            raise ValueError('Fully audited positive parent evidence required')
        raw_key = 'raw_parent_games_sha256' if field == 'parent_increment_out' else 'raw_games_sha256'
        if digest(directory / 'games-private.jsonl') != audit[raw_key]:
            raise ValueError('Parent evidence raw records changed')
        for name, expected in protocol['candidate_files'].items():
            if digest(ROOT / cfg['warm_start'] / name) != expected:
                raise ValueError('Parent evidence checkpoint differs')
        evidence_files.extend(directory / name for name in
                              ('evaluation-audit.json', 'protocol.json', 'summary.json', 'games-private.jsonl'))
    # Reuse every established parent/exposure check, without editing that source.
    prepare_coverage()
    try:
        prepared = json.loads((out / 'preparation.json').read_text())
        verify_preparation(ROOT, cfg, prepared)
        records = [json.loads(line) for line in (data / 'validation.jsonl').open()]
        reactions, actions = reaction_subset(records)
        suite = data / 'reaction-validation'
        suite.mkdir()
        path = suite / 'development.jsonl'
        with path.open('w') as stream:
            for record in reactions:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        write(suite / 'manifest.json', {
            'files': {'development.jsonl': {'sha256': digest(path), 'records': len(reactions)}},
            'holdout_sources': [],
            'scope': 'Exact reaction subset of already-opened validation; not independent test',
        })
        selection = {'scope': 'reaction', 'trainer_metric': 'transfer_nll',
            'records': len(reactions), 'actions': actions,
            'matches': len({r['_meta']['group_id'] for r in reactions}),
            'full_validation_records': len(records),
            'full_validation_retained_for_diagnostics_and_native_parity': True,
            'labels_and_options_unchanged': True, 'test_opened': False}
        prepared['selection'] = selection
        for p in (path, suite / 'manifest.json', ROOT / 'reaction_selection.py',
                  ROOT / 'scripts/prepare_reaction_continuation.py', *evidence_files):
            prepared['frozen_files'][str(p.relative_to(ROOT))] = digest(p)
        write(data / 'provenance.json', prepared)
        write(out / 'preparation.json', prepared)
        verify_preparation(ROOT, cfg, prepared)
        # Exercise the actual SDK checksum/count reader on the new selection suite.
        from jevany.suite import load_split
        if load_split(suite, 'development') != reactions:
            raise ValueError('SDK reaction selection data differs')
        print(json.dumps({'complete': True, 'records': prepared['records'],
                          'selection': selection}), flush=True)
    except Exception as exc:
        write(out / 'invalidation.json', {'reason': 'Reaction preparation incomplete',
              'error': str(exc), 'training_allowed': False, 'deployed': False})
        raise


if __name__ == '__main__':
    main()
