"""Freeze two TRAIN fitting pilots after the completed underlearning diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_fit_control import choose_control
from call_fit_diagnostic import reaction_records, summarize_fit
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from feature_continuation import verify_preparation
from head_rl import write
from scripts.diagnose_call_training_fit import registration as diagnostic_registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out, data = ROOT / cfg['out'], ROOT / cfg['data']
    if out.exists() or data.exists():
        raise ValueError('Fresh immutable pilot directories required')
    diag_cfg = json.loads((ROOT / cfg['diagnostic_config']).read_text())
    diag_protocol, diag_records = diagnostic_registration(diag_cfg)
    diag = ROOT / diag_cfg['out']
    summary = json.loads((diag / 'summary.json').read_text())
    if summary['status'] != 'passed_complete' or summary['optimizer_updates'] != 0 or summary['human_test_opened']:
        raise ValueError('Completed read-only diagnostic required')
    if json.loads((diag / 'protocol.json').read_text()) != diag_protocol:
        raise ValueError('Diagnostic identity changed')
    evidence = [diag / n for n in ('protocol.json', 'summary.json')]
    for model in diag_cfg['models']:
        if summary['summaries'][model]['validation_parity']['checked'] != 1024:
            raise ValueError('Full validation parity required')
        for split in ('train', 'validation'):
            path = diag / (model + '-' + split + '-private.jsonl')
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            if summarize_fit(diag_records[split], rows) != summary['summaries'][model][split]:
                raise ValueError('Diagnostic statistics changed')
            evidence.append(path)
    source_cfg = json.loads((ROOT / cfg['source_config']).read_text())
    parent_cfg = json.loads((ROOT / cfg['parent_config']).read_text())
    source = ROOT / source_cfg['out']
    parent = ROOT / parent_cfg['out']
    prepared = json.loads((source / 'preparation.json').read_text())
    verify_preparation(ROOT, source_cfg, prepared)
    result = json.loads((parent / 'result.json').read_text())
    if result['checkpoint'] != cfg['warm_start'] or result['selected_step'] != 200:
        raise ValueError('Preserved stage2 initialization required')
    from jevany.data import load_records
    from jevany.suite import load_split
    from reaction_selection import verify_benchmark_contract
    records = load_records(ROOT / source_cfg['data'] / 'train.jsonl')
    validation = load_records(ROOT / source_cfg['data'] / 'validation.jsonl')
    if len(records) != 4800 or len(validation) != 1408:
        raise ValueError('Original complete stage4 TRAIN/validation required')
    validation_groups = {r['_meta']['group_id'] for r in validation}
    if validation_groups & {r['_meta']['group_id'] for r in records}:
        raise ValueError('Whole-match TRAIN/validation overlap')
    # Freeze the explicit diagnostic-to-next-action chain, without altering any
    # source or historical manifest. Pilots reuse TRAIN; not new unseen data.
    evidence += [ROOT / cfg['source_config'], ROOT / cfg['parent_config'],
                 ROOT / cfg['diagnostic_config'], source / 'preparation.json',
                 parent / 'result.json', parent / 'preparation.json',
                 ROOT / source_cfg['data'] / 'train.jsonl',
                 ROOT / source_cfg['data'] / 'validation.jsonl']
    code = ['call_fit_control.py', 'call_fit_diagnostic.py',
            'scripts/prepare_call_fit_control.py', 'scripts/train_call_fit_control.py',
            'call_fit_control.slurm', 'vendor/JevAny/jevany/train.py',
            'vendor/JevAny/jevany/data.py', 'vendor/JevAny/jevany/model.py',
            'vendor/JevAny/jevany/backbones.py', 'vendor/JevAny/jevany/checkpoint.py',
            'vendor/JevAny/jevany/benchmark.py', 'vendor/JevAny/jevany/inference.py']
    recipes = {}
    for arm, path in cfg['recipes'].items():
        recipe = tomllib.loads((ROOT / path).read_text())
        expected = {'init_from': cfg['warm_start'], 'data': cfg['data'] + '/' + arm + '/train.jsonl',
            'out': cfg['out'] + '/' + arm + '/training-run',
            'eval_data': cfg['data'] + '/' + arm + '/validation.jsonl',
            'eval_transfer_suite': cfg['data'] + '/' + arm + '/fit-suite',
            'epochs': 4, 'max_steps': 128, 'batch': 1, 'accum': 16, 'lora': 16,
            'lr': .00002, 'head_lr': .0001, 'eval_every_steps': 16,
            'checkpoint_every_steps': 16, 'early_stop_patience': 3,
            'early_stop_metric': 'transfer_nll', 'p_none': 0., 'p_none_distract': 0.,
            'p_distract': 0., 'p_none_pair': 0., 'eval_records': 0,
            'seed': cfg['seed'], 'eval_before_start': True}
        if any(recipe.get(k) != v for k, v in expected.items()) or recipe.get('resume'):
            raise ValueError('Control recipe differs: ' + arm)
        recipes[arm] = recipe
    if set(recipes) != {'natural', 'balanced'}:
        raise ValueError('Both registered arms required')
    out.mkdir(parents=True)
    data.mkdir(parents=True)
    arms = {}
    for arm in cfg['recipes']:
        selected, counts = choose_control(records, arm, cfg['seed'])
        directory = data / arm
        directory.mkdir()
        for filename, rows in [('train.jsonl', selected), ('validation.jsonl', validation)]:
            with (directory / filename).open('w') as stream:
                for record in rows:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        suite = directory / 'fit-suite'
        suite.mkdir()
        path = suite / 'development.jsonl'
        path.write_bytes((directory / 'train.jsonl').read_bytes())
        write(suite / 'manifest.json', {'files': {'development.jsonl': {'sha256': digest(path), 'records': 512}},
            'holdout_sources': [], 'scope': 'TRAIN fitting probe, not held-out validation or strength selection'})
        if load_split(suite, 'development') != selected:
            raise ValueError('SDK fit-suite normalization differs')
        verify_benchmark_contract(selected)
        arms[arm] = {'records': 512, 'reaction_records': 448, 'ordinary_records': 64,
                     'train_strata': counts, 'unique_requests': 512, 'max_repeated_exposures': 2048}
    files = evidence + [ROOT / n for n in code] + [ROOT / p for p in cfg['recipes'].values()]
    files += [args.config.resolve()] + [p for p in data.rglob('*') if p.is_file()]
    protocol = {'config': cfg, 'arms': arms, 'warm_start_files': completed_checkpoint_files(ROOT / cfg['warm_start'], 200),
                'frozen_files': {str(p.relative_to(ROOT)): digest(p) for p in files},
                'selection': 'TRAIN fit NLL only, never promotable as a strength candidate',
                'new_unseen_requests': 0, 'human_test_opened': False, 'deployed': False}
    write(out / 'preparation.json', protocol)
    write(out / 'status.json', {'status': 'prepared_two_fit_controls', 'arms': arms})
    print(json.dumps({'prepared': True, 'arms': arms}), flush=True)


if __name__ == '__main__':
    main()
