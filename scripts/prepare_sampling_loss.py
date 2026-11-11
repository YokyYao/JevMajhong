"""Prepare an isolated inclusion-aware likelihood experiment after fit audits."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from feature_continuation import verify_preparation
from head_rl import write
from scripts.diagnose_call_sampling_prior import registration, audit_arm
from scripts.train_call_fit_control import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    data, out = ROOT / cfg['data'], ROOT / cfg['out']
    if data.exists() or out.exists():
        raise ValueError('New immutable experiment directories required')
    study_cfg = json.loads((ROOT / cfg['sampling_diagnostic_config']).read_text())
    registered, controls, control_prepared = registration(study_cfg, ROOT / cfg['sampling_diagnostic_config'])
    study = ROOT / study_cfg['out']
    if json.loads((study / 'protocol.json').read_text()) != registered:
        raise ValueError('Sampling diagnostic registration changed')
    summary = json.loads((study / 'summary.json').read_text())
    if summary['status'] != 'passed_complete' or summary['tuned_parameters'] or not summary['not_strength_evidence']:
        raise ValueError('Complete fixed diagnostic required')
    for arm in ('natural', 'balanced'):
        actual, _, _ = audit_arm(controls, arm)
        if actual != summary['control_audits'][arm]:
            raise ValueError('Complete fit-control evidence changed')
    verify(control_prepared)
    parent_cfg = json.loads((ROOT / cfg['parent_config']).read_text())
    parent = ROOT / parent_cfg['out']
    selected = json.loads((parent / 'result.json').read_text())
    if selected['checkpoint'] != cfg['warm_start'] or selected['selected_step'] != cfg['parent_step']:
        raise ValueError('Preserved stage2 parent required')
    parent_prepared = json.loads((parent / 'preparation.json').read_text())
    verify_preparation(ROOT, parent_cfg, parent_prepared)
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    reference = json.loads((ROOT / cfg['reference_protocol']).read_text())['reference_files']
    if reference != parent_protocol['reference_files']:
        raise ValueError('Frozen oldB reference differs')
    recipe = tomllib.loads((ROOT / cfg['trainer_config']).read_text())
    expected = {'init_from': cfg['warm_start'], 'data': cfg['data'] + '/train.jsonl',
                'eval_data': cfg['data'] + '/validation.jsonl', 'out': cfg['out'] + '/training-run',
                'eval_transfer_suite': cfg['data'] + '/reaction-validation',
                'epochs': 4, 'batch': 1, 'accum': 16, 'max_steps': 128, 'lora': 16,
                'lr': .00002, 'head_lr': .0001, 'seed': cfg['seed'],
                'eval_every_steps': 16, 'checkpoint_every_steps': 16,
                'early_stop_metric': 'transfer_nll', 'early_stop_patience': 3,
                'eval_before_start': True, 'eval_records': 0, 'p_none': 0.,
                'p_none_distract': 0., 'p_distract': 0., 'p_none_pair': 0.}
    if any(recipe.get(k) != v for k, v in expected.items()) or recipe.get('resume') or recipe.get('rlcr'):
        raise ValueError('Registered inclusion-likelihood recipe differs')
    data.mkdir(parents=True)
    out.mkdir(parents=True)
    for split in ('train', 'validation'):
        shutil.copyfile(ROOT / controls['data'] / 'balanced' / (split + '.jsonl'), data / (split + '.jsonl'))
    # Exact already-opened validation, normalized through the SDK before writing.
    from jevany.data import load_records
    from jevany.suite import load_split
    from reaction_selection import reaction_subset, verify_benchmark_contract
    validation = load_records(data / 'validation.jsonl')
    reactions, _ = reaction_subset(validation)
    suite = data / 'reaction-validation'
    suite.mkdir()
    path = suite / 'development.jsonl'
    with path.open('w') as f:
        for r in reactions:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    write(suite / 'manifest.json', {'files': {'development.jsonl': {'records': 1024, 'sha256': digest(path)}},
          'holdout_sources': [], 'scope': 'Already-opened reaction validation, raw probabilities without inclusion offset'})
    train = load_records(data / 'train.jsonl')
    if len(train) != 512 or len(validation) != 1408 or load_split(suite, 'development') != reactions:
        raise ValueError('Full unchanged dataset/selection contract differs')
    if {r['_meta']['group_id'] for r in train} & {r['_meta']['group_id'] for r in validation}:
        raise ValueError('Whole-match TRAIN/validation overlap')
    verify_benchmark_contract(reactions)
    warm_files = completed_checkpoint_files(ROOT / cfg['warm_start'], cfg['parent_step'])
    frozen = set(control_prepared['frozen_files']) | set(registered['frozen_files'])
    frozen.update([str(args.config.resolve().relative_to(ROOT)), cfg['trainer_config'],
        'sampling_loss.py', 'scripts/prepare_sampling_loss.py', 'scripts/train_sampling_loss.py',
        'sampling_loss.slurm', cfg['parent_config'], cfg['reference_protocol'],
        str((study / 'summary.json').relative_to(ROOT)), str((study / 'protocol.json').relative_to(ROOT))])
    for audit in summary['control_audits'].values():
        frozen.update(audit['source_sha256'])
    frozen.update(str(p.relative_to(ROOT)) for p in data.rglob('*') if p.is_file())
    frozen.update(str((parent / n).relative_to(ROOT)) for n in ('protocol.json', 'preparation.json', 'result.json'))
    frozen.update(str((ROOT / cfg['warm_start'] / n).relative_to(ROOT)) for n in warm_files)
    frozen.update(str((ROOT / cfg['reference'] / n).relative_to(ROOT)) for n in reference)
    prepared = {'complete': True, 'config': cfg, 'model_test_opened': False,
        'human_test_opened': False, 'deployed': False,
        'records': {'train': 512, 'validation': 1408},
        'files': {s: digest(data / (s + '.jsonl')) for s in ('train', 'validation')},
        'warm_start_files': warm_files, 'reference_files': reference,
        'inverse_inclusion_weights': registered['fixed_inverse_inclusion_weights'],
        'new_unseen_requests': 0, 'training_source': 'Exact balanced fit-control subset, reused TRAIN, no new matches',
        'frozen_files': {n: digest(ROOT / n) for n in sorted(frozen)}}
    write(data / 'provenance.json', prepared)
    write(out / 'preparation.json', prepared)
    verify_preparation(ROOT, cfg, prepared)
    print(json.dumps({'prepared': True, 'records': prepared['records'],
          'inverse_inclusion_weights': prepared['inverse_inclusion_weights']}), flush=True)


if __name__ == '__main__':
    main()
