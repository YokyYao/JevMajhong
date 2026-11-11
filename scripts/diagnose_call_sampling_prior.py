"""Audit completed fit controls and inspect a pre-registered sampling correction.

Uses existing TRAIN/validation predictions only; no GPU, parameter update,
threshold fit, test opening, candidate promotion or serving-policy mutation.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_fit_control import stratum
from call_fit_diagnostic import reaction_records, summarize_fit
from call_policy import digest
from call_sampling_prior import CLASSES, corrected_probabilities, inverse_inclusion
from checkpoint_identity import completed_checkpoint_files
from head_rl import write
from scripts.train_call_fit_control import verify


def registration(cfg, config_path):
    from jevany.data import load_records
    control_cfg = json.loads((ROOT / cfg['control_config']).read_text())
    prepared = json.loads((ROOT / control_cfg['out'] / 'preparation.json').read_text())
    verify(prepared)
    source_cfg = json.loads((ROOT / control_cfg['source_config']).read_text())
    source_path = ROOT / source_cfg['data'] / 'train.jsonl'
    sample_path = ROOT / control_cfg['data'] / 'balanced/train.jsonl'
    source_counts = Counter(stratum(r) for r in reaction_records(load_records(source_path), 'train'))
    sample_counts = Counter(stratum(r) for r in reaction_records(load_records(sample_path), 'train'))
    weights = inverse_inclusion(dict(source_counts), dict(sample_counts))
    files = [ROOT / cfg['control_config'], ROOT / control_cfg['out'] / 'preparation.json',
             source_path, sample_path, config_path.resolve(),
             *(ROOT / n for n in ('call_sampling_prior.py', 'scripts/diagnose_call_sampling_prior.py',
                                  'call_sampling_prior.slurm', 'call_fit_control.py',
                                  'call_fit_diagnostic.py', 'scripts/train_call_fit_control.py'))]
    return {'config': cfg, 'source_counts': dict(source_counts), 'sample_counts': dict(sample_counts),
            'fixed_inverse_inclusion_weights': weights,
            'frozen_files': {str(p.relative_to(ROOT)): digest(p) for p in files},
            'tuned_parameters': 0, 'optimizer_updates': 0, 'human_test_opened': False,
            'deployed': False}, control_cfg, prepared


def audit_arm(control_cfg, arm):
    import torch
    from jevany.data import load_records
    from jevany.benchmark import summarize
    out = ROOT / control_cfg['out'] / arm
    result = json.loads((out / 'result.json').read_text())
    if result['status'] != 'fit_control_complete' or result['strength_candidate']:
        raise ValueError('Completed non-promotable fit control required')
    curve = json.loads((out / 'curve.json').read_text())
    steps = result['optimizer_steps']
    if not 0 < steps <= 128 or steps % 16 or [r['step'] for r in curve] != list(range(0, steps + 1, 16)):
        raise ValueError('Complete registered validation curve required')
    datasets = {s: reaction_records(load_records(ROOT / control_cfg['data'] / arm / (s + '.jsonl')), s)
                for s in ('train', 'validation')}
    checked = 0
    evidence = [out / n for n in ('result.json', 'curve.json', 'initialization-audit.json', 'token-admission.json')]
    initial = json.loads((out / 'initialization-audit.json').read_text())
    if not initial['warm_start_exact_adapter_and_head'] or not initial['base_frozen']:
        raise ValueError('Exact frozen-base initialization audit required')
    for item in curve:
        directory = out / 'training-run/training_eval' / f"step-{item['step']:06d}"
        for split, subdir, nll_key in [('train', 'transfer', 'fit_selection_nll'),
                                       ('validation', 'development', 'full_validation_nll')]:
            rows = json.loads((directory / subdir / 'rows.json').read_text())
            evidence.append(directory / subdir / 'rows.json')
            checked += len(rows)
            ids = {r['_meta']['id'] for r in datasets[split]}
            if summarize_fit(datasets[split], [r for r in rows if r['id'] in ids]) != item['reaction_fit'][split]:
                raise ValueError('Fit metrics failed recomputation')
            if not math.isclose(summarize(rows)['clean']['nll'], item[nll_key], abs_tol=1e-12):
                raise ValueError('SDK selection/validation loss differs')
    best = min(curve, key=lambda r: r['fit_selection_nll'])
    selection = result['selection']
    if selection['step'] != best['step'] or selection['value'] != best['fit_selection_nll']:
        raise ValueError('Selection must use registered TRAIN fit loss')
    if selection['step'] == 0:
        raise ValueError('No updated fit checkpoint selected')
    run = out / 'training-run'
    metrics = json.loads((run / 'training_metrics.json').read_text())
    if metrics['optimizer_steps'] != steps or metrics['records_seen'] != steps * 16 or metrics['admitted_records'] != 512 or metrics['rejected_records']:
        raise ValueError('Actual optimization budget differs')
    state = torch.load(run / f'checkpoints/step-{steps:06d}/trainer_state.pt', map_location='cpu', weights_only=True)
    if (state['records_seen'] != steps * 16 or state['optimizer_step'] != steps or
            state['world_size'] != 1 or state['epoch'] * 512 + state['next_microbatch'] != steps * 16 or
            sorted(state['order'].tolist()) != list(range(512))):
        raise ValueError('Saved exposure cursor differs')
    identity = completed_checkpoint_files(run / selection['selected_checkpoint'], selection['step'])
    completed_checkpoint_files(run / f'checkpoints/step-{steps:06d}', steps)
    evidence += [run / 'training_metrics.json', run / 'selection.json',
                 run / selection['selected_checkpoint'] / 'trainer_state.pt',
                 run / f'checkpoints/step-{steps:06d}' / 'trainer_state.pt']
    audit = {'status': 'passed_complete', 'arm': arm, 'optimizer_steps': steps,
             'request_exposures': steps * 16, 'unique_requests': 512,
             'all_prediction_rows_recomputed': checked, 'selected_step': selection['step'],
             'selected_files': identity,
             'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in evidence},
             'strength_candidate': False}
    return audit, datasets, out / 'training-run/training_eval' / f"step-{selection['step']:06d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol, control_cfg, prepared = registration(cfg, args.config)
    out = ROOT / cfg['out']
    if args.prepare:
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'protocol.json', protocol)
        write(out / 'status.json', {'status': 'registered_waiting_completed_fit_controls'})
        print(json.dumps({'registered': True, 'weights': protocol['fixed_inverse_inclusion_weights']}))
        return
    if json.loads((out / 'protocol.json').read_text()) != protocol or (out / 'summary.json').exists():
        raise ValueError('Changed or already-completed correction diagnostic')
    audits, original, corrected = {}, {}, {}
    for arm in ('natural', 'balanced'):
        audit, datasets, directory = audit_arm(control_cfg, arm)
        audits[arm] = audit
        original[arm] = {}
        for split, subdir in [('train', 'transfer'), ('validation', 'development')]:
            records = datasets[split]
            lookup = {r['_meta']['id']: r for r in records}
            rows = [r for r in json.loads((directory / subdir / 'rows.json').read_text()) if r['id'] in lookup]
            original[arm][split] = summarize_fit(records, rows)
            if arm == 'balanced':
                adjusted = []
                for row in rows:
                    q = lookup[row['id']]['questions']['action']
                    probabilities = corrected_probabilities(q['criteria'], dict(zip(row['keys'], row['p'])),
                                                           protocol['fixed_inverse_inclusion_weights'])
                    adjusted.append({**row, 'p': [probabilities[k] for k in row['keys']]})
                corrected[split] = summarize_fit(records, adjusted)
    verify(prepared)
    if registration(cfg, args.config)[0] != protocol:
        raise ValueError('Correction diagnostic inputs changed')
    result = {'status': 'passed_complete', 'control_audits': audits,
              'fixed_inverse_inclusion_weights': protocol['fixed_inverse_inclusion_weights'],
              'original_selected_fit_models': original, 'balanced_fixed_correction': corrected,
              'selection_was_train_only': True, 'tuned_parameters': 0,
              'calibrated_sampled_posterior_assumption_not_proven': True,
              'not_strength_evidence': True, 'strength_candidate': False,
              'human_test_opened': False, 'deployed': False}
    write(out / 'summary.json', result)
    write(out / 'status.json', {'status': 'passed_complete', 'control_audits': audits,
                              'tuned_parameters': 0, 'strength_candidate': False})
    print(json.dumps({'status': 'passed_complete', 'arms': {
        arm: {'optimizer_steps': a['optimizer_steps'], 'selected_step': a['selected_step'],
              'all_prediction_rows_recomputed': a['all_prediction_rows_recomputed']}
        for arm, a in audits.items()}}), flush=True)


if __name__ == '__main__':
    main()
