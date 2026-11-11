"""Recover only a proven post-training relative-path export failure.

Preserve the original frozen trainer, failure and checkpoints. Recompute all
validation, selection, source and exposure evidence before writing result.json.
No optimization, source-manifest rewrite, new selection or inference occurs.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from encoder_training_guard import exact_metrics
from feature_continuation import UpdatedFeatureSkillGuard, verify_preparation
from head_rl import write
from sampling_finalization import canonical_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = ROOT / cfg['out']
    run = out / 'training-run'
    if (out / 'result.json').exists() or (out / 'result-finalization-audit.json').exists():
        raise ValueError('Refuse overwriting a finalized result')
    prepared = json.loads((out / 'preparation.json').read_text())
    verify_preparation(ROOT, cfg, prepared)
    failure = json.loads((out / 'failure.json').read_text())
    selected = json.loads((run / 'selection.json').read_text())
    relative_run = Path(cfg['out']) / 'training-run'
    expected_error = f"'{relative_run / selected['selected_checkpoint']}' is not in the subpath of '{ROOT}'"
    if failure['type'] != 'ValueError' or failure['error'] != expected_error:
        raise ValueError('Only the exact final path-normalization failure is recoverable')
    from jevany.data import load_records
    from jevany.benchmark import summarize
    from jevany.train import early_stop_update
    import torch
    data = ROOT / cfg['data']
    records = load_records(data / 'validation.jsonl')
    train = load_records(data / 'train.jsonl')
    metrics = json.loads((run / 'training_metrics.json').read_text())
    steps = metrics['optimizer_steps']
    curve = json.loads((out / 'validation-curve.json').read_text())
    if (not 0 < steps <= cfg['max_steps'] or steps % cfg['validation_every']
            or [r['step'] for r in curve] != list(range(0, steps + 1, cfg['validation_every']))
            or metrics['admitted_records'] != len(train) or len(train) != 512
            or metrics['rejected_records'] or metrics['global_effective_batch'] != 16):
        raise ValueError('Complete registered training/validation budget required')
    guard = UpdatedFeatureSkillGuard(cfg['views'])
    best, stale, best_step, checked = None, 0, None, 0
    evidence = [out / n for n in ('protocol.json', 'preparation.json', 'failure.json',
        'validation-curve.json', 'loss-hook-audit.json', 'initialization-audit.json', 'baseline-parity.json')]
    for item in curve:
        directory = run / 'training_eval' / f"step-{item['step']:06d}"
        path = directory / 'development/rows.json'
        rows = json.loads(path.read_text())
        evidence.append(path)
        checked += len(rows)
        actual = exact_metrics(records, rows)
        if actual != item['views'] or not math.isclose(summarize(rows)['clean']['nll'], item['nll'], abs_tol=1e-12):
            raise ValueError('Full raw validation recomputation failed')
        for phase in ('closed_discard', 'open_discard', 'reaction'):
            subset = [r for r in records if r['_meta']['phase'] == phase]
            ids = {r['_meta']['id'] for r in subset}
            if exact_metrics(subset, [r for r in rows if r['id'] in ids]) != item['by_phase'][phase]:
                raise ValueError('Phase validation recomputation failed')
        path = directory / 'transfer/rows.json'
        reaction_rows = json.loads(path.read_text())
        evidence.append(path)
        checked += len(reaction_rows)
        if (len(reaction_rows) != 1024 or
                not math.isclose(summarize(reaction_rows)['clean']['nll'], item['reaction_selection_nll'], abs_tol=1e-12)
                or not math.isclose(item['by_phase']['reaction']['native']['nll'], item['reaction_selection_nll'], abs_tol=1e-12)
                or item['sampling_offsets_applied_to_validation']):
            raise ValueError('Unadjusted reaction selection differs')
        guard.step = item['step']
        if guard.assess(actual) != item['guard']:
            raise ValueError('Checkpoint admission differs')
        best, stale, improved = guard.update(early_stop_update, item['reaction_selection_nll'], best, stale, maximize=False)
        if improved:
            best_step = item['step']
        if stale >= cfg['patience'] and item['step'] != steps:
            raise ValueError('Training went beyond registered early stop')
    if (best_step != selected['step'] or best != selected['value']
            or metrics['early_stop_best_step'] != best_step or metrics['early_stop_best'] != best
            or metrics['early_stop_stale_evaluations'] != stale
            or (steps < cfg['max_steps'] and not (metrics['early_stopped'] and stale >= cfg['patience']))):
        raise ValueError('Original selection or early-stop rule differs')
    checkpoint = canonical_checkpoint(ROOT, relative_run, selected['selected_checkpoint'])
    selected_files = completed_checkpoint_files(ROOT / checkpoint, best_step)
    final = run / f'checkpoints/step-{steps:06d}'
    final_files = completed_checkpoint_files(final, steps)
    state = torch.load(final / 'trainer_state.pt', map_location='cpu', weights_only=True)
    order = state['order'].tolist()
    if (sorted(order) != list(range(512)) or state['world_size'] != 1
            or state['optimizer_step'] != steps or state['records_seen'] != steps * 16
            or metrics['records_seen'] != steps * 16
            or state['epoch'] * 512 + state['next_microbatch'] != steps * 16):
        raise ValueError('Actual saved training cursor differs')
    expected_reactions = state['epoch'] * 448 + sum(
        train[i]['_meta']['phase'] == 'reaction' for i in order[:state['next_microbatch']])
    hook = json.loads((out / 'loss-hook-audit.json').read_text())
    if (hook['reaction_records'] != expected_reactions or hook['adjusted_losses'] != expected_reactions
            or hook['ordinary_records'] != steps * 16 - expected_reactions):
        raise ValueError('Actual sampled-likelihood loss coverage differs')
    initial = json.loads((out / 'initialization-audit.json').read_text())
    parity = json.loads((out / 'baseline-parity.json').read_text())
    if (not initial['warm_start_exact_adapter_and_head'] or not initial['base_frozen']
            or parity['checked'] != 1408 or not parity['decisions_equal']
            or parity['maximum_probability_error'] > .0005 or parity['sampling_offsets_applied']):
        raise ValueError('Exact raw parent initialization evidence differs')
    evidence += [run / n for n in ('selection.json', 'training_metrics.json')]
    evidence += [ROOT / checkpoint / 'trainer_state.pt', final / 'trainer_state.pt']
    audit = {'status': 'passed_result_finalization_recovery',
        'reason': 'SDK returned a relative run path; export resolves before relative_to(ROOT)',
        'training_job_failed_after_optimizer_completion': True,
        'original_failure_retained': True, 'original_frozen_trainer_unchanged': True,
        'optimizer_steps': steps, 'request_exposures': steps * 16, 'unique_requests': 512,
        'selected_step': best_step, 'selected_files': selected_files, 'final_files': final_files,
        'all_prediction_rows_recomputed': checked, 'loss_hook_coverage_verified': True,
        'selection_and_stopping_recomputed': True, 'source_identity_verified': True,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in evidence},
        'recovery_code_sha256': {n: digest(ROOT / n) for n in ('sampling_finalization.py', 'scripts/finalize_sampling_result.py')},
        'new_optimizer_updates': 0, 'human_test_opened': False, 'deployed': False}
    verify_preparation(ROOT, cfg, prepared)
    write(out / 'result-finalization-audit.json', audit)
    chosen = next(r for r in curve if r['step'] == best_step)
    result = {'status': 'clean_policy_candidate', 'optimizer_steps': steps, 'selected_step': best_step,
        'checkpoint': checkpoint, 'validation': chosen, 'parent_checkpoint': cfg['warm_start'],
        'feature_only_baseline_selected': False, 'policy_changed_by_feature_format': False,
        'human_test_opened': False, 'deployed': False, 'strength_improvement_proven': False,
        'finalization_recovery': {'audit': str((out / 'result-finalization-audit.json').relative_to(ROOT)),
            'audit_sha256': digest(out / 'result-finalization-audit.json'), 'original_failure_retained': True}}
    write(out / 'result.json', result)
    write(out / 'status.json', {k: v for k, v in result.items() if k != 'validation'})
    print(json.dumps({'recovered': True, 'optimizer_steps': steps, 'selected_step': best_step,
        'all_prediction_rows_recomputed': checked, 'new_optimizer_updates': 0}), flush=True)


if __name__ == '__main__':
    main()
