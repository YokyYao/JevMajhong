"""Train with sampled-condition likelihood; select raw reaction validation NLL."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from encoder_training_guard import exact_metrics
from feature_continuation import UpdatedFeatureSkillGuard, verify_preparation
from head_rl import write
from sampling_loss import SamplingLossHooks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    cfg = json.loads(p.parse_args().config.read_text())
    out, data = ROOT / cfg['out'], ROOT / cfg['data']
    if (out / 'protocol.json').exists():
        raise ValueError('Refuse overwriting an existing training run')
    prepared = json.loads((out / 'preparation.json').read_text())
    verify_preparation(ROOT, cfg, prepared)
    write(out / 'protocol.json', {'config': cfg,
        'code_sha256': {n: h for n, h in prepared['frozen_files'].items() if n.endswith('.py')},
        'reference_files': prepared['reference_files'], 'warm_start_files': prepared['warm_start_files'],
        'prepared_provenance_sha256': digest(data / 'provenance.json'),
        'parent_checkpoint': cfg['warm_start'], 'parent_step': cfg['parent_step'],
        'selection': cfg['selection_policy'], 'test_opened': False, 'deployed': False,
        'training_loss': 'CE on logits plus fixed log inclusion; raw logits for validation and serving'})
    from jevany import train as trainer
    from jevany.data import load_records
    import torch
    records = load_records(data / 'validation.jsonl')
    guard = UpdatedFeatureSkillGuard(cfg['views'])
    curve = []
    original_evaluate, original_update, original_requests = trainer.evaluate_during_training, trainer.early_stop_update, trainer.training_requests

    def requests(*a, **kw):
        result, admission = original_requests(*a, **kw)
        write(out / 'token-admission.json', admission)
        if admission['admitted_records'] != 512 or admission['rejected_records']:
            raise ValueError('All512 registered TRAIN requests required')
        return result, admission

    def evaluate(*a, **kw):
        step = kw.get('step', a[5] if len(a) > 5 else None)
        guard.step = step
        if step == 0:
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            model = kw.get('model', a[0] if a else None)
            warm = ROOT / cfg['warm_start']
            actual = get_peft_model_state_dict(model.lm)
            expected = load_file(str(warm / 'adapter_model.safetensors'), device='cpu')
            if set(actual) != set(expected) or any(not torch.equal(actual[k].detach().cpu(), v) for k, v in expected.items()):
                raise ValueError('Parent adapter initialization differs')
            actual = model.head.state_dict()
            expected = torch.load(warm / 'head.pt', map_location='cpu', weights_only=True)['head']
            if set(actual) != set(expected) or any(not torch.equal(actual[k].detach().cpu(), v) for k, v in expected.items()):
                raise ValueError('Parent head initialization differs')
            if any(v.requires_grad and 'lora_' not in n for n, v in model.lm.named_parameters()):
                raise ValueError('This experiment keeps base frozen')
            write(out / 'initialization-audit.json', {'warm_start_exact_adapter_and_head': True,
                'base_frozen': True, 'new_optimizer': True,
                'trainable_parameters': sum(v.numel() for v in model.trainable_parameters())})
        summary = original_evaluate(*a, **kw)
        directory = Path(kw.get('out_dir', a[4] if len(a) > 4 else '')) / 'training_eval' / f'step-{step:06d}'
        rows = json.loads((directory / 'development/rows.json').read_text())
        if step == 0:
            expected = json.loads((ROOT / cfg['baseline_rows']).read_text())
            if len(rows) != len(expected) or len(rows) != 1408:
                raise ValueError('Complete parent baseline coverage differs')
            maximum = 0.
            for actual, saved in zip(rows, expected):
                if any(actual[k] != saved[k] for k in ('id', 'keys', 'label')):
                    raise ValueError('Parent baseline option alignment differs')
                error = max(abs(a - b) for a, b in zip(actual['p'], saved['p']))
                maximum = max(maximum, error)
                if error > .0005 or max(range(len(actual['p'])), key=actual['p'].__getitem__) != max(range(len(saved['p'])), key=saved['p'].__getitem__):
                    raise ValueError('Raw parent baseline probabilities/actions differ')
            write(out / 'baseline-parity.json', {'checked': 1408, 'maximum_probability_error': maximum,
                'decisions_equal': True, 'sampling_offsets_applied': False})
        metrics = exact_metrics(records, rows)
        admission = guard.assess(metrics)
        by_phase = {}
        for phase in ('closed_discard', 'open_discard', 'reaction'):
            subset = [r for r in records if r['_meta']['phase'] == phase]
            ids = {r['_meta']['id'] for r in subset}
            by_phase[phase] = exact_metrics(subset, [r for r in rows if r['id'] in ids])
        item = {'step': step, 'nll': summary['clean']['nll'], 'views': metrics,
                'by_phase': by_phase, 'reaction_selection_nll': summary['transfer']['clean']['nll'],
                'guard': admission, 'sampling_offsets_applied_to_validation': False}
        if not abs(item['reaction_selection_nll'] - by_phase['reaction']['native']['nll']) < 1e-12:
            raise ValueError('Raw reaction selection data differs')
        curve.append(item)
        write(out / 'validation-curve.json', curve)
        write(out / 'status.json', {'status': 'sampling_loss_validation', 'step': step,
            'max_steps': 128, 'reaction_nll': item['reaction_selection_nll'], 'admissible': admission['admissible']})
        print(json.dumps({'step': step, 'raw_reaction_nll': item['reaction_selection_nll']}), flush=True)
        return summary

    def update(value, best, stale, maximize=True, min_delta=0.):
        return guard.update(original_update, value, best, stale, maximize=maximize, min_delta=min_delta)

    trainer.training_requests, trainer.evaluate_during_training, trainer.early_stop_update = requests, evaluate, update
    write(out / 'status.json', {'status': 'loading_preserved_stage2_sampling_likelihood', 'step': 0})
    hooks = SamplingLossHooks(trainer, prepared['inverse_inclusion_weights'])
    try:
        base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
        with hooks:
            run = trainer.main(['--config', str(ROOT / cfg['trainer_config']), '--base_load_path', str(base)])
        verify_preparation(ROOT, cfg, prepared)
        metrics = json.loads((run / 'training_metrics.json').read_text())
        if (metrics['global_effective_batch'] != 16 or metrics['admitted_records'] != 512 or metrics['rejected_records']
                or hooks.reaction_records + hooks.ordinary_records != metrics['records_seen']
                or hooks.adjusted_losses != hooks.reaction_records or not hooks.adjusted_losses):
            raise ValueError('Actual sampling likelihood exposure differs')
        write(out / 'loss-hook-audit.json', {'reaction_records': hooks.reaction_records,
            'ordinary_records': hooks.ordinary_records, 'adjusted_losses': hooks.adjusted_losses,
            'training_only': True, 'raw_logits_unchanged_for_serving': True})
        selected = json.loads((run / 'selection.json').read_text())
        step = selected.get('step')
        chosen = next((r for r in curve if r['step'] == step), None)
        qualified = step is not None and step > 0 and chosen is not None and chosen['guard']['admissible']
        result = {'status': 'clean_policy_candidate' if qualified else 'clean_policy_no_admissible_improvement',
            'optimizer_steps': metrics['optimizer_steps'], 'selected_step': step if qualified else None,
            'checkpoint': str((run / selected['selected_checkpoint']).relative_to(ROOT)) if qualified else None,
            'validation': chosen if qualified else None, 'parent_checkpoint': cfg['warm_start'],
            'feature_only_baseline_selected': False, 'policy_changed_by_feature_format': False,
            'human_test_opened': False, 'deployed': False, 'strength_improvement_proven': False}
        write(out / 'result.json', result)
        write(out / 'status.json', {k: v for k, v in result.items() if k != 'validation'})
    except Exception as exc:
        write(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'deployed': False})
        raise
    finally:
        trainer.training_requests, trainer.evaluate_during_training, trainer.early_stop_update = original_requests, original_evaluate, original_update


if __name__ == '__main__':
    main()
