"""Train a new immutable stage on requests excluded from the parent exposure."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from encoder_training_guard import exact_metrics
from feature_continuation import UpdatedFeatureSkillGuard, verify_preparation
from head_rl import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    data, out = ROOT / cfg['data'], ROOT / cfg['out']
    if (out / 'protocol.json').exists() or (out / 'invalidation.json').exists():
        raise ValueError('Refuse overwriting an existing experiment')
    try:
        prepared = json.loads((data / 'provenance.json').read_text())
        verify_preparation(ROOT, cfg, prepared)
        recipe = tomllib.loads((ROOT / cfg['trainer_config']).read_text())
        expected = {'init_from': cfg['warm_start'], 'out': cfg['out'] + '/training-run',
            'data': cfg['data'] + '/train.jsonl', 'eval_data': cfg['data'] + '/validation.jsonl',
            'max_steps': cfg['max_steps'], 'eval_every_steps': cfg['validation_every'],
            'early_stop_patience': cfg['patience'], 'seed': cfg['seed'],
            'batch': 1, 'accum': 16, 'epochs': 1, 'p_none_pair': 0.0,
            'p_none': 0.0, 'p_none_distract': 0.0, 'p_distract': 0.0}
        if any(recipe.get(k) != v for k, v in expected.items()) or recipe.get('resume'):
            raise ValueError('Registered new-stage recipe differs')
        warm = ROOT / cfg['warm_start']
        code = {name: digest(ROOT / name) for name in prepared['frozen_files'] if name.endswith('.py')}
        protocol = {'config': cfg, 'code_sha256': code,
            'trainer_config_sha256': digest(ROOT / cfg['trainer_config']),
            'prepared_provenance_sha256': digest(data / 'provenance.json'),
            'reference_files': prepared['reference_files'], 'warm_start_files': prepared['warm_start_files'],
            'parent_checkpoint': cfg['warm_start'], 'parent_step': cfg['parent_step'],
            'selection': cfg['selection_policy'], 'test_opened': False,
            'deployed': False, 'strength_improvement_proven': False}
        write(out / 'protocol.json', protocol)
        records = [json.loads(line) for line in (data / 'validation.jsonl').open()]
        from jevany import train as trainer
        import torch
        original_requests = trainer.training_requests
        original_evaluate = trainer.evaluate_during_training
        original_update = trainer.early_stop_update
        guard = UpdatedFeatureSkillGuard(cfg['views'])
        curve, started = [], time.monotonic()

        def audited_requests(*args, **kwargs):
            requests, admission = original_requests(*args, **kwargs)
            write(out / 'token-admission.json', admission)
            if admission['rejected_records'] or admission['admitted_records'] != prepared['records']['train']:
                raise ValueError('All registered unseen requests must be admitted')
            return requests, admission

        def verify_warm(model):
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            actual = get_peft_model_state_dict(model.lm)
            expected_adapter = load_file(str(warm / 'adapter_model.safetensors'), device='cpu')
            if set(actual) != set(expected_adapter) or any(
                    not torch.equal(actual[k].detach().cpu(), expected_adapter[k]) for k in expected_adapter):
                raise ValueError('Warm adapter is not the frozen parent candidate')
            actual = model.head.state_dict()
            expected_head = torch.load(warm / 'head.pt', map_location='cpu', weights_only=True)['head']
            if set(actual) != set(expected_head) or any(
                    not torch.equal(actual[k].detach().cpu(), expected_head[k]) for k in expected_head):
                raise ValueError('Warm pointer differs from the parent')
            write(out / 'initialization-audit.json', {'warm_start_exact_adapter_and_head': True,
                'no_oldB_weights_loaded': True, 'parent_checkpoint': cfg['warm_start'],
                'new_optimizer_and_schedule': True, 'adapter_tensor_count': len(expected_adapter),
                'trainable_parameters': sum(v.numel() for v in model.trainable_parameters())})

        def evaluate(*args, **kwargs):
            step = kwargs.get('step', args[5] if len(args) > 5 else None)
            guard.step = step
            if step == 0:
                verify_warm(kwargs.get('model', args[0] if args else None))
            summary = original_evaluate(*args, **kwargs)
            directory = Path(kwargs.get('out_dir', args[4] if len(args) > 4 else ''))
            rows = json.loads((directory / 'training_eval' / f'step-{step:06d}' / 'development' / 'rows.json').read_text())
            metrics = exact_metrics(records, rows)
            admission = guard.assess(metrics)
            by_phase = {}
            for phase in ('closed_discard', 'open_discard', 'reaction'):
                subset = [r for r in records if r['_meta']['phase'] == phase]
                ids = {r['_meta']['id'] for r in subset}
                by_phase[phase] = exact_metrics(subset, [r for r in rows if r['id'] in ids])
            item = {'step': step, 'nll': summary['clean']['nll'], 'views': metrics,
                'by_phase': by_phase, 'guard': admission, 'seconds': time.monotonic() - started}
            curve.append(item)
            write(out / f'validation-{step:06d}.json', item)
            write(out / 'validation-curve.json', curve)
            status = {'status': 'clean_policy_validation', 'step': step,
                'max_steps': cfg['max_steps'], 'nll': item['nll'], 'admissible': admission['admissible']}
            write(out / 'status.json', status)
            print(json.dumps(status), flush=True)
            return summary

        def update(value, best, stale, maximize=True, min_delta=0.0):
            return guard.update(original_update, value, best, stale,
                                maximize=maximize, min_delta=min_delta)

        trainer.training_requests = audited_requests
        trainer.evaluate_during_training = evaluate
        trainer.early_stop_update = update
        write(out / 'status.json', {'status': 'loading_frozen_feature_parent_unseen_stage', 'step': 0})
        try:
            base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
            train_dir = trainer.main(['--config', str(ROOT / cfg['trainer_config']), '--base_load_path', str(base)])
        finally:
            trainer.training_requests = original_requests
            trainer.evaluate_during_training = original_evaluate
            trainer.early_stop_update = original_update
        verify_preparation(ROOT, cfg, prepared)
        selection = json.loads((train_dir / 'selection.json').read_text())
        metrics = json.loads((train_dir / 'training_metrics.json').read_text())
        if metrics['global_effective_batch'] != 16 or metrics['rejected_records'] or metrics['admitted_records'] != prepared['records']['train']:
            raise ValueError('Actual continuation recipe differs')
        step = selection.get('step')
        chosen = next((r for r in curve if r['step'] == step), None)
        qualified = (selection.get('selected_checkpoint') is not None and step is not None
                     and step > 0 and chosen is not None and chosen['guard']['admissible'])
        result = {'status': 'clean_policy_candidate' if qualified else 'clean_policy_no_admissible_improvement',
            'optimizer_steps': metrics['optimizer_steps'], 'selected_step': step if qualified else None,
            'checkpoint': str((train_dir / selection['selected_checkpoint']).resolve().relative_to(ROOT)) if qualified else None,
            'validation': chosen if qualified else None, 'parent_checkpoint': cfg['warm_start'],
            'feature_only_baseline_selected': False, 'policy_changed_by_feature_format': False,
            'frozenB_unchanged': True, 'parent_confirmation_weights_unchanged': True,
            'human_test_opened': False, 'deployed': False, 'strength_improvement_proven': False,
            'next': 'Serving parity and fresh complete-East net points; compare with the frozen parent on matched walls before claiming incremental benefit'}
        write(out / 'result.json', result)
        write(out / 'status.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'validation'}), flush=True)
    except Exception as exc:
        write(out / 'training-failure.json', {'type': type(exc).__name__, 'error': str(exc), 'deployed': False})
        raise


if __name__ == '__main__':
    main()
