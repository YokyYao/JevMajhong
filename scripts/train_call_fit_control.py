"""Bounded optimization controls; checkpoint selection measures TRAIN fit only."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_fit_diagnostic import reaction_records, summarize_fit
from call_policy import digest
from head_rl import write


def verify(prepared):
    if prepared['human_test_opened'] or prepared['deployed']:
        raise ValueError('Fit controls must remain private experiments')
    for name, expected in prepared['frozen_files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Fit control source changed: ' + name)
    for name, expected in prepared['warm_start_files'].items():
        if digest(ROOT / prepared['config']['warm_start'] / name) != expected:
            raise ValueError('Preserved stage2 weights changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('natural', 'balanced'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    prepared = json.loads((ROOT / cfg['out'] / 'preparation.json').read_text())
    if prepared['config'] != cfg:
        raise ValueError('Control config differs')
    verify(prepared)
    out = ROOT / cfg['out'] / args.arm
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', {'config': cfg, 'arm': args.arm,
        'preparation_sha256': digest(ROOT / cfg['out'] / 'preparation.json'),
        'purpose': 'learning capability / class mixture control; not strength candidate'})
    from jevany import train as trainer
    from jevany.data import load_records
    import torch
    train_records = reaction_records(load_records(ROOT / cfg['data'] / args.arm / 'train.jsonl'), 'train')
    validation_records = reaction_records(load_records(ROOT / cfg['data'] / args.arm / 'validation.jsonl'), 'validation')
    original_evaluate, original_requests = trainer.evaluate_during_training, trainer.training_requests
    warm = ROOT / cfg['warm_start']
    curve = []

    def audited_requests(*a, **kw):
        requests, admission = original_requests(*a, **kw)
        write(out / 'token-admission.json', admission)
        if admission['rejected_records'] or admission['admitted_records'] != 512:
            raise ValueError('Every control request must be admitted')
        return requests, admission

    def evaluate(*a, **kw):
        step = kw.get('step', a[5] if len(a) > 5 else None)
        model = kw.get('model', a[0] if a else None)
        if step == 0:
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            expected = load_file(str(warm / 'adapter_model.safetensors'), device='cpu')
            actual = get_peft_model_state_dict(model.lm)
            if set(expected) != set(actual) or any(not torch.equal(actual[k].detach().cpu(), v) for k, v in expected.items()):
                raise ValueError('Control adapter initialization differs')
            expected = torch.load(warm / 'head.pt', map_location='cpu', weights_only=True)['head']
            actual = model.head.state_dict()
            if set(expected) != set(actual) or any(not torch.equal(actual[k].detach().cpu(), v) for k, v in expected.items()):
                raise ValueError('Control head initialization differs')
            if any(p.requires_grad and 'lora_' not in name for name, p in model.lm.named_parameters()):
                raise ValueError('This control must keep base parameters frozen')
            write(out / 'initialization-audit.json', {'warm_start_exact_adapter_and_head': True,
                'base_frozen': True, 'fresh_optimizer': True,
                'trainable_parameters': sum(p.numel() for p in model.trainable_parameters())})
        summary = original_evaluate(*a, **kw)
        directory = Path(kw.get('out_dir', a[4] if len(a) > 4 else '')) / 'training_eval' / f'step-{step:06d}'
        metrics = {}
        for split, records, subdir in [('train', train_records, 'transfer'),
                                       ('validation', validation_records, 'development')]:
            rows = json.loads((directory / subdir / 'rows.json').read_text())
            ids = {r['_meta']['id'] for r in records}
            metrics[split] = summarize_fit(records, [r for r in rows if r['id'] in ids])
        item = {'step': step, 'full_validation_nll': summary['clean']['nll'],
                'fit_selection_nll': summary['transfer']['clean']['nll'], 'reaction_fit': metrics,
                'selection_is_train_only': True}
        curve.append(item)
        write(out / 'curve.json', curve)
        write(out / 'status.json', {'status': 'fit_control_validation', 'arm': args.arm, 'step': step,
            'max_steps': 128, 'train_closed_chi': metrics['train'].get('closed_chi'),
            'validation_closed_chi': metrics['validation']['closed_chi']})
        print(json.dumps({'arm': args.arm, 'step': step, 'fit_selection_nll': item['fit_selection_nll']}), flush=True)
        return summary

    trainer.evaluate_during_training, trainer.training_requests = evaluate, audited_requests
    write(out / 'status.json', {'status': 'loading_preserved_stage2_for_fit_control', 'arm': args.arm})
    try:
        base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
        run = trainer.main(['--config', str(ROOT / cfg['recipes'][args.arm]), '--base_load_path', str(base)])
        verify(prepared)
        metrics = json.loads((run / 'training_metrics.json').read_text())
        if metrics['global_effective_batch'] != 16 or metrics['admitted_records'] != 512 or metrics['rejected_records']:
            raise ValueError('Actual control budget differs')
        write(out / 'result.json', {'status': 'fit_control_complete', 'optimizer_steps': metrics['optimizer_steps'],
            'selection': json.loads((run / 'selection.json').read_text()), 'strength_candidate': False,
            'validation_used_for_diagnostics_only': True, 'new_unseen_requests': 0,
            'human_test_opened': False, 'deployed': False})
        write(out / 'status.json', {'status': 'fit_control_complete', 'optimizer_steps': metrics['optimizer_steps']})
    except Exception as exc:
        write(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'deployed': False})
        raise
    finally:
        trainer.evaluate_during_training, trainer.training_requests = original_evaluate, original_requests


if __name__ == '__main__':
    main()
