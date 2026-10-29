"""Freeze a training candidate for confirmation against the declared original benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, out, root=ROOT):
    if out.exists():
        raise ValueError('Confirmation output already exists')
    original = json.loads((source / 'protocol.json').read_text())
    selected = json.loads((source / 'confirmation-selection.json').read_text())
    analysis = json.loads((source / 'pilot-analysis.json').read_text())
    config = dict(original['config'])
    if original['sha256'] != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError('Source protocol fingerprint differs')
    if selected['config'] != config or not analysis['selected_for_confirmation']:
        raise ValueError('Source must have one completed development selection')
    if analysis['selected_for_confirmation']['checkpoint'] != selected['checkpoint']:
        raise ValueError('Source candidate selection differs')
    checkpoint = root / selected['checkpoint']
    if not checkpoint.resolve().is_relative_to((source / 'checkpoints').resolve()):
        raise ValueError('Candidate must be a checkpoint of this source trial')
    if not (checkpoint / 'complete.json').exists():
        raise ValueError('Source checkpoint is incomplete')
    head_sha = fingerprint(checkpoint / 'head.pt')
    if selected['candidate_head_sha256'] != head_sha:
        raise ValueError('Selected head changed')
    benchmark = root / config['final_benchmark_reference']
    benchmark_sha = fingerprint(benchmark / 'head.pt')
    if benchmark_sha != config['final_benchmark_head_sha256']:
        raise ValueError('Declared original benchmark changed')
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json',
                 'tokenizer_config.json', 'chat_template.jinja'):
        if fingerprint(checkpoint / name) != fingerprint(benchmark / name):
            raise ValueError(f'Candidate and benchmark frozen input differs: {name}')
    target = config['final_confirmation_target_net_points_gain']
    if target < 500 or config['reward_mode'] != 'net_points' or config['game_mode'] != '4p-red-east':
        raise ValueError('Final net-point benchmark gate cannot be weakened')
    if config['seed_scheme'] != 'spaced-v1' or config['confirmation_seeds'] < 128:
        raise ValueError('Final test requires at least128 independent wall streams')
    if config['final_confirmation_target_mean_net_points'] < 0:
        raise ValueError('Final mean net points must be nonnegative')
    for name, expected in original['code_sha256'].items():
        if fingerprint(root / name) != expected:
            raise ValueError(f'Numerical code changed since training: {name}')
    files = sorted(source.glob('training-games-stage-*.json'))
    if not files:
        raise ValueError('Missing training provenance')
    training = {row['seed'] for path in files for row in json.loads(path.read_text())}
    development = set(range(config['development_seed_start'],
                            config['development_seed_start'] + config['development_seeds']))
    confirmation = set(range(config['confirmation_seed_start'],
                             config['confirmation_seed_start'] + config['confirmation_seeds']))
    if training & development or training & confirmation or development & confirmation:
        raise ValueError('Training and evaluation streams overlap')
    training_reference = config['reference']
    config.update(reference=config['final_benchmark_reference'], out=str(out),
                  target_net_points_gain=target,
                  target_mean_net_points=config['final_confirmation_target_mean_net_points'])
    selection = {**selected, 'config': config, 'source_training_reference': training_reference,
                 'source_run': str(source), 'candidate_head_sha256': head_sha,
                 'selection_rule': 'One preselected development candidate; final benchmark and threshold declared before training'}
    protocol = {**original, 'config': config,
                'sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                'reference_head_sha256': benchmark_sha,
                'reference_adapter_sha256': fingerprint(benchmark / 'adapter_model.safetensors'),
                'source_protocol_sha256': original['sha256'], 'source_run': str(source),
                'source_training_reference': training_reference,
                'study_type': 'final_confirmation_against_original_benchmark'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=out.name + '.prepare-') as temporary:
        staged = Path(temporary) / 'study'
        staged.mkdir()
        for name, value in [('protocol.json', protocol), ('confirmation-selection.json', selection),
                            ('status.json', {'status': 'awaiting_confirmation', 'checkpoint': selected['checkpoint']})]:
            (staged / name).write_text(json.dumps(value, indent=2) + '\n')
        for path in files + sorted(source.glob('reference-controls-stage-*.json')):
            shutil.copy2(path, staged / path.name)
        shutil.copy2(source / 'protocol.json', staged / 'source-training-protocol.json')
        shutil.copy2(source / 'pilot-analysis.json', staged / 'source-training-analysis.json')
        staged.rename(out)
    return {'out': str(out), 'checkpoint': selected['checkpoint'], 'candidate_head_sha256': head_sha,
            'training_reference': training_reference, 'final_benchmark': config['reference'],
            'final_gain_target': target}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out)), flush=True)


if __name__ == '__main__':
    main()
