"""Freeze maintained source and model files for a new run."""
from pathlib import Path
from jevmahjong.core.io import digest

ROOT = Path(__file__).resolve().parents[2]


def snapshot(config):
    models = {}
    for role in ('discard', 'call'):
        directory = Path(config[role + '_checkpoint']).resolve()
        models[role] = {str(p): digest(p) for p in directory.iterdir()
            if p.is_file() and p.name in ('head.pt', 'adapter_model.safetensors', 'adapter_config.json',
                'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
        if len(models[role]) != 6:
            raise ValueError('Incomplete inference model: ' + role)
    result = {'config': config, 'code_sha256': {str(p): digest(p) for p in (ROOT / 'jevmahjong').rglob('*.py')},
        'model_files': models, 'human_test_opened': False}
    for name in ('initial_heads', 'behavior_heads', 'call_context_baseline'):
        if config.get(name):
            initial = config[name]; path = Path(initial['checkpoint']).resolve()
            if digest(path) != initial['sha256']: raise ValueError(name + ' checkpoint changed')
            result[name] = {'checkpoint': str(path), 'sha256': initial['sha256']}
    return result


def verify(protocol):
    for path, expected in protocol['code_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Registered source changed: ' + path)
    for files in protocol['model_files'].values():
        for path, expected in files.items():
            if digest(Path(path)) != expected:
                raise ValueError('Registered model changed: ' + path)
    for name in ('initial_heads', 'behavior_heads', 'call_context_baseline'):
        if protocol.get(name):
            initial = protocol[name]
            if digest(Path(initial['checkpoint'])) != initial['sha256']:
                raise ValueError(name + ' checkpoint changed')
