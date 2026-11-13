"""Pin an unchanged calling residual used by kan collection and continuation."""
from pathlib import Path
import torch
from jevmahjong.core.io import digest
from .kan_context import CallContextPolicyClient, feature_names, probabilities
from .routing import is_call_request


def identity(config):
    spec = config.get('call_context_baseline')
    if spec is None:
        return None
    if (not isinstance(spec, dict) or set(spec) != {'checkpoint', 'sha256'}
            or not isinstance(spec['checkpoint'], str) or not spec['checkpoint']
            or not isinstance(spec['sha256'], str) or len(spec['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in spec['sha256'])):
        raise ValueError('Explicit calling baseline checkpoint and SHA256 required')
    return spec['sha256']


def load(config, model_files):
    checksum = identity(config)
    if checksum is None:
        return None
    if config.get('train_scopes') != ['ordinary'] or config.get('decision_filter') != 'kan_windows':
        raise ValueError('Calling baseline currently requires a declared ordinary kan workflow')
    path = Path(config['call_context_baseline']['checkpoint'])
    if digest(path) != checksum:
        raise ValueError('Calling baseline checkpoint changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('base_and_LoRA_frozen') is not True or saved.get('model_files') != model_files
            or saved.get('trained_scopes') != ['calling'] or saved.get('decision_filter') != 'call_windows'):
        raise ValueError('Calling baseline must preserve the matching encoders and pure-call scope')
    for scope, role in [('ordinary', 'discard'), ('calling', 'call')]:
        paths = [p for p in model_files[role] if Path(p).name == 'head.pt']
        if len(paths) != 1 or digest(Path(paths[0])) != model_files[role][paths[0]]:
            raise ValueError('Calling baseline original head identity differs')
        original = torch.load(paths[0], map_location='cpu', weights_only=True)['head']
        actual = saved['heads'][scope]
        if set(actual) != set(original) or any(not torch.equal(actual[k].cpu(), v.cpu()) for k, v in original.items()):
            raise ValueError('Calling baseline must leave both original pointer heads frozen')
    context = saved['kan_context']
    if tuple(context['names']) != feature_names(context['version']):
        raise ValueError('Calling baseline feature schema differs')
    return context


def wrap(frozen, context):
    return frozen if context is None else CallContextPolicyClient(frozen, context)


def transform(request, reference, context):
    if context is None or not is_call_request(request):
        return reference
    return torch.tensor(probabilities(request, reference.tolist(), context), dtype=reference.dtype,
                        device=reference.device)
