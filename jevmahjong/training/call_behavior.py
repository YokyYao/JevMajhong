"""Explicit, auditable calling-only exploration for native input collection."""
from pathlib import Path
import torch
from jevmahjong.core.io import digest
from jevmahjong.policy.routing import make_head


def collection_mode(config):
    mode = config.get('call_collection_policy', 'frozen_play')
    expected = 'calling_only_pointer' if config.get('behavior_heads') else 'frozen_play'
    if mode != expected:
        raise ValueError('Explicit calling-only collection policy required for behavior heads')
    if config.get('call_context_baseline') and config.get('behavior_heads'):
        raise ValueError('A calling baseline cannot also use behavior-head exploration')
    if expected == 'calling_only_pointer' and (
            config.get('train_scopes') != ['calling'] or config.get('decision_filter') != 'all'):
        raise ValueError('Exploration collection must capture only pure calling windows')
    return mode


def load_collection_heads(config, protocol, device='cpu'):
    """Keep source encoders and ordinary/kan head exact; change only call head."""
    if collection_mode(config) == 'frozen_play':
        if protocol.get('behavior_heads'):
            raise ValueError('Unexpected collection behavior receipt')
        return None
    declaration = config['behavior_heads']
    path = Path(declaration['checkpoint']).resolve()
    receipt = {'checkpoint': str(path), 'sha256': declaration['sha256']}
    if protocol.get('behavior_heads') != receipt or digest(path) != receipt['sha256']:
        raise ValueError('Collection behavior checkpoint receipt differs')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('base_and_LoRA_frozen') is not True
            or saved.get('model_files') != protocol['model_files']
            or saved.get('trained_scopes') != ['calling']
            or saved.get('decision_filter') != 'all'
            or set(saved.get('heads', {})) != {'ordinary', 'calling'}):
        raise ValueError('Calling-only frozen-encoder pointer checkpoint required')
    heads = {}
    for scope, role in [('ordinary', 'discard'), ('calling', 'call')]:
        paths = [p for p in protocol['model_files'][role] if Path(p).name == 'head.pt']
        if len(paths) != 1 or digest(Path(paths[0])) != protocol['model_files'][role][paths[0]]:
            raise ValueError('Registered original collection head differs')
        head = make_head(torch.load(paths[0], map_location='cpu', weights_only=True), 'cpu')
        original = head.state_dict(); replacement = saved['heads'][scope]
        if set(replacement) != set(original):
            raise ValueError('Collection head parameter keys differ')
        for key, value in replacement.items():
            base = original[key]
            if (not isinstance(value, torch.Tensor) or value.shape != base.shape
                    or value.dtype != base.dtype or not torch.isfinite(value).all()):
                raise ValueError('Invalid collection head parameter')
            if scope == 'ordinary' and not torch.equal(value, base):
                raise ValueError('Ordinary and kan collection policy must remain frozen Play')
        head.load_state_dict(replacement)
        heads[scope] = head.to(device).eval().requires_grad_(False)
    return heads
