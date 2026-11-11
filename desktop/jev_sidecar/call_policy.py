"""Jev Call v1 adapter with the Jev Discard v1 fallback."""
import copy
import hashlib
import json
from pathlib import Path

METADATA = ('scores', 'dealer', 'round_wind', 'honba', 'sticks', 'hand_number', 'riichi')
IDENTITY_PATH = Path(__file__).with_name('call_policy_identity.json')


def is_call_request(request):
    criteria = request['questions']['action']['criteria']
    kinds = {c['type'] for c in criteria.values()}
    return (len(criteria) >= 2 and 'none' in kinds and bool(kinds & {'chi', 'pon'})
            and kinds <= {'chi', 'pon', 'none'})


def validate_call_checkpoint(directory):
    import torch
    from .model_identity import tensor_sha
    directory = Path(directory)
    identity = json.loads(IDENTITY_PATH.read_text(encoding='utf-8'))
    for name, expected in identity['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Jev Call v1 模型校验失败：' + name)
    head = torch.load(directory / 'head.pt', map_location='cpu', weights_only=True)
    if any(head.get(k) != v for k, v in identity['runtime'].items()):
        raise ValueError('Jev Call v1 运行配置不一致')
    if tensor_sha(head['head']) != identity['head_tensor_sha256']:
        raise ValueError('Jev Call v1 动作头不一致')
    return {'model_id': identity['model_id'], 'head_sha256': identity['files']['head.pt'],
        'adapter_sha256': identity['files']['adapter_model.safetensors']}


class CallPolicyClient:
    def __init__(self, ordinary, calling):
        self.ordinary, self.calling = ordinary, calling

    def system_one(self, **request):
        if not is_call_request(request):
            return self.ordinary.system_one(**request)
        from jevany.data import api_request
        from .call_features import enrich_request
        identity = json.loads(IDENTITY_PATH.read_text(encoding='utf-8'))
        request = copy.deepcopy(request)
        screen = request['state'].get('screen_observation', {})
        if screen and not screen.get('metadata_confirmed', False):
            confirmed = set(screen.get('confirmed_metadata', []))
            for field in METADATA:
                if field not in confirmed:
                    request['state'][field] = None
        request['questions']['action']['instructions'] = identity['policy_instruction']
        return self.calling.system_one(**api_request(enrich_request(request)))
