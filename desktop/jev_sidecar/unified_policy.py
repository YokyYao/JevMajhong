"""Portable serving and pinned identity for the Jev Play v2 release."""
import copy
import hashlib
import json
from pathlib import Path

IDENTITY_PATH = Path(__file__).with_name('play_v2_identity.json')
MODEL_ID = 'jev-play-v2'
MODEL_NAME = 'Jev Play v2'
STATE_KEYS = {'seat', 'dealer', 'round_wind', 'honba', 'sticks', 'scores', 'hand',
    'dora_indicators', 'riichi', 'rivers', 'melds', 'discards_count', 'hand_number',
    'tsumogiri_flags', 'riichi_sutehais', 'last_tedashis', 'last_discard', 'screen_observation'}


def identity():
    return json.loads(IDENTITY_PATH.read_text(encoding='utf-8'))


def default_checkpoint(root):
    return Path(root) / 'models' / MODEL_ID


def validate_checkpoint(directory):
    import torch
    from .model_identity import tensor_sha
    directory = Path(directory)
    expected = identity()
    marker = directory / 'play-v2-identity.json'
    if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != expected:
        raise ValueError('Jev Play v2 模型身份不一致')
    for name, digest in expected['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Jev Play v2 模型文件校验失败：' + name)
    payload = torch.load(directory / 'head.pt', map_location='cpu', weights_only=True)
    if any(payload.get(k) != v for k, v in expected['runtime'].items()):
        raise ValueError('Jev Play v2 运行配置不一致')
    if tensor_sha(payload['head']) != expected['head_tensor_sha256']:
        raise ValueError('Jev Play v2 动作头不一致')
    return {'model_id': MODEL_ID, 'source_model_id': expected['source_model_id'],
        'source_step': expected['source_step'], 'identity_sha256': expected['identity_sha256']}


def serving_request(request):
    """Match the unified training format without importing simulation packages."""
    if not isinstance(request.get('state'), dict) or set(request['state']) - STATE_KEYS:
        raise ValueError('Only the shared public-state allowlist is permitted')
    if set(request.get('questions', {})) != {'action'}:
        raise ValueError('Only the public action question is permitted')
    from .call_features import enrich_request
    payload = copy.deepcopy(request)
    payload['questions']['action']['instructions'] = identity()['policy_instruction']
    payload = enrich_request(payload)
    return {'state': payload['state'], 'questions': {
        name: {k: v for k, v in question.items() if k in ('type', 'instructions', 'criteria')}
        for name, question in payload['questions'].items()}}


class UnifiedPolicyClient:
    def __init__(self, model):
        self.model = model

    def system_one_with_trace(self, **request):
        payload = serving_request(request)
        trace = {'version': 'play-model-input-v2', 'route': 'unified',
            'model_id': MODEL_ID, 'source_request': copy.deepcopy(request),
            'model_request': copy.deepcopy(payload)}
        return self.model.system_one(**payload), trace

    def system_one(self, **request):
        return self.system_one_with_trace(**request)[0]
