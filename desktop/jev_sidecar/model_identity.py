"""Accept only the confirmed model, including byte-independent release reconstruction."""
import hashlib
import json
from pathlib import Path

IDENTITY = json.loads(Path(__file__).with_name('identity_data.json').read_text(encoding='utf-8'))


def tensor_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint(directory):
    import torch
    directory = Path(directory)
    for filename, expected in IDENTITY['base_files'].items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('模型文件校验失败：' + filename)
    payload = torch.load(directory / 'head.pt', map_location='cpu', weights_only=True)
    for field, expected in IDENTITY['runtime'].items():
        if payload.get(field) != expected:
            raise ValueError('模型运行参数校验失败：' + field)
    actual = tensor_sha(payload['head'])
    if actual != IDENTITY['head_tensor_sha256']:
        raise ValueError('评分头与已确认的校正版2不一致')
    return {'tensor_sha256': actual, 'source_head_sha256': IDENTITY['source_head_sha256']}
