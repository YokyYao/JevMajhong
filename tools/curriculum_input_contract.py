"""Bind a context-aware checkpoint to its public preprocessing version."""
import hashlib
import json
from pathlib import Path

from tools.curriculum_riichi_context import VERSION, enrich_request

ROOT = Path(__file__).resolve().parents[1]
NAME = 'input-contract.json'
SCHEMA = 'jev-curriculum-input-contract-v1'


def expected_contract(manifest, model):
    c = manifest['curriculum']
    if c.get('riichi_input_schema') != VERSION:
        raise ValueError('Explicit supported riichi input schema required')
    path = ROOT/'tools/curriculum_riichi_context.py'
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest['inputs'].get(str(path.resolve())) != actual:
        raise ValueError('Public riichi transform differs from registered code')
    value = dict(schema=SCHEMA, input_schema=VERSION, feature_code_sha256=actual,
        model_bundle_identity_sha256=model['identity_sha256'], model_files=model['files'],
        decision_path='native_forced_win_guarded_flat_pointer_v1', extra_model_forward_passes=0)
    value['policy_identity_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return value


def seal_checkpoint(checkpoint, manifest, bundle_fn):
    checkpoint = Path(checkpoint)
    expected = expected_contract(manifest, bundle_fn(checkpoint))
    path = checkpoint/NAME
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Existing checkpoint input contract differs')
    else:
        with path.open('x') as f:
            json.dump(expected, f, indent=2)
            f.write('\n')
    return expected


def verify_checkpoint(checkpoint, manifest, bundle_fn):
    path = Path(checkpoint)/NAME
    if not path.exists():
        raise ValueError('Context-aware checkpoint is missing its input contract')
    expected = expected_contract(manifest, bundle_fn(checkpoint))
    if json.loads(path.read_text()) != expected:
        raise ValueError('Checkpoint weights or public input contract changed')
    return expected


class PublicRiichiClient:
    """One public transform then one underlying model call; no action override."""
    def __init__(self, inner):
        self.inner = inner

    def system_one(self, **request):
        return self.inner.system_one(**enrich_request(request))


def load_context_client(inner, checkpoint, manifest, bundle_fn):
    verify_checkpoint(checkpoint, manifest, bundle_fn)
    return PublicRiichiClient(inner)
