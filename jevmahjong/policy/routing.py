import copy
import os
from pathlib import Path
from threading import RLock

from jevmahjong.core.io import digest, request_hash
METADATA = ('scores','dealer','round_wind','honba','sticks','hand_number','riichi')

def is_call_request(request):
    criteria = request['questions']['action']['criteria']
    kinds = {v['type'] for v in criteria.values()}
    return len(criteria) >= 2 and 'none' in kinds and bool(kinds & {'chi', 'pon'}) and kinds <= {'chi','pon','none'}


class CallPolicyClient:
    """Frozen Jev Call v1 public-feature adapter."""
    def __init__(self, candidate): self.candidate = candidate

    def system_one(self, **request):
        from jevany.data import api_request
        from jevmahjong.data.policy_data import policy_request
        from jevmahjong.policy.call_features import enrich_request
        return self.candidate.system_one(**api_request(enrich_request(policy_request(request))))


class HeadPolicyClient:
    """Replace ordinary and call heads, preserving both frozen feature encoders."""
    def __init__(self, ordinary, calling, heads):
        self.ordinary, self.calling, self.heads = ordinary, calling, heads
        self.call_policy = CallPolicyClient(calling)
        self.lock = RLock()

    def system_one(self, **request):
        from jevmahjong.training.rewards import scope_for
        scope = scope_for(request)
        if scope is None:
            return self.ordinary.system_one(**request)
        client = self.ordinary if scope == 'ordinary' else self.calling
        invoke = self.ordinary if scope == 'ordinary' else self.call_policy
        model = client.runtime.model
        with self.lock:
            original = model.head
            try:
                model.head = self.heads[scope]
                return invoke.system_one(**request)
            finally:
                model.head = original


class HybridPolicy:
    """Jev Play v1: Discard v1 for ordinary/kan; Call v1 for chi/pon/pass."""
    def __init__(self, ordinary, calling):
        self.ordinary = ordinary
        self.calling = CallPolicyClient(calling)

    def system_one(self, **request):
        if not is_call_request(request):
            return self.ordinary.system_one(**request)
        request = copy.deepcopy(request)
        screen = request['state'].get('screen_observation', {})
        if screen and not screen.get('metadata_confirmed', False):
            confirmed = set(screen.get('confirmed_metadata', []))
            for field in METADATA:
                if field not in confirmed:
                    request['state'][field] = None
        return self.calling.system_one(**request)


def load_model(path, device='cuda'):
    import torch
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if device == 'cpu':
        import inspect
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
    base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() /
        '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
    return JevModel.from_pretrained(str(path), device=device,
        dtype='bf16' if device == 'cuda' else 'fp32',
        options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
            fused_kernels=False, base_load_path=str(base) if base.is_dir() else None),
        inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
            max_packed_tokens=8192, prefix_cache_size=0))


def make_head(meta, device):
    from jevany.model import PointerHead
    if meta.get('temperature') != 1. or meta.get('head_residual_dim') != 0:
        raise ValueError('Expected plain temperature1 pointer')
    dp, d = meta['head']['q.weight'].shape
    head = PointerHead(d, dp).to(device)
    head.load_state_dict(meta['head'])
    return head


def load_updated_heads(checkpoint, model_files, device):
    """Load two frozen-encoder heads, without replacing the opponent's heads."""
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('base_and_LoRA_frozen') is not True or saved['model_files'] != model_files:
        raise ValueError('Updated heads use different frozen encoders')
    if set(saved['heads']) != {'ordinary', 'calling'}:
        raise ValueError('Both updated policy heads required')
    result = {}
    for scope, role in (('ordinary', 'discard'), ('calling', 'call')):
        paths = [p for p in model_files[role] if Path(p).name == 'head.pt']
        if len(paths) != 1: raise ValueError('One source head per encoder required')
        head = make_head(torch.load(paths[0], map_location='cpu', weights_only=True), device)
        head.load_state_dict(saved['heads'][scope])
        result[scope] = head.eval().requires_grad_(False)
    return result
