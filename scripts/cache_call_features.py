"""Resume audited frozen reaction features for TRAIN/validation; never encode test."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest, is_call_request, request_hash
from scripts.train_call_repair import make_head, payload


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def expected_rows(records, views):
    rows = []
    for record in records:
        if not is_call_request(record) or request_hash(record) != record['_meta']['text_sha256']:
            raise ValueError('Reaction identity mismatch')
        q = record['questions']['action']; keys = list(q['criteria'])
        for view in views:
            rows.append({'id': record['_meta']['id'], 'group': record['_meta']['group_id'],
                'view': view, 'label': keys.index(q['label']), 'types': [a['type'] for a in q['criteria'].values()],
                'text_sha256': record['_meta']['text_sha256']})
    return rows


def verified_chunk(path, expected, protocol_sha, dimension):
    sidecar = path.with_suffix('.audit.json')
    if not path.exists() or not sidecar.exists(): return None
    audit = json.loads(sidecar.read_text())
    if (audit['protocol_sha256'] != protocol_sha or audit['checks'] != len(expected)
        or audit['tensor_sha256'] != digest(path)):
        raise ValueError('Cached shard checksum/protocol mismatch')
    rows = torch.load(path, map_location='cpu', weights_only=True)
    if len(rows) != len(expected): raise ValueError('Cached shard count mismatch')
    for row, identity in zip(rows, expected):
        if any(row[k] != v for k, v in identity.items()): raise ValueError('Cached row identity mismatch')
        if (row['decide'].shape != (dimension,) or row['options'].shape != (len(row['types']), dimension)
            or not torch.isfinite(row['decide']).all() or not torch.isfinite(row['options']).all()):
            raise ValueError('Cached feature shape/values mismatch')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    data, out, reference = (ROOT / cfg[k] for k in ('data', 'out', 'reference'))
    audit = json.loads((data / 'provenance.json').read_text())
    if not audit['complete']: raise ValueError('Incomplete data')
    if cfg['feature_splits'] != ['train', 'validation']: raise ValueError('Test encoding is forbidden')
    for split in cfg['feature_splits']:
        if audit['counts'][split] != cfg['expected_records'][split] or digest(data / f'{split}.jsonl') != audit['files'][split]:
            raise ValueError('Source split count/checksum changed')
    if digest(reference / 'head.pt') != cfg['reference_head_sha256']: raise ValueError('Reference changed')
    source_protocol = json.loads((ROOT / cfg['reference_training_protocol']).read_text())
    for name, h in source_protocol['reference_files'].items():
        if digest(reference / name) != h: raise ValueError('Frozen reference file changed')
    code = ['scripts/cache_call_features.py', 'scripts/train_call_repair.py', 'call_policy.py',
            'semantic.py', 'vendor/JevAny/jevany/model.py']
    protocol = {'config': cfg, 'feature_splits': cfg['feature_splits'], 'test_opened': False,
        'data_provenance_sha256': digest(data / 'provenance.json'), 'reference_files': source_protocol['reference_files'],
        'code_sha256': {name: digest(ROOT / name) for name in code}, 'parameter_updates': 0}
    out.mkdir(parents=True, exist_ok=True); (out / 'features').mkdir(exist_ok=True)
    lock = (out / 'cache.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol: raise ValueError('Resume protocol mismatch')
    else: write(out / 'protocol.json', protocol)
    protocol_sha = digest(out / 'protocol.json')
    from jevany import JevModel
    from jevany.api import validate_distribution
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    base = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
    write(out / 'status.json', {'status': 'loading_frozen_model', 'parameter_updates': 0, 'test_opened': False})
    client = JevModel.from_pretrained(str(reference), device='cuda', dtype='bf16',
        options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
            fused_kernels=False, base_load_path=str(base)),
        inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192, max_packed_tokens=8192, prefix_cache_size=0))
    client.runtime.model.requires_grad_(False)
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=True)
    head = make_head(meta, 'cpu').eval().requires_grad_(False); dimension = meta['head']['q.weight'].shape[1]
    capture = []
    hook = client.runtime.model.head.register_forward_pre_hook(lambda module, inputs: capture.append(tuple(t.detach() for t in inputs)))
    completed_checks, reused_checks, started = 0, 0, time.monotonic()
    tensors = {}
    try:
        for split in cfg['feature_splits']:
            records = [json.loads(s) for s in (data / f'{split}.jsonl').read_text().splitlines()]
            if len(records) != audit['counts'][split]: raise ValueError('Raw record count mismatch')
            for start in range(0, len(records), 64):
                chunk = records[start:start + 64]; expected = expected_rows(chunk, cfg['views'])
                path = out / 'features' / f'{split}-{start // 64:04d}.pt'
                rows = verified_chunk(path, expected, protocol_sha, dimension)
                if rows is not None: reused_checks += len(rows)
                else:
                    rows = []
                    for record in chunk:
                        keys = list(record['questions']['action']['criteria'])
                        for view in cfg['views']:
                            capture.clear(); response = client.system_one(**payload(record, view))
                            if len(capture) != 1: raise ValueError('Expected one pointer feature capture')
                            decide, options = (t.clone().to('cpu', dtype=torch.float32) for t in capture.pop())
                            if torch.is_inference(decide) or torch.is_inference(options): raise ValueError('Autograd-incompatible features')
                            probabilities, _ = validate_distribution(response['answers']['action']['probabilities'], keys)
                            with torch.no_grad(): cached = head(decide, options).softmax(-1)
                            if not torch.allclose(torch.tensor(probabilities), cached, atol=2e-5, rtol=2e-4):
                                raise ValueError('Cache does not reproduce native reference probabilities')
                            rows.append({**expected[len(rows)], 'decide': decide, 'options': options})
                    tmp = path.with_suffix('.pt.tmp'); torch.save(rows, tmp); tmp.replace(path)
                    write(path.with_suffix('.audit.json'), {'protocol_sha256': protocol_sha,
                        'tensor_sha256': digest(path), 'checks': len(rows)})
                    verified_chunk(path, expected, protocol_sha, dimension)
                completed_checks += len(rows); tensors[path.name] = digest(path)
                state = {'status': 'encoding_natural_call_features', 'split': split,
                    'raw_completed': start + len(chunk), 'raw_total': len(records), 'completed_probability_checks': completed_checks,
                    'reused_probability_checks': reused_checks, 'seconds': time.monotonic() - started,
                    'test_opened': False, 'parameter_updates': 0}
                write(out / 'status.json', state); print(json.dumps(state), flush=True)
    finally: hook.remove()
    expected_checks = sum(audit['counts'][s] for s in cfg['feature_splits']) * len(cfg['views'])
    if completed_checks != expected_checks: raise ValueError('Incomplete probability audit')
    if {p.name for p in (out / 'features').glob('*.pt')} != set(tensors): raise ValueError('Unexpected cached shards')
    if any(digest(reference / name) != h for name, h in protocol['reference_files'].items()): raise ValueError('Reference files changed during encoding')
    result = {'status': 'feature_cache_complete', 'native_feature_probability_checks': completed_checks,
        'reused_probability_checks': reused_checks, 'reference_files_unchanged': True,
        'feature_splits': cfg['feature_splits'], 'tensor_sha256': tensors, 'parameter_updates': 0, 'test_opened': False}
    write(out / 'runtime-audit.json', result); write(out / 'status.json', result)
    print(json.dumps({'status': result['status'], 'checks': completed_checks, 'test_opened': False}), flush=True)


if __name__ == '__main__': main()
