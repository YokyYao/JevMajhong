"""Semantic reserved-seed checks for explicit hash-bound raw rollout shards.

This checks stored source artifacts only. It cannot prove complete ancestry,
reproduce labels, admit a candidate or run a reserved native environment.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def semantic_findings(value, seed_start=2026300000, groups=32):
    import torch
    if type(seed_start) is not int or type(groups) is not int or groups <= 0:
        raise ValueError('Explicit positive reserved logical-seed interval required')
    pattern = re.compile(r'(?<![0-9])(?:'+'|'.join(str(s) for s in range(seed_start,seed_start+groups))+r')(?![0-9])')
    counts = Counter(); hits = []; total = 0; seen = set(); stack = [('', value)]
    def finding(path, kind, seed):
        nonlocal total
        total += 1
        if len(hits) < 32: hits.append(dict(path=path, kind=kind, seed=seed))
    while stack:
        path, current = stack.pop(); counts['objects'] += 1
        if current is None or type(current) is bool: continue
        if type(current) in (int, float):
            counts['numeric_scalars'] += 1
            if type(current) is float and not math.isfinite(current): counts['nonfinite_scalars'] += 1
            elif seed_start <= current < seed_start+groups: finding(path, type(current).__name__, current)
        elif isinstance(current, (str, bytes, bytearray)):
            counts['strings_or_bytes'] += 1
            text = current if isinstance(current,str) else bytes(current).decode('latin1')
            for match in pattern.finditer(text): finding(path, type(current).__name__, int(match[0]))
        elif isinstance(current, torch.Tensor):
            if id(current) in seen: continue
            seen.add(id(current)); counts['tensors'] += 1; counts['tensor_elements'] += current.numel()
            if current.device.type != 'cpu' or current.layout != torch.strided or current.is_quantized:
                raise ValueError('Only actual CPU strided unquantized raw-data tensors supported')
            if current.dtype == torch.bool: continue
            flat = torch.view_as_real(current).reshape(-1) if current.is_complex() else current.reshape(-1)
            for start in range(0, flat.numel(), 262144):
                chunk = flat[start:start+262144].to(torch.float64)
                counts['nonfinite_tensor_elements'] += int((~torch.isfinite(chunk)).sum())
                selected = torch.nonzero((chunk>=seed_start)&(chunk<seed_start+groups)).reshape(-1)
                for index in selected.tolist(): finding(path+'/'+str(start+index), str(current.dtype), float(chunk[index]))
        elif isinstance(current, dict):
            if id(current) in seen: continue
            seen.add(id(current)); counts['mappings'] += 1
            for index,(key,child) in enumerate(current.items()):
                stack.append((path+'/key:'+str(index),key)); stack.append((path+'/'+str(key),child))
        elif isinstance(current, (list,tuple,set,frozenset)):
            if id(current) in seen: continue
            seen.add(id(current)); counts['sequences'] += 1
            stack.extend((path+'/'+str(index),child) for index,child in enumerate(current))
        elif isinstance(current, (torch.dtype, torch.device)):
            counts['torch_type_metadata'] += 1
            for match in pattern.finditer(str(current)): finding(path,type(current).__name__,int(match[0]))
        else:
            raise ValueError('Unclassified binary object type: '+type(current).__module__+'.'+type(current).__name__)
    return dict(counts=dict(counts), reserved_occurrences=total, first_hits_private=hits,
        stored_objects_only_not_original_precast_values=True)


def scan_raw_rollout(path, expected_sha256, seed_start=2026300000, groups=32):
    import torch
    path = Path(path).resolve()
    if path.suffix != '.pt' or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Explicit raw .pt source and expected SHA256 required')
    before = path.stat()
    with path.open('rb') as stream: digest = hashlib.file_digest(stream,'sha256').hexdigest()
    if digest != expected_sha256: raise ValueError('Original raw binary source hash differs')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or not isinstance(payload.get('rows'),list):
        raise ValueError('Actual raw rollout rows schema required; model weights are not source records')
    episodes = 0; future_seeds = 0; future_windows = 0
    for row in payload['rows']:
        if not isinstance(row,dict): raise ValueError('Raw rollout row mapping required')
        if 'episode_key' in row:
            if not isinstance(row['episode_key'],(list,tuple)) or not row['episode_key'] or type(row['episode_key'][0]) is not int:
                raise ValueError('Unclassified native episode-key seed encoding')
            episodes += 1
        if 'counterfactual_worlds' in row:
            if not isinstance(row['counterfactual_worlds'],list): raise ValueError('Explicit complete future-world list required')
            future_windows += 1
            for world in row['counterfactual_worlds']:
                if not isinstance(world,dict) or type(world.get('future_seed')) is not int:
                    raise ValueError('Unclassified future-world seed encoding')
                future_seeds += 1
    findings = semantic_findings(payload,seed_start,groups); after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError('Raw source changed during binary semantic scan')
    return dict(path=str(path),sha256=digest,bytes_scanned=before.st_size,
        raw_rows=len(payload['rows']),native_episode_seed_fields=episodes,
        counterfactual_windows=future_windows,future_seed_fields=future_seeds,**findings,
        actual_raw_rollout_schema_verified=True,weights_only_loader=True,stable_during_scan=True,
        source_subset_only_not_complete_candidate_ancestry=True,confirmation_admitted=False)
