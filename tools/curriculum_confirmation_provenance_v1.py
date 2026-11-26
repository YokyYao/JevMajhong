"""Streaming reserved-seed literal checks on explicitly inventoried artifacts.

This is file-backed evidence only. A clean subset is never global proof that
a test grid was unused; admission must bind the full actual ancestry/catalog.
"""
import hashlib
from pathlib import Path
import re


def scan_jsonl(paths,seed_start=2026300000,groups=32):
    if type(seed_start) is not int or type(groups) is not int or groups<=0:
        raise ValueError('Explicit positive reserved logical-seed interval required')
    seeds=range(seed_start,seed_start+groups)
    pattern=re.compile(rb'(?<![0-9])(?:'+b'|'.join(str(seed).encode() for seed in seeds)+rb')(?![0-9])')
    canonical=sorted({Path(path).resolve() for path in paths})
    if not canonical or any(not path.is_file() or path.suffix!='.jsonl' for path in canonical):
        raise ValueError('Nonempty complete explicit JSONL file inventory required')
    receipts=[];hits=[];total_hits=0
    for path in canonical:
        before=path.stat();digest=hashlib.sha256();records=0;size=0
        with path.open('rb') as stream:
            for line_number,line in enumerate(stream,1):
                digest.update(line);size+=len(line);records+=bool(line.strip())
                for match in pattern.finditer(line):
                    total_hits+=1
                    if len(hits)<32:hits.append(dict(path=str(path),line=line_number,seed=int(match[0])))
        after=path.stat()
        receipts.append(dict(path=str(path),sha256=digest.hexdigest(),bytes_scanned=size,nonempty_lines=records,
            stable_during_scan=(before.st_size==after.st_size==size and before.st_mtime_ns==after.st_mtime_ns)))
    return dict(method='explicit_jsonl_stream_reserved_seed_literals_v1',seed_start=seed_start,seed_groups=groups,
        files=receipts,total_files=len(receipts),total_bytes=sum(r['bytes_scanned'] for r in receipts),
        total_nonempty_lines=sum(r['nonempty_lines'] for r in receipts),reserved_literal_occurrences=total_hits,
        first_hits_private=hits,all_inventoried_files_stable=all(r['stable_during_scan'] for r in receipts),
        checked_files_only_not_global_unused_grid_proof=True,
        JSON_or_binary_artifacts_not_checked_by_this_function=True,
        test_simulations_or_model_predictions_generated=0,confirmation_admitted=False)
