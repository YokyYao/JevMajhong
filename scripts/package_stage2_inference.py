"""Package only the authorized stage2 inference files, never optimizer or logs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from clean_policy_data import POLICY_INSTRUCTION

FILES = ('adapter_config.json', 'adapter_model.safetensors', 'chat_template.jinja',
         'head.pt', 'tokenizer.json', 'tokenizer_config.json')


def main():
    import torch
    from desktop.jev_sidecar.model_identity import tensor_sha, IDENTITY
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT /
        'runs/feature-unseen-continuation-20261110/training-run/checkpoints/step-000200')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); checkpoint = args.checkpoint
    completed = completed_checkpoint_files(checkpoint, 200)
    expected_head = '8f3421dca9a700c63d2a7b1694a81c80a0ef9f29c263a13a75fb13e426452a04'
    expected_adapter = '7788cff96a921f1358f0e68fa6ee16f69782174c106e98c6a11008acdff0becb'
    if completed['head.pt'] != expected_head or completed['adapter_model.safetensors'] != expected_adapter:
        raise ValueError('Only the preserved stage2 candidate is authorized here')
    head = torch.load(checkpoint / 'head.pt', map_location='cpu', weights_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    identity = {'model_id': 'feature-stage2-step200', 'optimizer_step': 200,
        'files': {name: completed[name] for name in FILES},
        'runtime': {k: head[k] for k in IDENTITY['runtime']},
        'head_tensor_sha256': tensor_sha(head['head']), 'policy_instruction': POLICY_INSTRUCTION,
        'policy_scope': 'chi/pon/pass only; ordinary and rare actions use oldB',
        'ordinary_head_sha256': IDENTITY['source_head_sha256']}
    (ROOT / 'desktop/jev_sidecar/stage2_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    archive = args.out / 'checkpoint.tar.gz'
    with archive.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=3) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|') as tar:
            for name in FILES:
                source = checkpoint / name; info = tar.gettarinfo(str(source), arcname=name)
                info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''; info.mode = 0o644
                with source.open('rb') as stream: tar.addfile(info, stream)
    parts = []
    with archive.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            name = f'checkpoint.tar.gz.part-{len(parts):03d}'
            (args.out / name).write_bytes(block)
            git_sha = hashlib.sha1(f'blob {len(block)}\0'.encode() + block).hexdigest()
            parts.append({'name': name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest(),
                          'git_blob_sha1': git_sha})
    manifest = {'format': 'jevmahjong-checkpoint-parts-v1', 'optimizer_step': 200, 'world_size': 1,
        'model_id': identity['model_id'], 'inference_only': True, 'base': identity['runtime']['base'],
        'base_revision': identity['runtime']['base_revision'], 'archive_sha256': digest(archive),
        'archive_bytes': archive.stat().st_size,
        'files': [{'name': name, 'bytes': (checkpoint / name).stat().st_size,
                   'sha256': completed[name]} for name in FILES], 'parts': parts}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # The transport needs parts only, avoiding an extra copy in Git history.
    archive.unlink()
    print(json.dumps({'inference_files': len(FILES), 'parts': len(parts),
        'archive_bytes': manifest['archive_bytes'], 'optimizer_included': False}))


if __name__ == '__main__': main()
