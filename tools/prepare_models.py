"""Restore published inference files and exactly reconstruct the confirmed score head."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(original, target):
    import torch
    from desktop.jev_sidecar.model_identity import IDENTITY, validate_checkpoint
    recipe = json.loads((ROOT / 'weights/jev-discard-v1/recipe.json').read_text())
    if target.exists():
        raise ValueError('Use a new model output directory')
    if hashlib.sha256((original / 'head.pt').read_bytes()).hexdigest() != recipe['original_head_sha256']:
        raise ValueError('Original published checkpoint changed')
    original_payload = torch.load(original / 'head.pt', map_location='cpu', weights_only=True)
    state = {k: v.clone() for k, v in original_payload['head'].items()}
    factor = torch.tensor(recipe['query_row_factors'], dtype=state['q.weight'].dtype)
    state['q.weight'].mul_(factor[:, None])
    state['q.bias'].mul_(factor)
    target.mkdir(parents=True)
    payload = {**IDENTITY['runtime'], 'head': state, 'training_method': 'exact confirmed diagonal calibration reconstruction'}
    torch.save(payload, target / 'head.pt')
    for filename in IDENTITY['base_files']:
        shutil.copy2(original / filename, target / filename)
    proof = validate_checkpoint(target)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=ROOT / 'runs/release-original')
    parser.add_argument('--out', type=Path, default=ROOT / 'models/jev-discard-v1')
    parser.add_argument('--call-out', type=Path, default=ROOT / 'models/jev-call-v1')
    args = parser.parse_args()
    if not args.original.exists():
        subprocess.run([sys.executable, str(ROOT / 'tools/restore_weights.py'),
                        str(ROOT / 'weights/jev-supervised-v1/manifest.json'), '--out', str(args.original)], check=True)
    print(json.dumps(prepare(args.original, args.out)))
    subprocess.run([sys.executable, str(ROOT / 'tools/restore_weights.py'),
                    str(ROOT / 'weights/jev-call-v1/manifest.json'),
                    '--out', str(args.call_out)], check=True)
    from desktop.jev_sidecar.call_policy import validate_call_checkpoint
    print(json.dumps(validate_call_checkpoint(args.call_out)))
