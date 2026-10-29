"""Restore an inference checkpoint from a verified frozen base and a small head patch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def fingerprint(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_name(name):
    if not name or Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('Patch manifest contains an invalid filename')
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists; use a new directory')
    spec = json.loads(args.manifest.read_text())
    if spec.get('format') != 'jevmahjong-frozen-adapter-head-patch-v1':
        raise ValueError('Unsupported head patch format')
    names = [safe_name(f['name']) for f in spec['base_files']]
    required = {'adapter_config.json', 'adapter_model.safetensors', 'chat_template.jinja',
                'tokenizer.json', 'tokenizer_config.json'}
    if set(names) != required or len(names) != len(required):
        raise ValueError('Base checkpoint file set is incomplete or duplicated')
    if fingerprint(args.base/'head.pt') != spec['base_head_sha256']:
        raise ValueError('This patch requires the specified original base checkpoint')
    for item in spec['base_files']:
        path = args.base/item['name']
        if path.stat().st_size != item['bytes'] or fingerprint(path) != item['sha256']:
            raise ValueError('Frozen base file checksum mismatch: '+item['name'])
    head = spec['head_file']
    if safe_name(head['name']) != 'head.pt':
        raise ValueError('The replacement head must be named head.pt')
    patch = args.manifest.parent/head['name']
    if patch.stat().st_size != head['bytes'] or fingerprint(patch) != head['sha256']:
        raise ValueError('Replacement head checksum mismatch')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix='head-patch-') as temporary:
        stage = Path(temporary)/'model'; stage.mkdir()
        for name in names:
            shutil.copy2(args.base/name, stage/name)
        shutil.copy2(patch, stage/'head.pt')
        (stage/'head-patch-provenance.json').write_text(json.dumps(spec, indent=2)+'\n')
        if args.out.exists():
            raise ValueError('Output was created concurrently; refusing to replace it')
        stage.rename(args.out)
    print(json.dumps({'restored': str(args.out), 'files_verified': len(names)+1,
                      'head_sha256':head['sha256'], 'training_state_included':False}))


if __name__ == '__main__':
    main()
