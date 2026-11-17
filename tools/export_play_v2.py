"""Export the pinned v8 final parameters as an inference-only Jev Play v2 bundle."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_IDENTITY = '5ebce6af3777854cbec55ba29db2dc90be0f959193114bb3154179c5552370fa'
SOURCE_FILES = {
    'adapter_model.safetensors': 'acfef83ec1ea91c5c929114d1f3fae1292ccce9930a20f70bd78b4ef379ac65a',
    'adapter_config.json': '372210475d6e99face2d6bce7103bf9b17157a56cd4f64ac72ace47f72e74d65',
    'head.pt': '03a63159c8fd70218aee25b1b7550fdf6cc7dfb0314b46953bf1b4b208ba671b',
    'tokenizer.json': 'a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f',
    'tokenizer_config.json': 'e50492ec49b138a18ea6a3b85a6a8765c856fa2e10297ebfbbeb96e27405bf68',
    'chat_template.jinja': '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(source, output, archive_dir, part_bytes):
    import torch
    from desktop.jev_sidecar.model_identity import IDENTITY, tensor_sha
    from desktop.jev_sidecar.unified_policy import identity
    INSTRUCTION = identity()['policy_instruction']
    source, output, archive_dir = map(Path, (source, output, archive_dir))
    if output.exists() or archive_dir.exists():
        raise ValueError('Choose new inference and archive directories')
    files = {name: digest(source / name) for name in SOURCE_FILES}
    if files != SOURCE_FILES:
        raise ValueError('Expected the registered v8 final-2088 checkpoint')
    if hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != SOURCE_IDENTITY:
        raise ValueError('Source checkpoint identity differs')
    original = torch.load(source / 'head.pt', map_location='cpu', weights_only=True)
    runtime = dict(IDENTITY['runtime'])
    if any(original.get(k) != v for k, v in runtime.items()):
        raise ValueError('Unexpected source runtime')
    output.mkdir(parents=True)
    for name in files:
        if name != 'head.pt':
            shutil.copy2(source / name, output / name)
    # Training args contain server paths and seeds; only runtime and tensors ship.
    torch.save({**runtime, 'head': original['head']}, output / 'head.pt')
    restored = torch.load(output / 'head.pt', map_location='cpu', weights_only=True)
    if set(restored) != set(runtime) | {'head'} or any(
            not torch.equal(restored['head'][k], v) for k, v in original['head'].items()):
        raise ValueError('Inference export changed model parameters')
    exported = {name: digest(output / name) for name in files}
    release = {'model_id': 'jev-play-v2', 'display_name': 'Jev Play v2',
        'source_model_id': 'jev-unified-attackrisk-v8', 'source_step': 2088,
        'source_identity_sha256': SOURCE_IDENTITY, 'source_head_sha256': SOURCE_FILES['head.pt'],
        'identity_sha256': hashlib.sha256(json.dumps(exported, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'files': exported, 'runtime': runtime, 'head_tensor_sha256': tensor_sha(original['head']),
        'policy_instruction': INSTRUCTION, 'single_policy_for_all_actions': True,
        'inference_only_export': True, 'parameter_values_unchanged': True,
        'independent_improvement_verified': False}
    text = json.dumps(release, ensure_ascii=False, indent=2) + '\n'
    (output / 'play-v2-identity.json').write_text(text, encoding='utf-8')
    (ROOT / 'desktop/jev_sidecar/play_v2_identity.json').write_text(text, encoding='utf-8')
    archive_dir.mkdir(parents=True)
    archive = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=archive, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as bundle:
            for path in sorted(output.iterdir()):
                content = path.read_bytes()
                member = tarfile.TarInfo(path.name)
                member.size, member.mtime, member.mode = len(content), 0, 0o644
                bundle.addfile(member, io.BytesIO(content))
    content = archive.getvalue()
    parts = []
    for offset in range(0, len(content), part_bytes):
        name = f'checkpoint.part{len(parts):03d}'
        chunk = content[offset:offset + part_bytes]
        (archive_dir / name).write_bytes(chunk)
        parts.append({'name': name, 'bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest()})
    manifest = {'format': 'jevmahjong-checkpoint-parts-v1', 'model_id': 'jev-play-v2',
        'optimizer_step': 2088, 'source_model_id': release['source_model_id'],
        'source_identity_sha256': SOURCE_IDENTITY, 'inference_only_export': True,
        'parameter_values_unchanged': True, 'archive_bytes': len(content),
        'archive_sha256': hashlib.sha256(content).hexdigest(), 'parts': parts,
        'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(output.iterdir())]}
    (archive_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return {'model_id': 'jev-play-v2', 'parts': len(parts), 'archive_bytes': len(content),
        'identity_sha256': release['identity_sha256'], 'parameters_unchanged': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'models/jev-play-v2')
    p.add_argument('--archive-dir', type=Path, default=ROOT / 'weights/jev-play-v2')
    p.add_argument('--part-bytes', type=int, default=1024 * 1024)
    a = p.parse_args()
    if a.part_bytes <= 0:
        p.error('Positive part size required')
    print(json.dumps(export(a.source, a.out, a.archive_dir, a.part_bytes)))
