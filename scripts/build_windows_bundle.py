"""Create a private Windows source distribution with the confirmed local checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop.jev_sidecar.engine import CHECKPOINT_SHA
from desktop.jev_sidecar.vision import DETECTOR_SHA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, checkpoint, detector):
    if sha(checkpoint / 'head.pt') != CHECKPOINT_SHA or sha(detector) != DETECTOR_SHA:
        raise ValueError('Use the verified incumbent and pinned detector')
    if output.exists():
        raise ValueError('Preserve existing distribution; choose a new output')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.windows-') as temporary:
        staged = Path(temporary) / 'JevMahjong-Windows'
        staged.mkdir()
        shutil.copytree(ROOT / 'desktop', staged / 'desktop', ignore=shutil.ignore_patterns('__pycache__', 'tests'))
        for p in (ROOT / 'desktop/windows').iterdir():
            shutil.copy2(p, staged / p.name)
        shutil.copy2(ROOT / 'desktop/README.md', staged / 'README-Windows.md')
        (staged / 'models/calibration-02').mkdir(parents=True)
        names = ['head.pt', 'adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']
        for name in names:
            shutil.copy2(checkpoint / name, staged / 'models/calibration-02' / name)
        (staged / 'models/calibration-02/manifest.json').write_text(json.dumps({name: sha(checkpoint / name) for name in names}, indent=2))
        (staged / 'assets').mkdir()
        shutil.copy2(detector, staged / 'assets/tiles.onnx')
        shutil.copy2(ROOT / 'runs/windows-assets/NotoSansSC.ttf', staged / 'assets/NotoSansSC.ttf')
        manifest = {str(p.relative_to(staged)): sha(p) for p in staged.rglob('*') if p.is_file()}
        (staged / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for p in staged.rglob('*'):
                if p.is_file():
                    archive.write(p, Path(staged.name) / p.relative_to(staged))
    return {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output),
            'checkpoint_head_sha256': CHECKPOINT_SHA, 'contains_training_records': False,
            'format': 'Windows Python source bundle and local checkpoint; not a precompiled EXE'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/head-rl-diagonal-value-20261030/checkpoints/calibration-02')
    parser.add_argument('--detector', type=Path, default=ROOT / 'runs/windows-assets/tiles.onnx')
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.checkpoint, args.detector)))
