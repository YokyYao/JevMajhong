"""Add only inference assets and corresponding source to a frozen desktop build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package(target):
    from desktop.jev_sidecar.model_identity import validate_checkpoint
    if not (target / 'JevMahjong.exe').is_file():
        raise ValueError('Build the Windows EXE first')
    validate_checkpoint(ROOT / 'models/calibration-02')
    shutil.copytree(ROOT / 'models/calibration-02', target / 'models/calibration-02')
    (target / 'assets').mkdir()
    for name in ['tiles.onnx', 'NotoSansSC.ttf']:
        shutil.copy2(ROOT / 'assets' / name, target / 'assets' / name)
    shutil.copytree(ROOT / 'desktop', target / 'desktop-source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'desktop/README.md', target / 'README.md')
    shutil.copy2(ROOT / 'desktop/LICENSE', target / 'LICENSE')
    shutil.copy2(ROOT / 'desktop/THIRD_PARTY_NOTICES.md', target / 'THIRD_PARTY_NOTICES.md')
    shutil.copytree(ROOT / 'desktop/licenses', target / 'licenses')
    for path in target.rglob('*.pdb'):
        path.unlink()
    manifest = {str(p.relative_to(target)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in target.rglob('*') if p.is_file()}
    (target / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(manifest), 'bytes': sum(p.stat().st_size for p in target.rglob('*') if p.is_file())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, required=True)
    package(parser.parse_args().dist)
