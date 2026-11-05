"""Download the pinned detector; no screenshots or game state leave this device."""
import hashlib
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop.jev_sidecar.vision import DETECTOR_SHA, DETECTOR_URL


def main():
    target = ROOT / 'assets/tiles.onnx'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == DETECTOR_SHA:
        print('Tile detector verified.')
        return
    temporary = target.with_suffix('.download')
    with urllib.request.urlopen(DETECTOR_URL, timeout=120) as response:
        temporary.write_bytes(response.read())
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != DETECTOR_SHA:
        temporary.unlink()
        raise ValueError('Detector checksum mismatch; no model installed')
    temporary.replace(target)
    print('Tile detector installed and verified.')


def font():
    target = ROOT / 'assets/NotoSansSC.ttf'
    expected = 'a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return
    url = 'https://raw.githubusercontent.com/google/fonts/fbe81817505e8aeadc53cc7a38a2b76fb57c333e/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf'
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Chinese font checksum mismatch')
    target.write_bytes(content)


if __name__ == '__main__':
    main()
    font()
