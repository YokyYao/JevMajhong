"""Download and verify a published runtime baseline on a build/test worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen
import zipfile


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(config, cache, output, manifest_only=False):
    if output.exists():
        raise FileExistsError(output)
    cache.mkdir(parents=True, exist_ok=True)
    base_url = f'https://github.com/{config["repository"]}/releases/download/{config["tag"]}'

    def download(record):
        target = cache / record['name']
        if target.exists() and target.stat().st_size == record['bytes'] and file_hash(target) == record['sha256']:
            return target
        temporary = target.with_suffix('.download')
        for attempt in range(3):
            try:
                with urlopen(base_url + '/' + record['name'], timeout=60) as response, temporary.open('wb') as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                if temporary.stat().st_size != record['bytes'] or file_hash(temporary) != record['sha256']:
                    raise ValueError(f'Baseline archive checksum failed: {record["name"]}')
                temporary.replace(target)
                print('Verified baseline archive:', target.name, flush=True)
                return target
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        archives = list(executor.map(download, config['archives']))
    output.parent.mkdir(parents=True, exist_ok=True)
    if manifest_only:
        for archive in archives:
            with zipfile.ZipFile(archive) as bundle:
                if 'SHA256SUMS.json' in bundle.namelist():
                    data = bundle.read('SHA256SUMS.json')
                    json.loads(data)
                    output.write_bytes(data)
                    return {'baseline': config['tag'], 'manifest_only': True, 'archives_verified': len(archives)}
        raise ValueError('Baseline file manifest not found')
    with tempfile.TemporaryDirectory(prefix='baseline-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'app'
        stage.mkdir()
        for archive in archives:
            with zipfile.ZipFile(archive) as bundle:
                for name in bundle.namelist():
                    path = Path(name)
                    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                        raise ValueError('Unsafe baseline archive path')
                bundle.extractall(stage)
        if not (stage / 'SHA256SUMS.json').is_file() or not (stage / 'JevMahjong.exe').is_file():
            raise ValueError('Incomplete baseline')
        stage.replace(output)
    return {'baseline': config['tag'], 'archives_verified': len(archives), 'path': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('desktop/windows/runtime-baseline.json'))
    parser.add_argument('--cache', type=Path, default=Path('runs/baseline-downloads'))
    parser.add_argument('--out', type=Path, default=Path('runs/runtime-baseline'))
    parser.add_argument('--manifest-only', action='store_true', help='Extract only the verified file index for compatibility comparisons')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(fetch(config, args.cache, args.out, args.manifest_only)))
