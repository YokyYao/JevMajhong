"""Create individually verifiable runtime archives below GitHub's asset limit."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ASSET_LIMIT = 2 * 1024**3


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package(source, output, group_limit=1_500_000_000):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in source.rglob('*') if p.is_file()),
                   key=lambda p: (-p.stat().st_size, p.as_posix()))
    if not (source / 'JevMahjong.exe').is_file():
        raise ValueError('Missing frozen executable')
    groups = []
    for path in files:
        size = path.stat().st_size
        group = next((g for g in groups if g['bytes'] + size <= group_limit), None)
        if group is None:
            group = {'bytes': 0, 'files': []}
            groups.append(group)
        group['files'].append(path)
        group['bytes'] += size
    entries, records = [], []
    for index, group in enumerate(groups, 1):
        name = f'JevMahjong-Runtime-0.2.1-{index:02}.zip'
        archive = output / name
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
            for path in group['files']:
                bundle.write(path, path.relative_to(source).as_posix())
        if archive.stat().st_size >= ASSET_LIMIT:
            raise ValueError(f'Archive exceeds GitHub limit: {name}')
        digest = sha256(archive)
        records.append({'name': name, 'sha256': digest, 'bytes': archive.stat().st_size,
                        'installed_bytes': group['bytes'], 'files': len(group['files'])})
        entries.append(f'Source: "{{code:RuntimeBaseUrl}}/{name}"; DestName: "{name}"; '
                       f'DestDir: "{{app}}"; ExternalSize: {group["bytes"]}; Hash: "{digest}"; '
                       'Flags: external download extractarchive recursesubdirs createallsubdirs ignoreversion')
    (output / 'runtime-files.iss').write_text('\n'.join(entries) + '\n', encoding='utf-8')
    (output / 'runtime-manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('dist/JevMahjong'))
    parser.add_argument('--output', type=Path, default=Path('release'))
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output), indent=2))
