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


def package(source, output, group_limit=1_500_000_000, version='0.4.4'):
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
        name = f'JevMahjong-Runtime-{version}-{index:02}.zip'
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



def safe_relative(name):
    from pathlib import PurePosixPath
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe managed path: {name}')
    return name


def runtime_file(name):
    return name.startswith('_internal/') and not (
        name.startswith('_internal/desktop/') or name == '_internal/base_library.zip')


def write_checks(path, hashes):
    path.write_text(''.join(f'{value}  {name}\n' for name, value in sorted(hashes.items())),
                    encoding='utf-8')


def package_upgrade(source, output, baseline, baseline_config, version, group_limit=1_500_000_000,
                    compatible_manifests=()):
    source, output, baseline = Path(source), Path(output), Path(baseline)
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((baseline / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    current = {safe_relative(p.relative_to(source).as_posix()): sha256(p)
               for p in source.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.json'}
    if 'JevMahjong.exe' not in current or not any(n.startswith('models/') for n in current):
        raise ValueError('Missing application or model')
    installed_manifest = json.loads((source / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    if current != installed_manifest:
        raise ValueError('Frozen application does not match its file manifest')
    current['SHA256SUMS.json'] = sha256(source / 'SHA256SUMS.json')
    compatible = [previous] + [json.loads(Path(path).read_text(encoding='utf-8'))
                               for path in compatible_manifests]
    common = {n: h for n, h in current.items()
              if runtime_file(n) and all(index.get(n) == h for index in compatible)}
    models = {n: h for n, h in current.items() if n.startswith('models/')}
    app = {n: h for n, h in current.items() if n not in common and n not in models}
    if not common:
        raise ValueError('No compatible baseline runtime; make a new full baseline first')
    for hashes in (previous, current):
        for name in hashes:
            safe_relative(name)
    write_checks(output / 'reuse-runtime.sha256', common)
    write_checks(output / 'reuse-app.sha256', app)
    write_checks(output / 'reuse-model.sha256', models)
    model_in_baseline = all(previous.get(n) == h for n, h in models.items())
    entries, components = [], []

    def installer_entry(record, function, url_function):
        entries.append(f'Source: "{{code:{url_function}}}/{record["name"]}"; '
                       f'DestName: "{record["name"]}"; DestDir: "{{app}}"; '
                       f'ExternalSize: {record["installed_bytes"]}; Hash: "{record["sha256"]}"; '
                       f'Check: {function}; '
                       'Flags: external download extractarchive recursesubdirs createallsubdirs ignoreversion')

    for record in baseline_config['archives']:
        installer_entry(record, 'NeedBaseline', 'BaselineBaseUrl')
    for kind, hashes, check in [('Application', app, 'NeedApplication'), ('Model', models, 'NeedModel')]:
        groups = []
        for name in sorted(hashes, key=lambda n: (-(source / n).stat().st_size, n)):
            size = (source / name).stat().st_size
            group = next((g for g in groups if g['bytes'] + size <= group_limit), None)
            if group is None:
                group = {'bytes': 0, 'files': []}
                groups.append(group)
            group['files'].append(name)
            group['bytes'] += size
        for index, group in enumerate(groups, 1):
            name = f'JevMahjong-{kind}-{version}-{index:02}.zip'
            archive = output / name
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
                for filename in group['files']:
                    bundle.write(source / filename, filename)
            if archive.stat().st_size >= ASSET_LIMIT:
                raise ValueError(f'Archive exceeds asset limit: {name}')
            record = {'kind': kind.lower(), 'name': name, 'sha256': sha256(archive),
                      'bytes': archive.stat().st_size, 'installed_bytes': group['bytes'],
                      'files': len(group['files'])}
            components.append(record)
            installer_entry(record, check, 'RuntimeBaseUrl')
    (output / 'runtime-files.iss').write_text('\n'.join(entries) + '\n', encoding='utf-8')
    (output / 'reuse-constants.iss').write_text(
        '#define ModelInBaseline "' + ('True' if model_in_baseline else 'False') + '"\n',
        encoding='utf-8')
    manifest = {'format': 'jevmahjong-components-v1', 'version': version,
                'baseline_tag': baseline_config['tag'], 'baseline_archives': baseline_config['archives'],
                'required_runtime_files': common, 'application_files': app, 'model_files': models,
                'model_in_baseline': model_in_baseline, 'components': components,
                'reused_runtime_bytes': sum((source / n).stat().st_size for n in common),
                'application_download_bytes': sum(c['bytes'] for c in components if c['kind'] == 'application'),
                'model_download_bytes': sum(c['bytes'] for c in components if c['kind'] == 'model')}
    (output / 'runtime-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('dist/JevMahjong'))
    parser.add_argument('--output', type=Path, default=Path('release'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--baseline-config', type=Path, default=Path('desktop/windows/runtime-baseline.json'))
    parser.add_argument('--version', default='0.4.4')
    parser.add_argument('--compatible-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.baseline:
        result = package_upgrade(args.source, args.output, args.baseline,
                                 json.loads(args.baseline_config.read_text(encoding='utf-8')), args.version,
                                 compatible_manifests=args.compatible_manifest)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ('required_runtime_files', 'application_files', 'model_files')}, indent=2))
    else:
        print(json.dumps(package(args.source, args.output, version=args.version), indent=2))
