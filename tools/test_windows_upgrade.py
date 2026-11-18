"""Exercise real Windows upgrades, repair and first installation against a local download server."""
import argparse
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.parse import unquote, urlsplit


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def execute(command, *, timeout=900, expect_success=True, env=None):
    process = subprocess.Popen([str(v) for v in command], env=env)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           timeout=30, check=False)
            process.wait(timeout=30)
        raise
    if (code == 0) != expect_success:
        raise AssertionError(f'Unexpected exit {code}: {command[0]}')
    return code


def verify_install(target):
    expected = json.loads((target / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, value in expected.items():
        if not (target / name).is_file() or digest(target / name) != value:
            raise AssertionError(f'Installed file mismatch: {name}')


def test(release, baseline, cache, base_model, output, label, upgrade_only=False):
    manifest = json.loads((release / 'runtime-manifest.json').read_text(encoding='utf-8'))
    installer = release / f'JevMahjong-Setup-{manifest["version"]}-win-x64.exe'
    if manifest['application_download_bytes'] > 200_000_000:
        raise AssertionError('Application update unexpectedly includes a large runtime payload')
    requests = []
    blocked = {'baseline': True}

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, value):
            path = unquote(urlsplit(value).path)
            head, _, name = path.lstrip('/').partition('/')
            if head not in ('current', 'baseline') or '/' in name or '\\' in name or ':' in name or name in ('', '.', '..'):
                return str(release / '__not_found__')
            return str((release if head == 'current' else cache) / name)

        def do_GET(self):
            requests.append({'method': 'GET', 'path': self.path})
            if blocked['baseline'] and urlsplit(self.path).path.startswith('/baseline/'):
                self.send_error(403, 'Baseline downloads forbidden in upgrade test')
            else:
                super().do_GET()

        def do_HEAD(self):
            requests.append({'method': 'HEAD', 'path': self.path})
            if blocked['baseline'] and urlsplit(self.path).path.startswith('/baseline/'):
                self.send_error(403, 'Baseline downloads forbidden in upgrade test')
            else:
                super().do_HEAD()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name='installer-test-http', daemon=True)
    thread.start()
    address = f'http://127.0.0.1:{server.server_address[1]}'
    output.mkdir(parents=True, exist_ok=True)
    settings = Path(os.environ['LOCALAPPDATA']) / 'JevMahjong/settings.json'
    settings.parent.mkdir(parents=True, exist_ok=True)
    saved_settings = settings.read_bytes() if settings.exists() else None
    sentinel = b'{"regions": {}, "upgrade_calibration_sentinel": "keep-me"}'
    settings.write_bytes(sentinel)
    cases = {}
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', JEV_BASE_MODEL_DIR=str(base_model),
               PATH=os.environ['SystemRoot'] + r'\System32;' + os.environ['SystemRoot'])
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)

    def install(target, name, expect_success=True):
        print(f'Starting installer case: {label}/{name}', flush=True)
        start = len(requests)
        log = output / f'{label}-{name}.log'
        try:
            execute([installer, '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                     f'/DIR={target}', f'/RUNTIMEURL={address}/current',
                     f'/BASELINEURL={address}/baseline', f'/LOG={log}'],
                    expect_success=expect_success, timeout=900 if expect_success else 120)
        except BaseException:
            if log.is_file():
                print(log.read_text(encoding='utf-8-sig', errors='replace')[-20000:], flush=True)
            print(json.dumps({'failed_case': name, 'requests': requests[start:]}), flush=True)
            raise
        if settings.read_bytes() != sentinel:
            raise AssertionError('Installer changed user calibration settings')
        observed = requests[start:]
        text = log.read_text(encoding='utf-8-sig', errors='replace')
        cases[name] = {'requests': observed, 'exit_expected_success': expect_success,
                       'decisions': [line.strip() for line in text.splitlines() if 'JEV_UPDATE ' in line]}
        print(f'Installer case passed: {label}/{name}; HTTP requests: {len(observed)}', flush=True)
        return observed

    def no_baseline(observed):
        if any(urlsplit(x['path']).path.startswith('/baseline/') for x in observed):
            raise AssertionError('Upgrade attempted a baseline runtime download')

    def infer(target, name):
        report = output / f'{label}-{name}-selftest.json'
        print(f'Starting installed inference: {label}/{name}', flush=True)
        try:
            execute([target / 'JevMahjong.exe', '--self-test', '--model-test', '--report', report],
                    env=env, timeout=300)
        except BaseException:
            error_log = settings.parent / 'error.log'
            if error_log.is_file():
                shutil.copy2(error_log, output / f'{label}-{name}-error.log')
                print(error_log.read_text(encoding='utf-8', errors='replace'), flush=True)
            raise
        data = json.loads(report.read_text(encoding='utf-8'))
        if not data.get('model') or not data.get('frozen_exe') or data.get('model_id') != 'jev-play-v2':
            raise AssertionError('Installed inference did not run')
        if len(data.get('decisions', {})) != 7:
            raise AssertionError('Incomplete action-path inference')
        print(f'Installed inference passed: {label}/{name}', flush=True)
        cases[name]['inference'] = data
        # The real app may maintain its own settings; the next installer must preserve this sentinel.
        settings.write_bytes(sentinel)

    fresh = baseline.parent / f'fresh-{label}'
    try:
        if not (baseline / 'JevMahjong.exe').is_file():
            raise AssertionError('An actual old installation is required')
        observed = install(baseline, 'upgrade')
        no_baseline(observed)
        if label == '0.4.0' and any('JevMahjong-Model-' in x['path'] for x in observed):
            raise AssertionError('Unchanged v2 model was unnecessarily downloaded')
        verify_install(baseline)
        infer(baseline, 'upgrade')
        if not upgrade_only:
            shutil.rmtree(baseline)
            blocked['baseline'] = False
            observed = install(fresh, 'fresh')
            expected_archives = {r['name'] for r in manifest['baseline_archives']}
            fetched = {Path(urlsplit(x['path']).path).name for x in observed
                       if x['method'] == 'GET' and '/baseline/' in x['path']}
            if fetched != expected_archives:
                raise AssertionError('Fresh install did not fetch the complete baseline')
            verify_install(fresh)
            infer(fresh, 'fresh')
            blocked['baseline'] = True
            observed = install(fresh, 'reinstall')
            if observed:
                raise AssertionError('Reinstall unnecessarily downloaded existing components')
            model_file = next(iter(manifest['model_files']))
            path = fresh / model_file
            path.write_bytes(b'corrupted model')
            observed = install(fresh, 'model-repair')
            no_baseline(observed)
            if not any('JevMahjong-Model-' in x['path'] for x in observed):
                raise AssertionError('Corrupted model was not redownloaded')
            verify_install(fresh)
            damaged = fresh / next(iter(manifest['required_runtime_files']))
            backup = damaged.with_name(damaged.name + '.test-backup')
            damaged.rename(backup)
            try:
                observed = install(fresh, 'runtime-repair-required', expect_success=False)
                if not any('/baseline/' in x['path'] for x in observed):
                    raise AssertionError('Missing runtime did not request repair')
            finally:
                backup.replace(damaged)
        result = {'status': 'passed', 'old_version': label,
                  'application_download_bytes': manifest['application_download_bytes'],
                  'model_download_bytes_when_changed': manifest['model_download_bytes'],
                  'reused_runtime_bytes': manifest['reused_runtime_bytes'], 'cases': cases}
        (output / f'upgrade-{label}-report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError('Test server failed to stop')
        if saved_settings is None:
            settings.unlink(missing_ok=True)
        else:
            settings.write_bytes(saved_settings)
        for path in (baseline, fresh):
            if path.exists():
                shutil.rmtree(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=Path('release'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/runtime-baseline'))
    parser.add_argument('--cache', type=Path, default=Path('runs/baseline-downloads'))
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('windows-upgrade-results'))
    parser.add_argument('--label', default='0.4.0')
    parser.add_argument('--upgrade-only', action='store_true')
    args = parser.parse_args()
    test(args.release.resolve(), args.baseline.resolve(), args.cache.resolve(),
         args.base_model.resolve(), args.output.resolve(), args.label, args.upgrade_only)
