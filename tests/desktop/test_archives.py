import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.package_runtime_archives import package


class ArchiveRoundTrip(unittest.TestCase):
    def test_multiple_runtime_archives_preserve_every_file_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'app', root / 'release'
            (source / '_internal/nested').mkdir(parents=True)
            expected = {'JevMahjong.exe': b'MZ' * 600,
                        '_internal/nested/library.dll': bytes(range(256)) * 8,
                        'models/head.pt': b'weights' * 300,
                        'desktop-source/README.md': '中文源代码'.encode()}
            for name, content in expected.items():
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            records = package(source, output, group_limit=2100)
            self.assertGreater(len(records), 1)
            recovered = {}
            for record in records:
                archive = output / record['name']
                self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), record['sha256'])
                with zipfile.ZipFile(archive) as bundle:
                    for name in bundle.namelist():
                        self.assertNotIn(name, recovered)
                        self.assertNotIn('..', Path(name).parts)
                        recovered[name] = bundle.read(name)
            self.assertEqual(expected, recovered)
            self.assertEqual(sum(r['files'] for r in records), len(expected))
            self.assertEqual(sum(r['installed_bytes'] for r in records), sum(map(len, expected.values())))
            self.assertEqual(records, json.loads((output / 'runtime-manifest.json').read_text()))
            installer = (output / 'runtime-files.iss').read_text()
            for record in records:
                self.assertIn(record['sha256'], installer)
                self.assertIn(record['name'], installer)



class ComponentUpgradeRoundTrip(unittest.TestCase):
    def test_reuses_matching_environment_and_separates_changed_runtime_and_model(self):
        from tools.package_runtime_archives import package_upgrade
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, source, output = root / 'baseline', root / 'app', root / 'release'
            previous = {'JevMahjong.exe': b'old exe',
                        '_internal/shared.dll': b'large shared runtime',
                        '_internal/package-2.dist-info/METADATA': b'new dependency metadata',
                        '_internal/changed.dll': b'old runtime data',
                        '_internal/base_library.zip': b'old generated stdlib',
                        '_internal/desktop/identity.json': b'app-owned data',
                        'models/jev-play-v2/head.pt': b'old model'}
            expected = {**previous, 'JevMahjong.exe': b'new exe',
                        '_internal/changed.dll': b'new runtime data',
                        '_internal/base_library.zip': b'new generated stdlib',
                        'models/jev-play-v2/head.pt': b'new model'}
            for folder, files in [(baseline, previous), (source, expected)]:
                for name, data in files.items():
                    path = folder / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                (folder / 'SHA256SUMS.json').write_text(json.dumps(
                    {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}))
            config = {'tag': 'desktop-v0.4.0', 'archives': [
                {'name': 'baseline.zip', 'sha256': '0' * 64, 'installed_bytes': 100}]}
            legacy_manifest = root / 'legacy.json'
            legacy_manifest.write_text(json.dumps({name: hashlib.sha256(data).hexdigest()
                for name, data in previous.items() if 'dist-info' not in name}))
            result = package_upgrade(source, output, baseline, config, '0.4.1', group_limit=10000,
                                     compatible_manifests=[legacy_manifest])
            self.assertEqual(set(result['required_runtime_files']), {'_internal/shared.dll'})
            self.assertFalse(result['model_in_baseline'])
            self.assertEqual(set(result['model_files']), {'models/jev-play-v2/head.pt'})
            self.assertIn('_internal/changed.dll', result['application_files'])
            self.assertIn('_internal/package-2.dist-info/METADATA', result['application_files'])
            self.assertIn('_internal/desktop/identity.json', result['application_files'])
            self.assertIn('_internal/base_library.zip', result['application_files'])
            target = root / 'upgraded'
            shutil.copytree(baseline, target)
            for record in result['components']:
                archive = output / record['name']
                self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), record['sha256'])
                with zipfile.ZipFile(archive) as bundle:
                    if record['kind'] == 'model':
                        self.assertTrue(all(n.startswith('models/') for n in bundle.namelist()))
                    self.assertNotIn('_internal/shared.dll', bundle.namelist())
                    bundle.extractall(target)
            for name, data in expected.items():
                self.assertEqual((target / name).read_bytes(), data)
            self.assertEqual((target / 'SHA256SUMS.json').read_bytes(), (source / 'SHA256SUMS.json').read_bytes())

    def test_unchanged_model_can_be_reused_after_baseline_repair(self):
        from tools.package_runtime_archives import package_upgrade
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'app'
            for name, data in {'JevMahjong.exe': b'exe', '_internal/shared.dll': b'runtime',
                               'models/jev-play-v2/head.pt': b'model'}.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (source / 'SHA256SUMS.json').write_text(json.dumps({
                p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file()}))
            result = package_upgrade(source, root / 'out', source,
                                     {'tag': 'baseline', 'archives': []}, '0.4.1')
            self.assertTrue(result['model_in_baseline'])
            self.assertEqual(result['reused_runtime_bytes'], len(b'runtime'))
            self.assertIn('#define ModelInBaseline "True"', (root / 'out/reuse-constants.iss').read_text())
            # Changing a managed runtime file invalidates reuse even though the old manifest still exists.
            (source / '_internal/shared.dll').write_bytes(b'broken')
            h = result['required_runtime_files']['_internal/shared.dll']
            self.assertNotEqual(hashlib.sha256((source / '_internal/shared.dll').read_bytes()).hexdigest(), h)
            with self.assertRaisesRegex(ValueError, 'file manifest'):
                package_upgrade(source, root / 'bad', source,
                                {'tag': 'baseline', 'archives': []}, '0.4.1')


if __name__ == '__main__':
    unittest.main()
