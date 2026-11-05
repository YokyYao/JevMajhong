import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.package_desktop_archives import package


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


if __name__ == '__main__':
    unittest.main()
