import json
from pathlib import Path
import tempfile
import unittest

from tools.curriculum_historical_code_alias_v1 import METHOD, digest, resolve


class HistoricalCodeAliasTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.original = self.root / 'jevmahjong/data/example.py'
        self.original.parent.mkdir(parents=True)
        self.original.write_text('new_source = True\n')
        self.archive = self.root / 'runs/code' / ('a' * 64) / 'jevmahjong/data/example.py'
        self.archive.parent.mkdir(parents=True)
        self.archive.write_text('old_source = True\n')
        self.expected = digest(self.archive)
        self.manifest = self.root / 'inputs-private.json'
        self.manifest.write_text(json.dumps({str(self.original): self.expected}))
        self.manifest_sha = digest(self.manifest)
        self.declaration = dict(method=METHOD, original_path=str(self.original),
            expected_sha256=self.expected, original_manifest=str(self.manifest),
            manifest_sha256=self.manifest_sha, archived_path=str(self.archive))
        self.bindings = {str(self.archive): self.expected}

    def apply(self):
        return resolve(self.original, self.expected, self.declaration, root=self.root,
            original_manifest=self.manifest, manifest_sha256=self.manifest_sha,
            protocol_code_bindings=self.bindings)

    def test_explicit_historic_identity_preserves_current_source_and_manifest(self):
        before = (self.original.read_bytes(), self.manifest.read_bytes())
        receipt = self.apply()
        self.assertEqual(receipt['archived_sha256'], self.expected)
        self.assertNotEqual(receipt['observed_current_sha256'], self.expected)
        self.assertFalse(receipt['final_confirmation_admitted'])
        self.assertEqual(before, (self.original.read_bytes(), self.manifest.read_bytes()))

    def test_data_alias_is_rejected_even_when_hash_and_protocol_membership_match(self):
        data = self.original.with_suffix('.jsonl')
        data.write_text('{}\n')
        self.original = data
        self.declaration['original_path'] = str(data)
        with self.assertRaises(ValueError):
            self.apply()

    def test_unbound_archive_is_rejected(self):
        self.bindings = {}
        with self.assertRaisesRegex(ValueError, 'verified actual ancestor'):
            self.apply()

    def test_changed_archive_and_changed_original_manifest_are_rejected(self):
        self.archive.write_text('unrelated = True\n')
        with self.assertRaisesRegex(ValueError, 'Archived source bytes'):
            self.apply()
        self.manifest.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unchanged original manifest'):
            self.apply()

    def test_different_module_path_and_symlink_are_rejected(self):
        other = self.archive.with_name('other.py')
        other.write_bytes(self.archive.read_bytes())
        self.bindings[str(other)] = self.expected
        self.declaration['archived_path'] = str(other)
        with self.assertRaisesRegex(ValueError, 'identical package-relative'):
            self.apply()
        link = self.archive.with_name('alias.py')
        link.symlink_to(self.archive)
        self.declaration['archived_path'] = str(link)
        with self.assertRaisesRegex(ValueError, 'without symlink'):
            self.apply()


if __name__ == '__main__':
    unittest.main()
