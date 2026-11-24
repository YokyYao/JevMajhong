import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.curriculum_confirmation_json_sources_v1 import scan_json_artifacts


class JsonSourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def declaration(self, path, pointer='/protected_source_ranges/0/start'):
        return {str(path): dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), start_pointers=[pointer])}

    def test_nested_values_keys_and_escaped_strings_are_checked(self):
        path = self.root / 'data.json'
        path.write_text('{"rows":[{"seed":2026300001},{"id":"run-2026300031"}],'
                        '"2026300005":0,"escaped":"\\u0032\\u0030\\u0032\\u0036\\u0033\\u0030\\u0030\\u0030\\u0030\\u0030"}')
        result = scan_json_artifacts([path])
        self.assertEqual(result['unreviewed_reserved_occurrences'], 4)
        self.assertEqual(result['reserved_native_resets_or_games'], 0)
        self.assertFalse(result['confirmation_admitted'])
        self.assertTrue(result['checked_files_only_not_complete_source_graph'])
        self.assertTrue(result['source_family_exclusion_not_proved_by_this_function'])

    def test_exact_declaration_does_not_exempt_other_fields(self):
        path = self.write('config.json', dict(protected_source_ranges=[dict(start=2026300000, count=32)],
            generated=[dict(seed=2026300007)]))
        result = scan_json_artifacts([path], self.declaration(path))
        self.assertEqual(result['reviewed_declaration_occurrences'], 1)
        self.assertEqual(result['unreviewed_reserved_occurrences'], 1)
        self.assertEqual(result['files'][0]['classification'], 'unreviewed_reserved_occurrences')

    def test_wrapped_config_declaration_and_whole_seed_boundaries(self):
        path = self.write('registration.json', dict(config=dict(protected_source_ranges=[dict(start=2026300000, count=32)]),
            not_reserved=[2026299999, 2026300032, '120263000007', True, None]))
        result = scan_json_artifacts([path, path], self.declaration(path, '/config/protected_source_ranges/0/start'))
        self.assertEqual(result['total_files'], 1)
        self.assertEqual(result['reviewed_declaration_occurrences'], 1)
        self.assertEqual(result['unreviewed_reserved_occurrences'], 0)
        self.assertTrue(result['all_inventoried_files_stable'])

    def test_declaration_hash_change_is_rejected(self):
        path = self.write('config.json', dict(protected_source_ranges=[dict(start=2026300000, count=32)]))
        declarations = self.declaration(path)
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            scan_json_artifacts([path], declarations)

    def test_scores_cannot_be_declared_as_protected_ranges(self):
        path = self.write('data.json', dict(scores=[2026300000]))
        with self.assertRaisesRegex(ValueError, 'Only explicit'):
            scan_json_artifacts([path], self.declaration(path, '/scores/0'))

    def test_wrong_range_or_extra_fields_cannot_be_declared(self):
        for value in (dict(start=2026300000, count=31), dict(start=2026300000, count=32, seed=2026300001)):
            path = self.write('config.json', dict(protected_source_ranges=[value]))
            with self.assertRaisesRegex(ValueError, 'Exact protected'):
                scan_json_artifacts([path], self.declaration(path))

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        path = self.root / 'invalid.json'
        for text in ('{"seed":2026300000,"seed":0}', '{"x":NaN}', '{"x":1e999}'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                scan_json_artifacts([path])

    def test_pointer_escaping_and_undeclared_seed_like_floats(self):
        path = self.write('data.json', {'a/b~c': [2026300000.5]})
        result = scan_json_artifacts([path])
        hits = result['files'][0]['first_unreviewed_private']
        self.assertEqual(hits, [dict(pointer='/a~1b~0c/0', kind='float', seed=2026300000)])

    def test_missing_inventory_and_unused_or_duplicate_declarations_rejected(self):
        path = self.write('config.json', dict(protected_source_ranges=[dict(start=2026300000, count=32)]))
        other = self.write('other.json', {})
        with self.assertRaisesRegex(ValueError, 'actual inventory'):
            scan_json_artifacts([other], self.declaration(path))
        declarations = self.declaration(path)
        declarations[str(path)]['start_pointers'] *= 2
        with self.assertRaisesRegex(ValueError, 'unique'):
            scan_json_artifacts([path], declarations)
        with self.assertRaises(ValueError):
            scan_json_artifacts([])


if __name__ == '__main__':
    unittest.main()
