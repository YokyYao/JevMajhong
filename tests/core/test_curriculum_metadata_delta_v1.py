import json
from pathlib import Path
import tempfile
import unittest

from tools.observe_curriculum_metadata_delta_v1 import decode, findings, observe_file
from tools.curriculum_transfer import sha


class MetadataDeltaBoundaries(unittest.TestCase):
    def test_exponent_encoded_logical_seed_is_decoded_before_check(self):
        _, hits = findings(decode('{"seed":2.026300001e9}'))
        self.assertEqual(hits[0]['encoding'], 'logical')
        self.assertEqual(hits[0]['value'], 2026300001)

    def test_native_hand_seed_in_dictionary_key_is_not_hidden(self):
        seed = 2026300031 * (1 << 32) + 99999
        _, hits = findings({str(seed): None})
        self.assertEqual(hits[0]['encoding'], 'native_spaced_hand')
        self.assertTrue(hits[0]['pointer'].endswith('/@key'))

    def test_protected_range_does_not_exempt_a_separate_data_occurrence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'metadata.json'
            path.write_text(json.dumps(dict(
                protected_source_ranges=[dict(start=2026300000, count=32)],
                source_seed=2026300000)))
            row = observe_file(path, sha(path))
            self.assertEqual(len(row['reviewed_exact_protected_range_declarations']), 1)
            self.assertEqual(len(row['unreviewed_reserved_occurrences']), 1)
            self.assertEqual(row['unreviewed_reserved_occurrences'][0]['pointer'], '/source_seed')

    def test_duplicate_keys_and_changed_input_are_rejected(self):
        with self.assertRaises(ValueError):
            decode('{"seed":1,"seed":2}')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'metadata.json'; path.write_text('{"step":2500}')
            expected = sha(path); path.write_text('{"step":2501}')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                observe_file(path, expected)


if __name__ == '__main__':
    unittest.main()
