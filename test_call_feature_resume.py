import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from call_policy import digest
from scripts.cache_call_features import verified_chunk
from scripts.watch_call_repair import observation_complete


class FeatureResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'train-0000.pt'
        self.expected = [{'id': 'fixture', 'group': 'fixture-match', 'view': 'native',
                          'label': 0, 'types': ['chi', 'none'], 'text_sha256': 'fixture-request'}]
        torch.save([{**self.expected[0], 'decide': torch.ones(3), 'options': torch.ones(2, 3)}], self.path)
        self.path.with_suffix('.audit.json').write_text(json.dumps({'protocol_sha256': 'protocol',
            'checks': 1, 'tensor_sha256': digest(self.path)}))

    def test_verified_completed_shard_is_reusable(self):
        rows = verified_chunk(self.path, self.expected, 'protocol', 3)
        self.assertEqual(rows[0]['id'], 'fixture')
        torch.testing.assert_close(rows[0]['decide'], torch.ones(3))

    def test_changed_label_or_protocol_is_not_silently_reused(self):
        changed = copy.deepcopy(self.expected); changed[0]['label'] = 1
        with self.assertRaises(ValueError): verified_chunk(self.path, changed, 'protocol', 3)
        with self.assertRaises(ValueError): verified_chunk(self.path, self.expected, 'other-protocol', 3)

    def test_modified_tensor_or_dimension_is_rejected(self):
        with self.assertRaises(ValueError): verified_chunk(self.path, self.expected, 'protocol', 4)
        with self.path.open('ab') as stream: stream.write(b'changed')
        with self.assertRaises(ValueError): verified_chunk(self.path, self.expected, 'protocol', 3)

    def test_orphan_without_probability_audit_requires_reencoding(self):
        self.path.with_suffix('.audit.json').unlink()
        self.assertIsNone(verified_chunk(self.path, self.expected, 'protocol', 3))

    def test_feature_observer_requires_complete_cache_marker(self):
        cfg = {'evaluation_stage': 'feature_cache'}
        self.assertFalse(observation_complete(cfg, {'training': {'status': 'encoding_natural_call_features'}}))
        self.assertTrue(observation_complete(cfg, {'training': {'status': 'feature_cache_complete'}}))


if __name__ == '__main__': unittest.main()
