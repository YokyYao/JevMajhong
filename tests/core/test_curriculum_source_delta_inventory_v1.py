"""Keep incomplete source discovery distinct from actual source verification."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import inventory_curriculum_source_delta_v1 as subject


class DeltaInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = dict(original_manifest='never', original_declaration_path='never',
            recovered_original_sha256='never', recovered_original_path='never')

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path, hashlib.sha256(value.encode()).hexdigest()

    def graph(self, known=None):
        return subject.Inventory(known or {}, self.history, root=self.root)

    def test_known_payload_uses_completed_observation_without_read(self):
        path, digest = self.put('previous.json', '{"old": true}')
        graph = self.graph({str(path): digest})
        with patch.object(subject, 'sha', side_effect=AssertionError('old payload rescanned')):
            graph.add(str(path), digest, 'source', '/old')
            graph.expand()
        self.assertTrue(graph.nodes[str(path)]['reused_completed_source_observation'])
        self.assertFalse(graph.parsed)
        self.assertFalse(graph.hashed)

    def test_mismatched_small_file_is_an_issue_not_an_admitted_node(self):
        path, _ = self.put('new.json', '{}')
        graph = self.graph()
        graph.add(str(path), 'a' * 64, 'source', '/new')
        self.assertNotIn(str(path), graph.nodes)
        self.assertEqual(graph.issues[0]['issue'], 'declared_file_hash_mismatch')

    def test_large_new_JSON_is_explicitly_unread_not_traversed(self):
        path, digest = self.put('large.json', '{"data": [1, 2, 3]}')
        graph = self.graph()
        with patch.object(subject, 'MAX_METADATA_BYTES', 2):
            with patch.object(subject, 'sha', side_effect=AssertionError('large payload read')):
                graph.add(str(path), digest, 'source', '/large')
                graph.expand()
        row = graph.nodes[str(path)]
        self.assertTrue(row['typed_source_observation_pending'])
        self.assertFalse(row['actual_bytes_hashed_in_this_inventory'])
        self.assertFalse(graph.parsed)

    def test_relative_bundle_members_and_conflicting_declarations(self):
        path, digest = self.put('bundle/member.txt', 'actual')
        graph = self.graph()
        graph.walk(dict(directory='bundle', files={'member.txt': digest}), 'source')
        self.assertIn(str(path), graph.nodes)
        graph.add(str(path), 'a' * 64, 'other', '/member')
        self.assertEqual(graph.issues[-1]['issue'], 'conflicting_declared_identity')
        self.assertEqual(graph.nodes[str(path)]['sha256'], digest)

    def test_reachable_small_metadata_discovers_its_actual_child(self):
        child, child_sha = self.put('child.txt', 'payload')
        parent, parent_sha = self.put('root.json', json.dumps({str(child): child_sha}))
        graph = self.graph()
        graph.add(str(parent), parent_sha, str(parent), '/root')
        graph.expand()
        self.assertEqual(set(graph.nodes), {str(parent), str(child)})
        self.assertEqual(graph.parsed, {str(parent)})
        self.assertTrue(all(row['typed_source_observation_pending'] for row in graph.nodes.values()))

    def test_duplicate_JSON_keys_cannot_hide_an_input_declaration(self):
        path, _ = self.put('duplicate.json', '{"input": 1, "input": 2}')
        with self.assertRaises(ValueError):
            subject.strict_read(path)


if __name__ == '__main__':
    unittest.main()
