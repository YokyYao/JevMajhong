import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_feature_parent


class ParentComparisonTests(unittest.TestCase):
    def test_reuses_candidate_games_and_replaces_originalB_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'development').mkdir()
            (root / 'parent').mkdir()
            oldB = {'policy': 'reference', 'player': 'oldB', 'seed': 1}
            candidate = {'policy': 'candidate', 'player': 'updated', 'seed': 1}
            parent = {'policy': 'reference', 'player': 'feature200', 'seed': 1}
            (root / 'development/games-private.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in (oldB, candidate)) + '\n')
            (root / 'parent/games-private.jsonl').write_text(json.dumps(parent) + '\n')
            with patch.object(evaluate_feature_parent, 'ROOT', root):
                rows = evaluate_feature_parent.combined_rows(
                    {'out': 'development', 'paired_parent_out': 'parent'})
            self.assertEqual(rows, [candidate, parent])
            self.assertNotIn(oldB, rows)

    def test_foreign_policy_in_parent_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'development').mkdir()
            (root / 'parent').mkdir()
            (root / 'development/games-private.jsonl').write_text('')
            (root / 'parent/games-private.jsonl').write_text('{"policy":"candidate"}\n')
            with patch.object(evaluate_feature_parent, 'ROOT', root), self.assertRaises(ValueError):
                evaluate_feature_parent.combined_rows({'out': 'development', 'paired_parent_out': 'parent'})


if __name__ == '__main__':
    unittest.main()
