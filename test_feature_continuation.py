import json
from pathlib import Path
import tempfile
import unittest

from call_policy import digest
from feature_continuation import remaining_indices, UpdatedFeatureSkillGuard, verify_preparation


class ExposureTests(unittest.TestCase):
    def recover(self, order, **kwargs):
        recipe = dict(epoch=0, next_microbatch=2, records_seen=2,
                      batch=1, world_size=1, p_none_pair=0)
        recipe.update(kwargs)
        return remaining_indices(order, 5, **recipe)

    def test_saved_shuffled_prefix_is_excluded(self):
        order = [4, 1, 3, 0, 2]
        tail = self.recover(order)
        self.assertEqual(tail, [3, 0, 2])
        self.assertFalse(set(tail) & set(order[:2]))
        self.assertEqual(set(tail) | set(order[:2]), set(range(5)))

    def test_changed_order_or_cursor_cannot_claim_unseen(self):
        for order, changes in [([4, 1, 1, 0, 2], {}),
                               ([4, 1, 3, 0, 2], {'records_seen': 3}),
                               ([4, 1, 3, 0, 2], {'next_microbatch': -1})]:
            with self.subTest(order=order, changes=changes), self.assertRaises(ValueError):
                self.recover(order, **changes)

    def test_unsupported_exposure_recipes_fail_closed(self):
        for changes in ({'epoch': 1}, {'batch': 2}, {'world_size': 2}, {'p_none_pair': 0.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.recover([4, 1, 3, 0, 2], **changes)

    def test_changed_frozen_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            for name in ('train', 'validation'):
                (root / 'data' / (name + '.jsonl')).write_text('{}\n')
            source = root / 'source.py'
            source.write_text('registered\n')
            config = {'data': 'data'}
            preparation = {'complete': True, 'config': config, 'model_test_opened': False,
                'frozen_files': {'source.py': digest(source)},
                'files': {s: digest(root / 'data' / (s + '.jsonl')) for s in ('train', 'validation')}}
            verify_preparation(root, config, preparation)
            source.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'source.py'):
                verify_preparation(root, config, preparation)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def update(value, best, stale, **kwargs):
        return (value, 0, True) if best is None or value < best else (best, stale + 1, False)

    def test_unchanged_parent_is_diagnostic_and_cannot_veto_new_score_trial(self):
        guard = UpdatedFeatureSkillGuard(['native'])
        guard.assess({'native': {'exact_tp': {'chi': 23, 'pon': 47}}})
        guard.step = 0
        best, stale, selected = guard.update(self.update, .6, None, 0)
        self.assertEqual((best, stale, selected), (None, 0, False))
        guard.step = 50
        self.assertEqual(guard.update(self.update, .7, best, stale), (.7, 0, True))

    def test_updated_all_pass_policy_is_not_selected(self):
        guard = UpdatedFeatureSkillGuard(['native'])
        guard.step = 50
        guard.assess({'native': {'exact_tp': {'none': 836}}})
        self.assertEqual(guard.update(self.update, .1, None, 0), (None, 1, False))


if __name__ == '__main__':
    unittest.main()
