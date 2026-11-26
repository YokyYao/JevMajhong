import unittest

from tools.prepare_curriculum_progression_union_v3 import require_partition


class ProgressionUnionV3Tests(unittest.TestCase):
    def test_current_bound_file_preserves_heldout_legacy_origin_markers(self):
        self.assertTrue(require_partition(dict(id='legacy'), 'validation'))
        meta = dict(id='legacy', split='train')
        self.assertFalse(require_partition(meta, 'validation'))
        self.assertEqual(meta, dict(id='legacy', split='train'))
        self.assertFalse(require_partition(dict(source_logical_seed=7, split='validation'), 'validation'))
        for teacher in (dict(source_logical_seed=7), dict(source_logical_seed=7, split='train')):
            with self.assertRaises(ValueError): require_partition(teacher, 'validation')
