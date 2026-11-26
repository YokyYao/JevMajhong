import unittest
from tools.prepare_curriculum_progression_union_v2 import require_partition


class ProgressionUnionV2Tests(unittest.TestCase):
    def test_original_file_split_handles_missing_legacy_field_but_not_teacher_omission(self):
        self.assertTrue(require_partition(dict(id='legacy'), 'validation'))
        self.assertFalse(require_partition(dict(id='teacher', source_logical_seed=7, split='train'), 'train'))
        for meta in (dict(source_logical_seed=7), dict(split='train'), dict(split=None)):
            with self.assertRaises(ValueError): require_partition(meta, 'validation')
