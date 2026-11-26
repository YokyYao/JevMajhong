"""Context variants preserve physical tiles and original family splits."""
from collections import defaultdict
import unittest
from tools.generate_curriculum_daiminkan_context_v1 import plans, physical_wall, CONTEXTS, DISCARDERS
from tools.generate_curriculum_kan_sources_v1 import plans as original_plans


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plans, cls.collision = plans()

    def test_original_family_partition_is_preserved_across_all_variants(self):
        original = {p['group_id']: p['split'] for p in original_plans(dict(seed_start=2026680000))}
        families = defaultdict(list)
        for p in self.plans:
            self.assertEqual(p['split'], original[p['group_id']])
            families[p['motif'], p['template']].append(p)
        self.assertEqual(len(self.plans), 576)
        self.assertEqual(len(families), 32)
        for rows in families.values():
            self.assertEqual(len(rows), 18)
            self.assertEqual(len({p['group_id'] for p in rows}), 1)
            self.assertEqual({p['context'] for p in rows}, set(CONTEXTS))
            self.assertEqual({p['discarder'] for p in rows}, set(DISCARDERS))

    def test_wall_is_a_repeatable_full_physical_permutation(self):
        for p in self.plans:
            with self.subTest(id=p['id']):
                wall = physical_wall(p)
                self.assertEqual(sorted(wall), list(range(136)))
                self.assertEqual(wall, physical_wall(p))
                self.assertEqual(len(p['initial13']), 13)
                self.assertEqual(p['initial13'].count(p['tile_type']), 3)
                self.assertFalse(2026300000 <= p['seed'] < 2026300032)

    def test_all_plans_are_unique_before_teacher_values(self):
        self.assertEqual(len({p['id'] for p in self.plans}), 576)
        self.assertEqual(len({p['seed'] for p in self.plans}), 576)
        self.assertEqual(self.plans, plans()[0])


if __name__ == '__main__': unittest.main()
