import random
import tempfile
from pathlib import Path
import unittest

from tools.curriculum_order import BalancedEpochRandom, balanced_order, trainer_random_proxy


class CurriculumOrderTests(unittest.TestCase):
    def test_real_counts_preserved_and_each_update_has_both_rare_labels(self):
        categories = ['plain']*8826+['riichi_yes']*194+['riichi_no']*272
        order = balanced_order(categories, random.Random(19), 64)
        self.assertEqual(sorted(order), list(range(9292)))
        for start in range(0, len(order), 64):
            group = {categories[i] for i in order[start:start+64]}
            self.assertTrue({'riichi_yes', 'riichi_no'} <= group)

    def test_reproducible_epoch_sequence_and_no_repeated_rows(self):
        categories = ['plain']*90+['riichi_yes']*10+['riichi_no']*10
        a = BalancedEpochRandom(7, categories, 32)
        b = BalancedEpochRandom(7, categories, 32)
        first = None
        for _ in range(3):
            x = list(range(len(categories)))
            y = list(reversed(x))
            a.shuffle(x)
            b.shuffle(y)
            self.assertEqual(x, y)
            self.assertEqual(len(set(x)), len(categories))
            if first is None:
                first = x
            else:
                self.assertNotEqual(first, x)

    def test_insufficient_coverage_rejected_instead_of_duplicated(self):
        with self.assertRaises(ValueError):
            balanced_order(['plain']*50+['riichi_yes']+['riichi_no'], random.Random(1), 16)

    def test_pinned_epoch_rng_only_and_auditable_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            categories = ['plain']*20+['riichi_yes']*4+['riichi_no']*4
            proxy, created = trainer_random_proxy(random, 71, categories, 16, temp)
            normal = proxy.Random('0:augmentation-item')
            self.assertIs(type(normal), random.Random)
            rng = proxy.Random(71)
            order = list(range(28))
            rng.shuffle(order)
            self.assertEqual(len(created), 1)
            self.assertTrue((Path(temp)/'epoch-01.json').is_file())
            with self.assertRaises(ValueError):
                proxy.Random(71)
