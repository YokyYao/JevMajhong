from collections import Counter
from pathlib import Path
import random
import tempfile
import unittest

from tools.curriculum_course_order import course_order,trainer_course_random_proxy


class CourseOrderTests(unittest.TestCase):
    def categories(self):
        counts=dict(plain=8826,open=5590,call=3569,kan=1786,riichi_yes=194,riichi_no=272)
        return [key for key,count in counts.items() for _ in range(count)]

    def test_stage2_real_counts_every_row_once_all_strata_and_large_tail(self):
        categories=self.categories();order=course_order(categories,random.Random(13),125)
        self.assertEqual(sorted(order),list(range(20237)))
        batches=[order[i:i+125] for i in range(0,len(order),125)]
        self.assertEqual(len(batches),162);self.assertEqual(len(batches[-1]),112)
        self.assertTrue(all({categories[i] for i in batch}==set(categories) for batch in batches))
        self.assertEqual(Counter(categories[i] for i in order),Counter(categories))
        self.assertEqual(order,course_order(categories,random.Random(13),125))
        self.assertNotEqual(order,course_order(categories,random.Random(14),125))

    def test_reject_tiny_tail_or_insufficient_distinct_stratum(self):
        with self.assertRaises(ValueError):course_order(self.categories(),random.Random(13),128)
        with self.assertRaises(ValueError):course_order(['plain']*20+['rare'],random.Random(13),7)

    def test_only_epoch_rng_is_replaced_with_new_receipts_per_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            categories=self.categories();proxy,created=trainer_course_random_proxy(random,13,categories,125,Path(temp))
            self.assertIs(type(proxy.Random(71)),random.Random)
            rng=proxy.Random(13);order=list(range(len(categories)));rng.shuffle(order);first=list(order);rng.shuffle(order)
            self.assertEqual(rng.epochs,2);self.assertNotEqual(order,first)
            self.assertTrue((Path(temp)/'epoch-02.json').exists())
            self.assertEqual(created,[rng])
            with self.assertRaises(ValueError):proxy.Random(13)
