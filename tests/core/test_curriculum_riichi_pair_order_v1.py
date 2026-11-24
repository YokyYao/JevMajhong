from collections import Counter
import random
import unittest

from tools.curriculum_riichi_pair_order_v1 import PREFIXES, pair_within_blocks


class RiichiPairOrderTests(unittest.TestCase):
    def categories(self, blocks=3):
        one = [source + ':riichi_' + label for source in PREFIXES for label in ('yes', 'no')]
        one += ['ordinary'] * 10
        return one * blocks

    def test_pairs_maximize_joint_classes_without_replicas_or_cross_block_moves(self):
        categories = self.categories()
        order = list(range(len(categories)))
        original = list(order)
        paired, receipt = pair_within_blocks(order, categories, random.Random(41), block_size=14, accumulation=7)
        self.assertEqual(order, original)
        self.assertEqual(sorted(paired), original)
        for start in range(0, len(order), 14):
            self.assertEqual(set(paired[start:start+14]), set(order[start:start+14]))
        for source in PREFIXES:
            self.assertEqual(receipt['after'][source]['updates_with_both_classes'], 3)
            self.assertEqual(receipt['after'][source]['positive_records'], 3)
            self.assertEqual(receipt['after'][source]['negative_records'], 3)
        self.assertTrue(receipt['each_original_once'])
        self.assertTrue(receipt['empty_source_updates_can_increase'])
        self.assertFalse(receipt['training_started'])
        self.assertTrue(receipt['model_improvement_not_established'])

    def test_same_seed_and_same_original_order_reproduce(self):
        categories = self.categories()
        order = list(range(len(categories)))
        a = pair_within_blocks(order, categories, random.Random(69), 14, 7)
        b = pair_within_blocks(order, categories, random.Random(69), 14, 7)
        self.assertEqual(a, b)

    def test_unequal_classes_preserve_counts_and_maximum_pair_bound(self):
        categories = self.categories(1) + [PREFIXES[0] + ':riichi_no'] * 4
        order = list(range(len(categories)))
        paired, receipt = pair_within_blocks(order, categories, random.Random(72), 21, 7)
        self.assertEqual(Counter(categories[index] for index in paired), Counter(categories))
        self.assertEqual(receipt['after'][PREFIXES[0]]['updates_with_both_classes'], 1)
        self.assertEqual(receipt['after'][PREFIXES[0]]['negative_records'], 5)
        self.assertEqual([len(paired[i:i+7]) for i in range(0, len(paired), 7)], [7, 7, 4])

    def test_actual_sized_68_record_tail_and_all_19_source_sets_preserved(self):
        names = [source + ':riichi_' + kind for source in PREFIXES for kind in ('yes', 'no')]
        names += ['other:' + str(index) for index in range(15)]
        categories = [names[i % 19] for i in range(511 * 2 + 214)]
        order = list(range(len(categories)))
        paired, receipt = pair_within_blocks(order, categories, random.Random(99))
        self.assertEqual(len(paired[-68:]), 68)
        self.assertEqual(len(paired) % 73, 68)
        for start in range(0, len(order), 511):
            self.assertEqual(Counter(categories[index] for index in paired[start:start+511]),
                             Counter(categories[index] for index in order[start:start+511]))
        self.assertEqual(receipt['copies_added'], 0)

    def test_missing_duplicate_indices_and_unaligned_sizes_rejected(self):
        categories = self.categories(1)
        order = list(range(len(categories)))
        for invalid in (order[:-1], order[:-1] + [0], [True] + order[1:]):
            with self.assertRaises(ValueError):
                pair_within_blocks(invalid, categories, random.Random(1), 14, 7)
        with self.assertRaises(ValueError):
            pair_within_blocks(order, categories, random.Random(1), 15, 7)
        with self.assertRaises(ValueError):
            pair_within_blocks(order, ['ordinary'] * len(categories), random.Random(1), 14, 7)


if __name__ == '__main__':
    unittest.main()
