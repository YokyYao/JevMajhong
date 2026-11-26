from collections import Counter
import random
import unittest
from tools.build_curriculum_calls_repair import MASS,layout
from tools.curriculum_course_order import course_order


class CallsRepairDataTests(unittest.TestCase):
    def test_large_child_keeps_rare_basic_sources_without_copies(self):
        counts={key:2000 for key in MASS};counts.update({'basic:legacy:plain':8826,'basic:legacy:riichi_yes':194,
            'basic:legacy:riichi_no':272,'basic:teacher:plain':3106,'basic:teacher:riichi_yes':1075,'basic:teacher:riichi_no':812})
        proposed=layout(counts);categories=[key for key,n in counts.items() for _ in range(n)]
        order=course_order(categories,random.Random(2026112209),proposed['accumulation'])
        self.assertEqual(Counter(categories[i] for i in order),Counter(counts))
        for start in range(0,len(order),proposed['accumulation']):self.assertEqual({categories[i] for i in order[start:start+proposed['accumulation']]},set(MASS))
        self.assertAlmostEqual(sum(v for k,v in MASS.items() if k.startswith('basic:')),.65)
        for key in MASS:self.assertAlmostEqual(counts[key]*proposed['example_weights'][key]/len(categories),MASS[key])

    def test_missing_rare_class_cannot_be_silently_dropped(self):
        counts=dict.fromkeys(MASS,100);del counts['basic:legacy:riichi_yes']
        with self.assertRaises(ValueError):layout(counts)


if __name__=='__main__':unittest.main()
