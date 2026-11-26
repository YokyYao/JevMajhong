from collections import Counter
import random
import unittest
from tools import curriculum_ri_coverage_replay_v2 as replay
from tools.curriculum_ri_every_update_replay_v1 import build as old_build


class CoverageReplayTests(unittest.TestCase):
    def fixture(self):
        counts={s:4 for s in replay.MASS}
        for s in replay.RI:
            counts[s]=25 if ':teacher:' in s and s.endswith(':riichi_yes') else 20 if ':teacher:' in s else 2 if s.endswith(':riichi_yes') else 3
        categories=[s for s,n in counts.items() for _ in range(n)]
        baseline=list(range(len(categories)));random.Random(123).shuffle(baseline)
        return categories,baseline,counts

    def test_larger_RI_pool_is_not_truncated_to_one_per_update(self):
        categories,baseline,counts=self.fixture()
        with self.assertRaises(ValueError):old_build(categories,baseline,7,batch=11)
        result=replay.build(categories,baseline,7,batch=11)
        self.assertEqual(set(result['indices']),set(baseline))
        self.assertLess(result['summary']['optimizer_updates'],max(counts[s] for s in replay.RI))
        for source in replay.RI:
            self.assertGreaterEqual(result['summary']['RI_per_update_range'][source][0],1)
        self.assertTrue(any(v[1]>1 for v in result['summary']['RI_per_update_range'].values()))

    def test_all_original_loss_masses_exact_and_repeats_bounded(self):
        categories,baseline,counts=self.fixture();result=replay.build(categories,baseline,7,batch=11)
        repeats=Counter(result['indices']);size=len(result['indices'])
        for i,source in enumerate(categories):
            actual=result['weights_by_original_index'][str(i)]*repeats[i]/size
            self.assertAlmostEqual(actual,replay.MASS[source]/counts[source],places=12)
        self.assertLessEqual(max(repeats.values()),8)
        self.assertEqual(result['nonRI_original_subsequence'],[i for i in baseline if categories[i] not in replay.RI])

    def test_deterministic_and_complete_physical_batches(self):
        categories,baseline,_=self.fixture();a=replay.build(categories,baseline,7,batch=11)
        self.assertEqual(a,replay.build(categories,baseline,7,batch=11))
        self.assertNotEqual(a['indices'],replay.build(categories,baseline,8,batch=11)['indices'])
        self.assertEqual(len(a['indices'])%11,0)
        for start in range(0,len(a['indices']),11):
            self.assertTrue(set(replay.RI)<=set(categories[i] for i in a['indices'][start:start+11]))

    def test_invalid_coverage_and_repeat_cap_rejected(self):
        categories,baseline,_=self.fixture()
        with self.assertRaises(ValueError):replay.build(categories,baseline[:-1],7,batch=11)
        with self.assertRaises(ValueError):replay.build(categories,baseline,7,batch=11,max_repeat=1)
        with self.assertRaises(ValueError):replay.build(categories,baseline,7,batch=4)


if __name__=='__main__':unittest.main()
