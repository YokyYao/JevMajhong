import math
import unittest
from tools.curriculum_ri_every_update_replay_v1 import build,MASS,RI


class ReplayTests(unittest.TestCase):
    def fixture(self):
        categories=[]
        for source in MASS:
            n=RI.index(source)+1 if source in RI else 2
            categories.extend([source]*n)
        return categories,list(range(len(categories)))

    def test_empirical_loss_and_gradient_preserved_with_unequal_repeats(self):
        categories,baseline=self.fixture();p=build(categories,baseline,7,batch=9)
        theta=1.7
        values=[(theta-(i%7))**2 for i in baseline]
        gradients=[2*(theta-(i%7)) for i in baseline]
        for function in (values,gradients):
            original=sum(MASS[s]*sum(function[i] for i in baseline if categories[i]==s)/categories.count(s) for s in MASS)
            actual=sum(float(p['weights_by_original_index'][str(i)])*function[i] for i in p['indices'])/len(p['indices'])
            self.assertAlmostEqual(original,actual,places=12)

    def test_each_update_has_all_four_RI_sources(self):
        categories,baseline=self.fixture();p=build(categories,baseline,7,batch=9)
        for start in range(0,len(p['indices']),9):
            group=p['indices'][start:start+9]
            self.assertEqual(len(group),9)
            for source in RI:self.assertEqual(sum(categories[i]==source for i in group),1)
        self.assertEqual(set(p['indices']),set(baseline))
        self.assertEqual(p['nonRI_original_subsequence'],[i for i in baseline if categories[i] not in RI])

    def test_deterministic_and_no_exposure_loss(self):
        categories,baseline=self.fixture()
        self.assertEqual(build(categories,baseline,7,batch=9),build(categories,baseline,7,batch=9))
        p=build(categories,baseline,8,batch=9)
        self.assertLessEqual(p['summary']['max_repeat'],8)
        self.assertTrue(p['summary']['per_original_empirical_objective_preserved'])
        self.assertTrue(all(math.isfinite(w) and w>0 for w in p['weights_by_original_index'].values()))

    def test_invalid_membership_or_caps_are_rejected(self):
        categories,baseline=self.fixture()
        with self.assertRaises(ValueError):build(categories,baseline[:-1],7,batch=9)
        with self.assertRaises(ValueError):build(categories,baseline,7,batch=9,max_repeat=1)
        with self.assertRaises(ValueError):build(categories,baseline,7,batch=9,max_weight=.001)


if __name__=='__main__':unittest.main()
