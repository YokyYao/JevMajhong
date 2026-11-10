import unittest
from feature_training_guard import FeatureSkillGuard


def update(value,best,stale,**kw):
    improved=best is None or value<best
    return (value,0,True) if improved else (best,stale+1,False)


class FeatureGuardTests(unittest.TestCase):
    def test_low_loss_all_pass_cannot_veto_later_skilled_policy(self):
        guard=FeatureSkillGuard(['native']);guard.step=0
        self.assertFalse(guard.assess({'native':{'exact_tp':{'none':100}}})['admissible'])
        best,stale,selected=guard.update(update,.1,None,0)
        self.assertIsNone(best);self.assertEqual(stale,0);self.assertFalse(selected)
        guard.step=50;guard.assess({'native':{'exact_tp':{'chi':1,'pon':1}}})
        self.assertEqual(guard.update(update,.4,best,stale),(.4,0,True))

    def test_feature_only_baseline_needs_real_chi_and_pon_hits(self):
        guard=FeatureSkillGuard(['native']);guard.step=0
        assessment=guard.assess({'native':{'exact_tp':{'chi':2,'pon':3}}})
        self.assertTrue(assessment['admissible']);self.assertTrue(assessment['feature_baseline'])
        self.assertEqual(guard.update(update,.3,None,0),(.3,0,True))

    def test_failed_training_checks_advance_patience(self):
        guard=FeatureSkillGuard(['native']);guard.step=50
        guard.assess({'native':{'exact_tp':{'pon':3}}})
        self.assertEqual(guard.update(update,.2,None,0),(None,1,False))


if __name__=='__main__':unittest.main()
