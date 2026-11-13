"""Win-priority target keeps costly calls and negative counterexamples visible."""
import copy
import unittest
import torch
from jevmahjong.training.call_head import complete_groups, distillation_target, distillation_loss
from jevmahjong.evaluation.call_head_audit import manual_distillation_target
from tests.core.test_call_head import row, Head


class CallWinTargetTests(unittest.TestCase):
    def group(self,win,speed,utility=.1):
        rows=[row(1,utility,100.),row(2,-.2,-100.)]
        for r,w,s in zip(rows,[win,.1],[speed,.05]):
            r['training_credit']['control'].update(own_win=.5,fast_reward=.3)
            r['training_credit']['branch'].update(own_win=.5+w,fast_reward=.3+s)
        g=complete_groups(rows)[0];g['records']=rows
        return g

    def test_removes_positive_boost_for_fewer_wins_or_equal_wins_but_slower(self):
        for win,speed in [(-.1,.1),(0.,-.1),(0.,0.)]:
            with self.subTest(win=win,speed=speed):
                g=self.group(win,speed);before=copy.deepcopy(g)
                new=distillation_target(g,.05,'win_compatible_v1')
                old=distillation_target(g,.05)
                self.assertLess(float(new[1]),float(old[1]))
                # Neutralized positive option keeps its original prior odds.
                self.assertAlmostEqual(float(new[1]/new[0]),float(g['prior'][1]/g['prior'][0]))
                # Negative option penalty remains exactly the original -0.2.
                self.assertAlmostEqual(float(torch.log(new[2]/new[0])-torch.log(g['prior'][2]/g['prior'][0])),-4.)
                self.assertTrue(torch.equal(g['utility'],before['utility']))
                self.assertEqual(g['credit'],before['credit']);self.assertEqual(len(new),3)
                self.assertTrue(torch.allclose(new,manual_distillation_target(g,.05,'win_compatible_v1'),atol=1e-15,rtol=0))

    def test_retains_positive_boost_for_more_wins_or_equal_wins_and_faster(self):
        for win,speed in [(.1,-.1),(0.,.1)]:
            g=self.group(win,speed)
            new=distillation_target(g,.05,'win_compatible_v1')
            self.assertTrue(torch.equal(new,distillation_target(g,.05)))
            self.assertTrue(torch.allclose(new,manual_distillation_target(g,.05,'win_compatible_v1')))

    def test_negative_risky_action_is_still_penalized_even_when_it_wins_more(self):
        g=self.group(.1,.1,-.1)
        self.assertTrue(torch.equal(distillation_target(g,.05),distillation_target(g,.05,'win_compatible_v1')))

    def test_gradient_reduces_conflicting_boost_and_still_teaches_supported_call(self):
        for useful in (False,True):
            g=self.group(.1 if useful else -.1,.05);head=Head()
            distillation_loss(head,g,.05,'win_compatible_v1').backward()
            if useful:self.assertLess(float(head.offset.grad[1]),0.)
            else:
                legacy=Head();distillation_loss(legacy,g,.05).backward()
                self.assertGreater(float(head.offset.grad[1]),float(legacy.offset.grad[1]))
                # Normalization after penalizing a third option can still raise
                # this option's mass: the rule withholds a boost, not a ban.

    def test_legacy_target_is_unchanged_and_unknown_modes_are_rejected(self):
        g=self.group(-.1,.05)
        expected=(g['prior'].log()+g['utility']/.05).softmax(-1)
        self.assertTrue(torch.equal(expected,distillation_target(g,.05)))
        for mode in ['typo',None]:
            with self.assertRaises(ValueError):distillation_target(g,.05,mode)
        g['credit'][1]['own_win']=float('nan')
        with self.assertRaises(ValueError):distillation_target(g,.05,'win_compatible_v1')


if __name__=='__main__':unittest.main()
