import unittest
from jevmahjong.endgame.reference import escape_supported_keys, greedy_win_target, restrict_distribution, improvement_target
from jevmahjong.endgame.fit import terminal_floor_guard


class EscapeConstraintTests(unittest.TestCase):
    def test_profitable_but_riskier_continue_does_not_replace_a_safe_win(self):
        p={'win':1.,'risk':0.,'safe':0.}
        escape={'win':.75,'risk':.5,'safe':.75}
        values={'win':0.,'risk':.5,'safe':.1}
        eligible=escape_supported_keys(p,escape,list(p))
        self.assertEqual(eligible,['win','safe'])
        q=greedy_win_target({k:{'type':'hora' if k=='win' else 'none'} for k in p},values,eligible)
        self.assertEqual(max(q,key=q.get),'safe')
        self.assertEqual(q['risk'],0.)

    def test_ordinary_action_constraint_uses_actual_greedy_baseline_not_soft_mean(self):
        p={'reach':.6,'discard':.4,'safe':0.}
        escape={'reach':.75,'discard':.5,'safe':1.}
        eligible=escape_supported_keys(p,escape,list(p))
        self.assertEqual(eligible,['reach','safe'])
        supported=restrict_distribution(p,eligible)
        values={'reach':.1,'safe':.2}
        q=improvement_target({k:supported[k] for k in eligible},values,.1,.1)
        self.assertGreaterEqual(escape[max(q,key=q.get)],escape['reach'])
        # Unsupported simulated actions cannot expand the serving menu.
        self.assertEqual(escape_supported_keys(p,escape,['reach']),['reach'])

    def test_invalid_escape_estimates_are_rejected(self):
        for value in (float('nan'),1.01,-.01):
            with self.assertRaises(ValueError):
                escape_supported_keys({'win':1.},{'win':value},['win'])

    def test_new_input_cannot_lower_original_policy_floor(self):
        passed,audit=terminal_floor_guard(0,[.4],[.8],None)
        self.assertTrue(passed)  # Valid no-update anchor, never a trained success.
        self.assertEqual(audit['required_nonzero_floor'],.8)
        self.assertFalse(terminal_floor_guard(8,[.6],[.8],.4)[0])
        self.assertTrue(terminal_floor_guard(48,[.8],[.8],.4)[0])
        self.assertFalse(terminal_floor_guard(48,[.8],[.6],.9)[0])
        with self.assertRaises(ValueError):
            terminal_floor_guard(8,[.8],[.8],None)


if __name__=='__main__':
    unittest.main()
