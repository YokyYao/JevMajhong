import unittest
from tools.audit_endgame_soft_progress import action_expectation


class SoftProgressTests(unittest.TestCase):
    def test_probabilities_can_improve_while_greedy_choice_is_flat(self):
        values={'a':0.,'b':1.};before=action_expectation({'a':.6,'b':.4},values,list(values))
        after=action_expectation({'a':.55,'b':.45},values,list(values))
        self.assertEqual(before['greedy_key'],after['greedy_key'])
        self.assertGreater(after['soft_value'],before['soft_value'])

    def test_soft_value_must_not_replace_the_greedy_deployment_guard(self):
        values={'a':1.,'b':0.,'c':3.}
        before=action_expectation({'a':.4,'b':.35,'c':.25},values,list(values))
        after=action_expectation({'a':.1,'b':.46,'c':.44},values,list(values))
        self.assertGreater(after['soft_value'],before['soft_value'])
        self.assertLess(after['greedy_value'],before['greedy_value'])
        with self.assertRaises(ValueError):action_expectation({'a':float('nan'),'b':.5,'c':.5},values,list(values))


if __name__=='__main__':unittest.main()
