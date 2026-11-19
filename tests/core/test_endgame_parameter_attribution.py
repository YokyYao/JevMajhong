import unittest
from unittest.mock import patch
from tools.audit_endgame_parameter_attribution import control_comparison


class ParameterAttributionControls(unittest.TestCase):
    def test_key_order_does_not_create_a_spurious_control_failure(self):
        rows=[{'_meta':{'id':'control'},'questions':{'action':{'criteria':{'a':{},'b':{},'c':{}}}}}]
        expected={'control':{'a':.6,'b':.3,'c':.1}}
        actual={'control':{'c':.1,'b':.30005,'a':.59995}}
        with patch('jevmahjong.endgame.public_policy.action_keys',return_value=['a','b']):
            result=control_comparison(rows,actual,expected,1e-4)
        self.assertTrue(result['passed']);self.assertEqual(result['greedy_flips'],0)

    def test_same_argmax_is_insufficient_when_probabilities_are_not_reproduced(self):
        rows=[{'_meta':{'id':'control'},'questions':{'action':{'criteria':{'a':{},'b':{}}}}}]
        with patch('jevmahjong.endgame.public_policy.action_keys',return_value=['a','b']):
            result=control_comparison(rows,{'control':{'a':.9,'b':.1}},
                                      {'control':{'a':.6,'b':.4}},1e-4)
        self.assertFalse(result['passed']);self.assertEqual(result['greedy_flips'],0)

    def test_tiny_probability_change_that_flips_served_choice_is_rejected(self):
        rows=[{'_meta':{'id':'control'},'questions':{'action':{'criteria':{'a':{},'b':{}}}}}]
        with patch('jevmahjong.endgame.public_policy.action_keys',return_value=['a','b']):
            result=control_comparison(rows,{'control':{'a':.49999,'b':.50001}},
                                      {'control':{'a':.50001,'b':.49999}},1e-4)
        self.assertFalse(result['passed']);self.assertEqual(result['greedy_flips'],1)


if __name__=='__main__':unittest.main()
