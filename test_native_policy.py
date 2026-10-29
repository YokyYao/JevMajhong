import unittest
from unittest.mock import patch

from native_policy import NativeModelAgent


class NativePolicyTests(unittest.TestCase):
    def test_recommendation_uses_the_frozen_evaluation_loading_contract(self):
        # JevAny defaults to merge=True; the evaluated head-only policy explicitly does not merge.
        with patch('jevany.JevModel.from_pretrained',return_value=object()) as load:
            agent=NativeModelAgent('local-checkpoint',guarded=True)
        args=load.call_args.kwargs
        self.assertFalse(args['options'].merge)
        self.assertEqual(args['options'].temperature,1.)
        self.assertEqual(args['device'],'cuda')
        self.assertEqual(args['dtype'],'bf16')
        self.assertEqual(args['inference_options'].max_state_tokens,4096)
        self.assertEqual(args['inference_options'].max_branch_tokens,8192)
        self.assertEqual(args['inference_options'].max_packed_tokens,8192)
        self.assertTrue(agent.guarded)
        self.assertEqual(agent.calls,0)


if __name__=='__main__':unittest.main()
