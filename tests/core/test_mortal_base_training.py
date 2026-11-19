import math
import unittest
from tools.train_mortal_base import learning_rate_scale,progress,validate_reference_protocol


class FreshBaseTrainingTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(warmup_updates=64,max_updates=1536,cosine_final_fraction=.1,
                    progress_relative_delta=.001,early_stop_patience=4)

    def test_warmup_and_decay_reach_registered_values(self):
        self.assertAlmostEqual(learning_rate_scale(1,self.c),1/64)
        self.assertEqual(learning_rate_scale(64,self.c),1.)
        self.assertAlmostEqual(learning_rate_scale(1536,self.c),.1)
        self.assertGreater(learning_rate_scale(768,self.c),.1)

    def test_early_noise_cannot_stop_or_select_underexposed_model(self):
        s=progress(0,2.,None,768,self.c,False)
        for update in range(64,768,64):s=progress(update,3.,s,768,self.c,True)
        self.assertFalse(s['stop']);self.assertIsNone(s['selected_update'])
        s=progress(768,3.,s,768,self.c,False)
        self.assertIsNone(s['selected_update'])
        s=progress(768,3.,s,768,self.c,True)
        self.assertEqual(s['selected_update'],768);self.assertEqual(s['stale'],0)
        for update in (832,896,960):s=progress(update,3.1,s,768,self.c,True)
        self.assertFalse(s['stop'])
        s=progress(1024,3.1,s,768,self.c,True);self.assertTrue(s['stop'])

    def test_later_improvement_selects_candidate_and_resets_patience(self):
        s=progress(768,1.,None,768,self.c,True)
        s=progress(832,1.1,s,768,self.c,True)
        s=progress(896,.9,s,768,self.c,True)
        self.assertEqual(s['selected_update'],896);self.assertEqual(s['stale'],0)
        with self.assertRaises(ValueError):progress(960,math.nan,s,768,self.c,True)

    def test_cached_parent_uses_bundle_digest_and_exact_opponent_grid(self):
        p=dict(arms={'parent':{'identity_sha256':'abc','files':{}}},seeds=[10,11],
               profiles=['calling','mortal'],seats=4,same_formatter_and_guard=True)
        validate_reference_protocol(p,[10,11],'abc')
        with self.assertRaises(ValueError):validate_reference_protocol(p,[10,12],'abc')
        with self.assertRaises(ValueError):validate_reference_protocol(p,[10,11],'other')


if __name__=='__main__':unittest.main()
