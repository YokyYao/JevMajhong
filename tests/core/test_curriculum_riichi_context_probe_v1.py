from collections import defaultdict
import unittest
import torch
from tools.curriculum_riichi_context_probe_v1 import FEATURES,RI_MASS,feature_vector,example_weights,standardize,confusion


class RiichiContextProbeTests(unittest.TestCase):
    def test_explicit_public_feature_allowlist_ignores_label_and_source(self):
        c={name:0 for name in FEATURES};before=feature_vector(c)
        c.update(label='reach',source='teacher',group_id='secret',future_win=True)
        self.assertEqual(feature_vector(c),before);self.assertEqual(len(before),2*len(FEATURES))

    def test_missingness_is_not_a_known_zero_and_nonfinite_is_rejected(self):
        c={name:0 for name in FEATURES};zero=feature_vector(c);c['safe_tenpai_discard_options']=None
        self.assertNotEqual(feature_vector(c),zero)
        c['turn']=float('nan')
        with self.assertRaises(ValueError):feature_vector(c)

    def test_training_normalization_and_fixed_source_class_mass(self):
        x=torch.zeros(3,2*len(FEATURES),dtype=torch.float64);x[:,0]=torch.tensor([1.,2.,3.])
        mean,std=standardize(x);self.assertEqual(float(mean[0]),2.);self.assertEqual(float(std[1]),1.)
        val=x+100;transformed=(val-mean)/std
        self.assertGreater(float(transformed[:,0].mean()),10.)
        sources=[key for i,key in enumerate(RI_MASS,1) for _ in range(i)];weights=example_weights(sources);mass=defaultdict(float)
        for source,weight in zip(sources,weights):mass[source]+=weight/len(sources)
        for source,value in RI_MASS.items():self.assertAlmostEqual(mass[source],value)
        with self.assertRaises(ValueError):example_weights(sources[:-4])

    def test_confusions_do_not_hide_source_or_binary_label_errors(self):
        sources=list(RI_MASS);truth=[s.endswith(':riichi_yes') for s in sources]
        result=confusion(sources,truth,truth);self.assertEqual(result['pooled']['balanced_recall'],1.)
        self.assertEqual(result['by_source']['legacy']['counts'],{'tp':1,'fn':0,'tn':1,'fp':0})
        with self.assertRaises(ValueError):confusion(sources,[not b for b in truth],truth)


if __name__=='__main__':unittest.main()
