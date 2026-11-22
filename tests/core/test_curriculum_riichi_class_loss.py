import copy
import math
import unittest

import torch
from tools.curriculum_riichi_class_loss import FIELD,annotate,parts,weighted_loss


class RiichiClassLossTests(unittest.TestCase):
    def q(self,kind,label,reach=2):
        return {'src':'curriculum1:teacher:'+kind,'label':label,FIELD:[] if kind=='plain' else [reach]}

    def test_plain_and_two_alternative_losses_equal_original_hard_ce(self):
        for question,values in ((self.q('plain',1),[.3,-.2,.1]),(self.q('riichi_yes',1,1),[.3,-.2]),
            (self.q('riichi_no',0,1),[.3,-.2])):
            z=torch.tensor(values,dtype=torch.float64,requires_grad=True);p=parts(z,question,'cpu')
            self.assertTrue(torch.allclose(p['objective'],p['standard_action_ce']))

    def test_replacement_changes_class_term_and_retains_conditional_discard_ce(self):
        for question in (self.q('riichi_yes',2),self.q('riichi_no',0)):
            z=torch.tensor([.3,-.2,.1],dtype=torch.float64,requires_grad=True);p=parts(z,question,'cpu')
            self.assertTrue(torch.allclose(p['standard_action_ce'],p['riichi_group_class_ce']+p['conditional_discard_ce']))
            self.assertTrue(torch.allclose(p['objective'],p['riichi_max_class_ce']+p['conditional_discard_ce']))
            self.assertFalse(torch.allclose(p['riichi_group_class_ce'],p['riichi_max_class_ce']))
        positive=parts(torch.zeros(10),self.q('riichi_yes',9,9),'cpu')
        self.assertAlmostEqual(float(positive['objective']),math.log(2),places=6)
        self.assertAlmostEqual(float(positive['standard_action_ce']),math.log(10),places=6)

    def test_tied_max_gradients_are_finite_shift_and_permutation_equivariant(self):
        z=torch.tensor([.2,.2,-.1],dtype=torch.float64,requires_grad=True);q=self.q('riichi_yes',2)
        value=parts(z,q,'cpu')['objective'];value.backward();gradient=z.grad.clone()
        self.assertAlmostEqual(float(gradient[0]),float(gradient[1]));self.assertAlmostEqual(float(gradient.sum()),0)
        perm=[2,0,1];other=z.detach()[perm].clone().requires_grad_();transformed=self.q('riichi_yes',0,0)
        actual=parts(other,transformed,'cpu')['objective'];actual.backward()
        self.assertTrue(torch.allclose(other.grad,gradient[perm]));self.assertTrue(torch.allclose(actual,value))
        shifted=parts(z.detach()+50,q,'cpu')['objective'];self.assertTrue(torch.allclose(shifted,value))

    def test_annotation_preserves_encoded_input_and_uses_actual_materializer_order(self):
        raw={'questions':{'action':{'src':'curriculum1:teacher:riichi_no','label':'b',
            'criteria':{'a':{'type':'dahai'},'r':{'type':'reach'},'b':{'type':'dahai'}}}}}
        encoded={'state':'original','questions':[{'qid':'action','keys':['r','b','a'],
            'label':1,'src':raw['questions']['action']['src'],'options':['R','B','A']}]}
        before=copy.deepcopy(encoded);raw_before=copy.deepcopy(raw);actual=annotate(encoded,raw)
        self.assertEqual(actual['questions'][0].pop(FIELD),[0]);self.assertEqual(actual,before);self.assertEqual(raw,raw_before)
        raw['questions']['action']['src']='curriculum1:teacher:riichi_yes'
        with self.assertRaises(ValueError):annotate(encoded,raw)

    def test_unannotated_or_inconsistent_targets_are_rejected_and_weights_apply_once(self):
        z=torch.tensor([.3,-.2,.1],dtype=torch.float64,requires_grad=True);q=self.q('riichi_no',0);observed=[]
        expected=parts(z,q,'cpu')['objective'];value=weighted_loss(z,q,'cpu',{'teacher:riichi_no':3.},observed)
        self.assertTrue(torch.allclose(value,3*expected));self.assertEqual(len(observed),1)
        for bad in (dict(q,label=2),dict(q,target=[1,0,0]),{'src':q['src'],'label':0},dict(q,**{FIELD:[1,2]})):
            with self.assertRaises(ValueError):parts(z,bad,'cpu')


if __name__=='__main__':unittest.main()
