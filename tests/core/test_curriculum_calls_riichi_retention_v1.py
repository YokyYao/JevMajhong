import copy
import unittest
import torch
from tools.build_curriculum_calls_repair import MASS
from tools.curriculum_riichi_class_loss import FIELD,parts as basic_parts
from tools.curriculum_calls_riichi_retention_v1 import MARKER,METHOD,annotate,parts,weighted_loss
from tools.train_curriculum_calls_repair import weighted_loss as old_weighted_loss


class CallsRiichiRetentionTests(unittest.TestCase):
    def row(self,key,types,label=0):
        keys=['a'+str(i) for i in range(len(types))];src='curriculum2:'+key
        raw=dict(src=src,label=keys[label],criteria={k:{'type':t} for k,t in zip(keys,types)})
        encoded=dict(ids=[5,7],questions=[dict(qid='action',src=src,label=label,keys=keys)])
        return encoded,{'questions':{'action':raw}}

    def test_private_annotation_keeps_materialized_and_visible_fields(self):
        encoded,request=self.row('basic:teacher:riichi_no',['dahai','reach','dahai'],2)
        before=copy.deepcopy(encoded);raw_before=copy.deepcopy(request)
        annotate(encoded,request);question=encoded['questions'][0]
        self.assertEqual(question[FIELD],[1]);self.assertEqual(question[MARKER],METHOD)
        stripped=copy.deepcopy(encoded)
        for q in stripped['questions']:q.pop(FIELD);q.pop(MARKER)
        self.assertEqual(stripped,before);self.assertEqual(request,raw_before)

    def test_basic_RI_values_and_gradients_equal_active_six_source_objective(self):
        for kind,label in (('riichi_yes',1),('riichi_no',2)):
            encoded,request=self.row('basic:legacy:'+kind,['dahai','reach','dahai'],label)
            annotate(encoded,request);q=encoded['questions'][0]
            x=torch.tensor([.1,-.2,.5],dtype=torch.float64,requires_grad=True);y=x.detach().clone().requires_grad_()
            actual=parts(x,q,'cpu');expected=basic_parts(y,dict(q,src='curriculum1:legacy:'+kind),'cpu')
            self.assertEqual(float(actual['objective'].detach()),float(expected['objective'].detach()))
            self.assertTrue(torch.equal(torch.autograd.grad(actual['objective'],x)[0],torch.autograd.grad(expected['objective'],y)[0]))

    def test_plain_and_advanced_values_and_gradients_are_old_calls_CE(self):
        weights={key:1.7 for key in MASS}
        for key in ('basic:teacher:plain','advanced:teacher:call','advanced:legacy:kan','advanced:teacher:post_meld'):
            # An advanced reach/kan alternative remains the old calls CE;
            # this proposal deliberately changes only basic RI replay.
            types=['dahai','dahai','dahai'] if key.startswith('basic:') else ['dahai','reach','ankan']
            encoded,request=self.row(key,types,1);old=copy.deepcopy(encoded['questions'][0]);annotate(encoded,request)
            for dtype in (torch.float64,torch.float32):
                x=torch.tensor([.1,-.2,.5],dtype=dtype,requires_grad=True);y=x.detach().clone().requires_grad_()
                a=weighted_loss(x,encoded['questions'][0],'cpu',weights);b=old_weighted_loss(y,old,'cpu',weights)
                self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(torch.autograd.grad(a,x)[0],torch.autograd.grad(b,y)[0]))

    def test_soft_targets_new_strata_relabelled_keys_and_double_annotation_rejected(self):
        for change in ('soft','new_source','wrong_label'):
            encoded,request=self.row('basic:teacher:riichi_yes',['dahai','reach'],1)
            if change=='soft':request['questions']['action']['target']=[.2,.8]
            elif change=='new_source':request['questions']['action']['src']='curriculum2:basic:teacher:post_reach_discard'
            else:encoded['questions'][0]['label']=0
            with self.assertRaises(ValueError):annotate(encoded,request)
        encoded,request=self.row('advanced:legacy:call',['none','pon'],1);annotate(encoded,request)
        with self.assertRaises(ValueError):annotate(encoded,request)

    def test_observer_has_objective_and_original_CE_and_no_eval_side_effects(self):
        encoded,request=self.row('basic:teacher:riichi_no',['dahai','reach','dahai'],2);annotate(encoded,request)
        weights={key:1.0 for key in MASS};x=torch.tensor([.1,-.2,.5],requires_grad=True);observed=[]
        weighted_loss(x,encoded['questions'][0],'cpu',weights,observed)
        self.assertEqual(len(observed),1);self.assertIn('standard_action_ce',observed[0]);self.assertIn('objective',observed[0])
        with torch.no_grad():
            ignored=[];weighted_loss(x,encoded['questions'][0],'cpu',weights,ignored)
        self.assertEqual(ignored,[])


if __name__=='__main__':unittest.main()
