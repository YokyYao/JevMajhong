import copy
from types import SimpleNamespace
import unittest
import torch
from tools import curriculum_call_type_loss_v1 as loss
from tests.core import test_curriculum_calls_classroute_v1 as fixtures


class CallTypeLossTests(unittest.TestCase):
    def fixture(self):
        helper=fixtures.CallsClassRouteTests();rows,_=helper.fixture()
        entries,_=loss.recipe.build(rows);index=loss.CallTypeIndex(rows,entries)
        return helper,rows,entries,index

    def test_loss_and_gradients_replace_the_source_scalar_once(self):
        helper,rows,entries,index=self.fixture();pending=[];expected_values=[]
        for row,entry in zip(rows,entries):
            plain=loss.original.annotate(helper.encoded(row),row)
            encoded=index.attach(copy.deepcopy(plain),row);q=encoded['questions'][0]
            a=torch.linspace(-.7,.9,len(q['keys']),dtype=torch.float64,requires_grad=True)
            b=a.detach().clone().requires_grad_()
            actual=loss.weighted_loss(a,q,'cpu',pending)
            expected=loss.original.parts(b,plain['questions'][0],'cpu')['objective']*entry['weight']
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(torch.autograd.grad(actual,a)[0],torch.autograd.grad(expected,b)[0]))
            expected_values.append(float(expected.detach()))
        summary=loss.summarize(pending)
        self.assertAlmostEqual(summary['loss'],sum(expected_values)/len(expected_values))
        self.assertEqual(sum(summary['source_counts'].values()),len(rows))

    def test_annotations_follow_encoding_and_raw_and_tokens_stay_unchanged(self):
        helper,rows,entries,index=self.fixture();before=copy.deepcopy(rows)
        encodings=[]
        def encoder(model,tok,args,chunk,epoch):
            variants=[]
            for raw in chunk:
                rec=loss.original.annotate(helper.encoded(raw),raw)
                self.assertNotIn(loss.FIELD,rec['questions'][0])
                enc={'ids':[1,2,3]};encodings.append(enc)
                variants.append(SimpleNamespace(rec=rec,enc=enc))
            return variants
        args=SimpleNamespace(batch=1,rlcr=False,p_none=0.,p_none_distract=0.,p_distract=0.,p_none_pair=0.)
        for i,row in enumerate(rows):
            variant=index.encode_batch(encoder,None,None,args,[row],0)[0]
            self.assertIs(variant.enc,encodings[i])
            self.assertEqual(variant.enc,{'ids':[1,2,3]})
            self.assertIn(loss.FIELD,variant.rec['questions'][0])
        self.assertEqual(rows,before)
        args.p_none=.1
        with self.assertRaises(ValueError):index.encode_batch(encoder,None,None,args,[rows[0]],0)

    def test_corrupt_index_wrong_label_double_annotations_and_weights_rejected(self):
        helper,rows,entries,index=self.fixture()
        altered=copy.deepcopy(entries);altered[0]['weight']*=2
        with self.assertRaises(ValueError):loss.CallTypeIndex(rows,altered)
        row=rows[0];encoded=loss.original.annotate(helper.encoded(row),row)
        index.attach(encoded,row)
        with self.assertRaises(ValueError):index.attach(encoded,row)
        encoded['questions'][0][loss.FIELD]=float('nan')
        with self.assertRaises(ValueError):loss.weighted_loss(torch.tensor([1.,2.]),encoded['questions'][0],'cpu')
        plain=loss.original.annotate(helper.encoded(row),row)
        plain['questions'][0]['label']=1-plain['questions'][0]['label']
        with self.assertRaises(ValueError):index.attach(plain,row)


if __name__=='__main__':unittest.main()
