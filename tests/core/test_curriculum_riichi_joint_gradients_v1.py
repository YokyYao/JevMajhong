import unittest
import torch
from tools.diagnostics.riichi_joint_gradients_v1 import geometry,select_panels,MASS


class RiichiJointGradientTests(unittest.TestCase):
    def test_known_opposition_orthogonality_and_zero_are_distinct(self):
        result=geometry({'a':torch.tensor([1.,0.]),'b':torch.tensor([-1.,0.]),'c':torch.tensor([0.,2.]),'zero':torch.zeros(2)})
        self.assertEqual(result['cosines']['a|b'],-1.);self.assertEqual(result['cosines']['a|c'],0.)
        self.assertIsNone(result['cosines']['a|zero']);self.assertEqual(result['norms']['c'],2.)

    def test_parameter_partition_uses_only_its_own_coordinates(self):
        values={'a':torch.tensor([1.,2.,0.]),'b':torch.tensor([-1.,2.,0.])}
        selected=geometry(values,torch.tensor([True,False,False]));self.assertEqual(selected['cosines']['a|b'],-1.)
        with self.assertRaises(ValueError):geometry({'a':torch.tensor([float('nan')])})

    def test_panels_deterministic_label_blind_and_group_distinct_per_source(self):
        rows=[]
        for source in MASS:
            for i in range(12):rows.append({'_meta':{'id':source+str(i),'group_id':source+str(i//1)},'questions':{'action':{'src':'curriculum1:'+source,'label':'a'}}})
        before=select_panels(rows)
        for row in rows:row['questions']['action']['label']='another'
        after=select_panels(list(reversed(rows)))
        self.assertEqual(before,after)
        for source in MASS:
            groups=[r['_meta']['group_id'] for panel in after for r in panel[source]]
            self.assertEqual(len(groups),8);self.assertEqual(len(set(groups)),8)

    def test_missing_source_and_insufficient_groups_rejected(self):
        with self.assertRaises(ValueError):select_panels([])
        rows=[{'_meta':{'id':source,'group_id':'same'},'questions':{'action':{'src':'curriculum1:'+source}}} for source in MASS]
        with self.assertRaises(ValueError):select_panels(rows)


if __name__=='__main__':unittest.main()
