import json
from pathlib import Path
import tempfile
import unittest
from tools.build_curriculum_stage1 import holdout,context_gaps,public_input,stratum
from tools.audit_curriculum_shapes import distances,winning_shapes,tile_index
from tools.train_curriculum import weighted_stage1_loss

class CurriculumDataTests(unittest.TestCase):
    def test_metadata_and_loss_tags_do_not_change_visible_input(self):
        import copy
        row={'state':{'hand':'1m 2m'},'questions':{'action':{'type':'choice','instructions':'choose',
             'criteria':{'a':{'type':'reach'},'b':{'type':'dahai','pai':'1m'}},'label':'a','src':'expert_replay'}}}
        tagged=copy.deepcopy(row);tagged['questions']['action']['src']='curriculum1:riichi_yes'
        tagged['_meta']={'group_id':'different','future_yaku':'irrelevant'}
        self.assertEqual(public_input(row),public_input(tagged))
        self.assertEqual(stratum(row),'riichi_yes')
        self.assertEqual(holdout('same-public-group'),holdout('same-public-group'))
        self.assertIn('riichi',context_gaps(row))

    def test_ui_pair_count_does_not_establish_best_hand_route(self):
        regular,seven=distances('1m 1m 2m 2m 3m 3m 4m 4m 5m 5m 6m 7m 8m'.split())
        self.assertLess(regular,seven)
        self.assertEqual(tile_index('E'),tile_index('1z'))
        self.assertEqual(tile_index('5mr'),tile_index('0m'))

    def test_ron_shape_reconstruction_without_pai_on_hora(self):
        events=[{'type':'start_kyoku','tehais':[
            '1m 1m 2m 2m 3p 3p 4p 4p 5s 5s E E C'.split(),['C'],[],[]]},
            {'type':'dahai','actor':1,'pai':'C'},{'type':'hora','actor':0,'target':1}]
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'games.jsonl';file.write_text(json.dumps(dict(policy='candidate',seat=0,mjai_log=events))+'\n')
            result,_=winning_shapes(file)
            self.assertEqual(result['candidate']['controlled_wins'],1)
            self.assertEqual(result['candidate']['seven_shape_only'],1)

    def test_loss_only_weight_scales_gradient_without_changing_target(self):
        import torch
        logits=torch.tensor([.2,-.3],requires_grad=True)
        q={'label':1,'src':'curriculum1:riichi_yes'};observer=[]
        loss=weighted_stage1_loss(logits,q,'cpu',{'riichi_yes':4.},observer)
        expected=4*torch.nn.functional.cross_entropy(logits[None],torch.tensor([1]))
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward();self.assertGreater(logits.grad[0].item(),0)
        self.assertEqual(observer[0][0],'riichi_yes')
