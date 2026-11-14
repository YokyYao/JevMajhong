import json
from pathlib import Path
import tempfile
import unittest
import torch
from jevmahjong.training.label_balance import source_weights,weighted_loss
from jevmahjong.training.base_lora import question_loss


class LabelBalanceTests(unittest.TestCase):
    def test_source_mass_changes_gradient_without_repeating_positions_or_weighting_validation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'train.jsonl'
            records=[dict(_meta=dict(label_type='expert_replay'),questions={'action':dict(label='a0')}) for _ in range(9)]
            records.append(dict(_meta=dict(label_type='paired_future_masked'),questions={'action':dict(src='paired_future_masked',target={'a0':.7,'a1':.3})}))
            p.write_text(''.join(json.dumps(r)+'\n' for r in records));before=p.read_bytes()
            result=source_weights(p,dict(expert_replay=.75,paired_future_masked=.25),10)
            self.assertEqual(result['counts'],dict(expert_replay=9,paired_future_masked=1))
            self.assertEqual(result['actual_weighted_mass'],dict(expert_replay=.75,paired_future_masked=.25))
            self.assertAlmostEqual(result['mean_example_weight'],1.);self.assertEqual(result['copies_added'],0)
            self.assertFalse(result['validation_weighted']);self.assertEqual(p.read_bytes(),before)
            for q in [dict(label=0),dict(src='paired_future_masked',target=[.7,.3,0.])]:
                z=torch.tensor([.1,.2,3.],requires_grad=True);base=question_loss(z,q,'cpu')
                weighted=weighted_loss(z,q,'cpu',result['loss_weights'])
                label='paired_future_masked' if 'target' in q else 'expert_replay'
                self.assertAlmostEqual(float((weighted/base).detach()),result['loss_weights'][label],places=6)
                weighted.backward()
                if 'target' in q:self.assertEqual(float(z.grad[-1]),0.)

    def test_missing_coverage_mistagging_and_unbounded_rare_source_weight_are_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'train.jsonl';expert=dict(_meta=dict(label_type='expert_replay'),questions={'action':dict(label='a0')})
            p.write_text(json.dumps(expert)+'\n');mass=dict(expert_replay=.75,paired_future_masked=.25)
            with self.assertRaisesRegex(ValueError,'coverage'):source_weights(p,mass,1)
            fake=dict(_meta=dict(label_type='paired_future_masked'),questions={'action':dict(label='a0')})
            p.write_text(json.dumps(fake)+'\n')
            with self.assertRaisesRegex(ValueError,'declaration'):source_weights(p,mass,1)
            future=dict(_meta=dict(label_type='paired_future_masked'),questions={'action':dict(src='paired_future_masked',target={'a0':1.})})
            p.write_text((json.dumps(expert)+'\n')*99+json.dumps(future)+'\n')
            with self.assertRaisesRegex(ValueError,'eightfold'):source_weights(p,mass,100)


if __name__=='__main__':unittest.main()
