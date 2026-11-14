import json
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from jevmahjong.training.label_balance import source_weights,weighted_loss,tag_training_sources,PHASE_PREFIX
from jevmahjong.training.base_lora import question_loss


class LabelBalanceTests(unittest.TestCase):
    def phase_record(self,phase):
        kinds={'closed_discard':['dahai','dahai'],'open_discard':['dahai','dahai'],
            'call_window':['pon','none'],'kan_window':['ankan','dahai']}[phase]
        return dict(state={'seat':0,'melds':[[{'type':1}]]+[[]]*3 if phase=='open_discard' else [[]]*4},
            _meta={'phase':phase,'label_type':'paired_future_masked'},questions={'action':dict(type='choice',
                instructions='choose',criteria={f'a{i}':{'type':k} for i,k in enumerate(kinds)},
                label='a0',target={'a0':.5,'a1':.5},src='paired_future_masked')})

    def test_phase_mass_is_actual_coefficient_mass_without_changing_examples(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'train.jsonl';expert=dict(_meta={'label_type':'expert_replay'},questions={'action':dict(label='a0')})
            mass=dict(expert_replay=.75,paired_future_masked=.25)
            phases=dict(closed_discard=.4,call_window=.3,kan_window=.2,open_discard=.1)
            rows=[copy.deepcopy(expert) for _ in range(48)]+[self.phase_record(name) for name in phases for _ in range(4)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows));before=p.read_bytes()
            proof=source_weights(p,mass,64,phases)
            for name,share in phases.items():
                self.assertAlmostEqual(proof['actual_weighted_mass_by_MC_phase'][name],.25*share)
                self.assertAlmostEqual(proof['MC_phase_loss_weights'][name],4*share)
            self.assertAlmostEqual(proof['mean_example_weight'],1.)
            self.assertAlmostEqual(proof['actual_weighted_mass']['expert_replay'],.75)
            original=copy.deepcopy(rows);tags=tag_training_sources(rows,proof['MC_phase_counts'])
            self.assertEqual(tags['tagged_MC_phase_counts'],{p:4 for p in phases})
            self.assertFalse(tags['source_tag_is_model_input']);self.assertEqual(p.read_bytes(),before)
            # All model inputs, targets, menus and expert records stay identical;
            # the training-only source tag is the sole transformed field.
            for after,old in zip(rows,original):
                restored=copy.deepcopy(after)
                if old['_meta']['label_type']=='paired_future_masked':restored['questions']['action']['src']='paired_future_masked'
                self.assertEqual(restored,old)
            with self.assertRaisesRegex(ValueError,'existing'):tag_training_sources(rows,proof['MC_phase_counts'])

    def test_phase_weighted_masked_loss_keeps_unknown_action_gradient_zero(self):
        weights=dict(expert_replay=.8,paired_future_masked=2.)
        for phase,value in [('closed_discard',5.),('kan_window',.5)]:
            z=torch.tensor([.1,.2,4.],requires_grad=True)
            q=dict(src=PHASE_PREFIX+phase,target=[.7,.3,0.]);before=copy.deepcopy(q)
            actual=weighted_loss(z,q,'cpu',weights,{phase:value})
            expected=value*question_loss(z,dict(q,src='paired_future_masked'),'cpu')
            self.assertTrue(torch.equal(actual,expected));actual.backward()
            self.assertEqual(float(z.grad[-1]),0.);self.assertEqual(q,before)
        with self.assertRaisesRegex(ValueError,'Missing MC phase'):weighted_loss(z,dict(src='paired_future_masked',target=[.5,.5,0.]),'cpu',weights,{'closed_discard':5.})
        with self.assertRaisesRegex(ValueError,'Unexpected'):weighted_loss(z,q,'cpu',weights)

    def test_missing_or_forged_public_phase_and_rare_overweight_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'train.jsonl';mass=dict(expert_replay=.75,paired_future_masked=.25)
            phases=dict(closed_discard=.4,call_window=.3,kan_window=.2,open_discard=.1)
            expert=dict(_meta={'label_type':'expert_replay'},questions={'action':dict(label='a0')})
            rows=[copy.deepcopy(expert) for _ in range(96)]+[self.phase_record(name) for name in phases]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'eightfold'):source_weights(p,mass,100,phases)
            rows[-1]['_meta']['phase']='closed_discard';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'public legal'):source_weights(p,mass,100,phases)
            valid=[self.phase_record('closed_discard')];before=copy.deepcopy(valid)
            with self.assertRaisesRegex(ValueError,'coverage'):tag_training_sources(valid,{'closed_discard':2})
            self.assertEqual(valid,before)
            valid[0]['questions']['action']['src']=PHASE_PREFIX+'closed_discard'
            p.write_text(json.dumps(valid[0])+'\n')
            with self.assertRaisesRegex(ValueError,'loss-only'):source_weights(p,mass,1,phases)

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
