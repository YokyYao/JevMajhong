import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import torch
from jevmahjong.core.io import digest
from jevmahjong.data.base_lora_pool import soft_target,complete_target,normalized_request,phase,position,prepare
from jevmahjong.training.base_lora import question_loss,target_cross_entropy,validation_profile


class BaseLoraPoolTests(unittest.TestCase):
    def test_outcome_target_prefers_pass_over_negative_calls_and_has_no_never_call_prior(self):
        zero=dict(utility=0.,own_win=0.,fast_reward=0.)
        bad=dict(utility=-.2,own_win=.1,fast_reward=.1)
        good=dict(utility=.2,own_win=.1,fast_reward=.1)
        self.assertGreater(soft_target([zero,bad],.05)[0],.98)
        self.assertGreater(soft_target([zero,good],.05)[1],.98)
        incompatible=dict(utility=.2,own_win=-.1,fast_reward=.1)
        self.assertEqual(soft_target([zero,incompatible],.05),[.5,.5])
        self.assertEqual(incompatible['utility'],.2)
        with self.assertRaises(ValueError):soft_target([good],0.)

    def test_full_menu_credit_is_soft_supervision_not_the_intervention_action(self):
        control=dict(utility=.3,own_win=.25,fast_reward=.2,dealin=0.,net_points=1000.)
        reward={'mode':'registered'};rows=[]
        for action in (1,2):
            branch=dict(control,utility=.1,own_win=.125)
            rows.append(dict(public_request=dict(state={},questions={'action':{'criteria':dict(a0={'type':'none'},a1={'type':'chi'},a2={'type':'pon'})}}),
                eligible=[0,1,2],teacher_action=0,action=action,_source_reward=reward,training_credit=dict(control=control,branch=branch)))
        target,credits=complete_target(rows,reward,.05)
        self.assertGreater(target['a0'],.96);self.assertEqual(len(credits),3)
        with self.assertRaisesRegex(ValueError,'Incomplete'):complete_target(rows[:1],reward,.05)
        wrong=copy.deepcopy(rows)
        wrong[1]['training_credit']['control']=dict(control,net_points=0.)
        with self.assertRaisesRegex(ValueError,'share control'):complete_target(wrong,reward,.05)
        partial=copy.deepcopy(rows[:1]);partial[0]['eligible']=[0,1]
        target,_=complete_target(partial,reward,.05)
        self.assertEqual(target['a2'],0.);self.assertAlmostEqual(sum(target.values()),1.)

    def test_unknown_legal_logit_has_no_direct_negative_gradient_and_validation_agrees(self):
        z=torch.tensor([1.,0.,9.],dtype=torch.float64,requires_grad=True)
        q=dict(target=[.75,.25,0.],src='paired_future_masked')
        loss=question_loss(z,q,'cpu');loss.backward()
        self.assertEqual(float(z.grad[2]),0.)
        self.assertGreater(float(z.grad[1]),0.)
        raw=dict(criteria={'a0':{},'a1':{},'a2':{}},label='a0',target={'a0':.75,'a1':.25,'a2':0.},src='paired_future_masked')
        actual=target_cross_entropy(raw,list(raw['criteria']),z.detach().softmax(-1).tolist())
        self.assertAlmostEqual(actual,float(loss.detach()),places=12)
        # An expert's one-hot target keeps full-menu normalization.
        z2=z.detach().clone().requires_grad_(True)
        question_loss(z2,dict(label=0),'cpu').backward()
        self.assertGreater(float(z2.grad[2]),0.)

    def test_validation_selection_uses_phase_weights_and_correct_masked_loss(self):
        records=[];rows=[]
        for i,phase in enumerate(['closed_discard','kan_window']):
            q=dict(criteria=dict(a0={},a1={},a2={}),label='a0')
            if i:q.update(target=dict(a0=.75,a1=.25,a2=0.),src='paired_future_masked')
            records.append(dict(questions={'action':q},_meta=dict(id=str(i),phase=phase,label_type='expert_replay' if i==0 else 'paired_future_masked')))
            rows.append(dict(id=str(i),keys=['a0','a1','a2'],p=[.2,.1,.7]))
        r=validation_profile(records,rows,dict(closed_discard=.9,kan_window=.1))
        expected=.9*target_cross_entropy(records[0]['questions']['action'],rows[0]['keys'],rows[0]['p'])+.1*target_cross_entropy(records[1]['questions']['action'],rows[1]['keys'],rows[1]['p'])
        self.assertAlmostEqual(r['weighted_target_ce'],expected)
        self.assertNotIn('kan_window',r['expert_accuracy'])
        with self.assertRaisesRegex(ValueError,'Validation dropped'):validation_profile(records,rows[:1],dict(closed_discard=1.))

    def test_public_normalization_removes_diagnostic_metadata_and_unifies_instruction(self):
        request=dict(state=dict(seat=0,hand='1m 2m',rivers=['']*4,melds=[[],[],[],[]]),
            questions={'action':dict(type='choice',instructions='maximize points',criteria={'a0':{'type':'dahai','pai':'1m'}},label='a0')},
            _meta=dict(future_seed='secret-label-field',training_credit={'own_win':1}))
        result=normalized_request(request,'replay:one','expert_replay',0)
        self.assertNotIn('future_seed',result['_meta']);self.assertNotIn('training_credit',result['_meta'])
        self.assertIn('winning sooner',result['questions']['action']['instructions'])
        self.assertEqual(phase(result),'closed_discard')
        self.assertEqual(position(request,'g'),position(result,'g'))
        request['state']['opponent_hands']=['hidden']
        with self.assertRaisesRegex(ValueError,'public-state'):normalized_request(request,'g','expert_replay',0)

    def test_real_pool_preparation_blocks_historical_groups_and_keeps_raw_request_views_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';data.mkdir();catalog=root/'catalog';catalog.mkdir()
            def record(group,split,view='native'):
                return dict(state=dict(seat=0,hand='1m 2m',rivers=['']*4,melds=[[],[],[],[]]),
                    questions={'action':dict(type='choice',instructions='old',criteria={'a0':{'type':'dahai','pai':'1m'}},label='a0')},
                    _meta=dict(group_id=group,id=group+'/0',split=split,view=view,schema='clean-full-policy-v1',call_repair_version='public-chi-pon-postcall-v2'))
            files={}
            for split,rows in [('train',[record('t','train'),record('t','train','screen'),record('v','train')]),('validation',[record('v','validation')])]:
                file=data/(split+'.jsonl');file.write_text(''.join(json.dumps(r)+'\n' for r in rows));files[split]=digest(file)
            provenance=data/'provenance.json';provenance.write_text(json.dumps({'files':files}))
            source=dict(path=str(data),kind='replay_jsonl',catalog_status='indexed_not_training_admission',provenance=str(provenance),provenance_sha256=digest(provenance),train_checksum_matches_provenance=True,holdout_group_metadata_present=True)
            (catalog/'sources-private.json').write_text(json.dumps([source]));(catalog/'result.json').write_text('{}')
            db=sqlite3.connect(catalog/'index-private.sqlite');db.execute('CREATE TABLE groups (group_key TEXT,split TEXT)');db.execute("INSERT INTO groups VALUES ('replay:v','validation')");db.commit();db.close()
            reward=dict(mode='own_win_first_v1',win_weight=1.,fast_weight=.25,dealin_penalty=2.,net_weight=0.,point_scale=8000.,speed_gamma=.98)
            cfg=dict(catalog=str(catalog),catalog_index_sha256=digest(catalog/'index-private.sqlite'),catalog_result_sha256=digest(catalog/'result.json'),catalog_sources_sha256=digest(catalog/'sources-private.json'),out=str(root/'out'),reward=reward,soft_target_mode='outcome_only_win_compatible_masked_v2',target_temperature=.05,sources=[dict(path=str(data),priority=1,validation=True)])
            result=prepare(cfg)
            self.assertEqual(result['groups_overlap'],0)
            self.assertEqual(result['statistics']['historically_held_train_rows_blocked'],1)
            self.assertEqual(result['statistics']['duplicate_position_rows'],1)
            self.assertEqual(sum(s['positions'] for s in result['strata']),2)
            self.assertFalse(result['cached_encoder_features_used']);self.assertFalse(result['test_payloads_opened'])


if __name__=='__main__':unittest.main()
