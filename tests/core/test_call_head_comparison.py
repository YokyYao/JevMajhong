"""Matched head continuation retains the calling MLP and audits actual win credit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from jevmahjong.core.io import digest
from jevmahjong.policy.call_baseline import transform
from jevmahjong.policy.call_tile_context import VERSION
from jevmahjong.policy.kan_context import feature_names
from jevmahjong.training.context_network import make_network
from jevmahjong.training.call_head import anchor_policy,check_matched_context,fit,load_inputs
from jevmahjong.evaluation.call_head_audit import audit
from tests.core.test_win_context import FrozenHead,pair


class CallHeadComparisonTests(unittest.TestCase):
    def test_unanchored_comparison_requires_explicit_frozen_baseline_and_no_extra_data(self):
        cfg=dict(anchor_policy='same_data_no_anchors',call_context_baseline={'sha256':'fixed'},
            selection_objective='hand_win',anchor_beta=0.,anchor_kl_limit=0.,
            reference_anchor_states=0,anchor_batch=0)
        self.assertEqual(anchor_policy(cfg),'same_data_no_anchors')
        for change in (dict(call_context_baseline=None),dict(anchor_beta=.1),dict(anchor_batch=1),
                dict(reference_anchor_data=['extra']),dict(route_loss_weights={'open':2}),
                dict(initial_heads={'checkpoint':'other'})):
            with self.subTest(change=change),self.assertRaises(ValueError):anchor_policy(dict(cfg,**change))

    def test_real_head_fit_and_independent_win_audit_keep_fixed_context(self):
        self._real_fit_and_audit('same_data_no_anchors')

    def test_native_train_anchors_keep_fixed_context_and_exclude_labelled_and_held_inputs(self):
        self._real_fit_and_audit('calling_baseline_kl')

    def test_win_compatible_real_optimizer_and_independent_target_loss_audit(self):
        self._real_fit_and_audit('calling_baseline_kl','win_compatible_v1')

    def _real_fit_and_audit(self,mode,target_mode='utility_v1'):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);roles={}
            for role in ('discard','call'):
                directory=root/role;directory.mkdir();torch.save({},directory/'head.pt')
                roles[role+'_checkpoint']=str(directory)
            torch.manual_seed(83);net=make_network(4,len(feature_names(VERSION)))
            with torch.no_grad():net[2].weight.fill_(.002)
            context=dict(version=VERSION,names=list(feature_names(VERSION)),model_kind='mlp',hidden=4,
                state_dict=copy.deepcopy(net.state_dict()),reference_prior_strength=.05)
            checkpoint=root/'baseline.pt';torch.save(dict(kan_context=context),checkpoint)
            baseline=dict(checkpoint=str(checkpoint),sha256=digest(checkpoint))
            reward=dict(mode='own_win_first_v1',win_weight=1.,fast_weight=.25,
                dealin_penalty=2.,net_weight=0.,point_scale=8000.,speed_gamma=.98)
            def log(own,points):
                return [dict(type='start_kyoku'),dict(type='tsumo',actor=1),
                    dict(type='hora',actor=0 if own else 1,target=1 if own else 2,
                        deltas=[points,-points,0,0] if own else [0,1000,-1000,0]),dict(type='end_kyoku')]
            parts={};source=root/'collection/source_0';(source/'walls-private').mkdir(parents=True)
            raw=[]
            for split,seed in [('train',100),('held',103)]:
                row=pair(seed);row.update(hand_index=0,draws_before=0)
                row['reference']=transform(row['public_request'],FrozenHead()(row['decide'],row['options']).detach().softmax(-1),context)
                row['training_credit']=dict(control=dict(own_win=.25,fast_reward=.245,dealin=0.,net_points=2000.,utility=.31125),
                    branch=dict(own_win=.5,fast_reward=.49,dealin=0.,net_points=500.,utility=.6225))
                row['training_advantage']=.31125
                row['counterfactual_worlds']=[dict(control={'mjai_log':log(i==0,8000)},
                    branch={'mjai_log':log(i<2,1000)}) for i in range(4)]
                parts[split]=[row];raw.append(row)
            shard=source/'walls-private/100.pt';torch.save({'rows':raw},shard)
            hashes={str(shard):digest(shard)};sources=[dict(path=str(source),seed_start=100,seeds=4,walls=4)]
            data_audit=root/'data-audit';data_audit.mkdir()
            (data_audit/'result.json').write_text(json.dumps(dict(status='passed_complete')))
            (data_audit/'protocol.json').write_text(json.dumps(dict(config={'collection':str(root/'collection')},
                model_files={},code_sha256={},source_files=hashes)))
            cfg=dict(**roles,out=str(root/'fit'),model_id='toy-head-comparison',seed=17,
                train_scopes=['calling'],decision_filter='all',fit_model='complete_option_call_head',
                call_context_baseline=baseline,selection_objective='hand_win',max_held_dealin_increase=.01,
                anchor_policy='same_data_no_anchors',anchor_beta=0.,anchor_kl_limit=0.,
                reference_anchor_states=0,anchor_batch=0,reference_prior_strength=.05,
                lr=.5,weight_decay=0.,epochs=12,patience_epochs=3,batch=1,
                data=[str(source)],required_data_audits=[str(data_audit)],reward=reward)
            native=root/'native';native.mkdir()
            (native/'protocol.json').write_text(json.dumps({'collection_method':'native_call_states_v1'}))
            anchor=pair(100);anchor.update(episode_key=(100,0,7),request_sha256='different-public-window')
            anchor['public_request']['state']['honba']=1
            anchor['reference']=transform(anchor['public_request'],FrozenHead()(anchor['decide'],anchor['options']).detach().softmax(-1),context)
            anchor.pop('training_credit');anchor.pop('training_advantage')
            native_parts={'train':[copy.deepcopy(parts['train'][0]),anchor],'held':[copy.deepcopy(parts['held'][0])]}
            native_audit=root/'native-audit';native_audit.mkdir()
            (native_audit/'result.json').write_text(json.dumps({'status':'passed_complete'}))
            (native_audit/'protocol.json').write_text(json.dumps(dict(config={'collection':str(native)},
                model_files={},code_sha256={},source_files={})))
            if mode=='calling_baseline_kl':
                cfg.update(anchor_policy=mode,anchor_beta=.001,anchor_kl_limit=100.,
                    anchor_batch=1,reference_anchor_states=1,reference_anchor_data=[str(native)],
                    required_data_audits=[str(data_audit),str(native_audit)])
                for change in (dict(call_context_baseline=None),dict(reference_anchor_data=[]),
                        dict(matched_context_fit={'config':'old-comparison'})):
                    with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Calling-baseline KL'):
                        anchor_policy(dict(cfg,**change))
            if target_mode!='utility_v1':cfg['distillation_target']=target_mode
            other=dict(cfg,fit_model='residual_policy',context_initialization='calling_baseline',policy_state_weighting='uniform')
            counterpart=root/'context-fit.json';counterpart.write_text(json.dumps(other))
            if mode=='same_data_no_anchors':cfg['matched_context_fit']=dict(config=str(counterpart),sha256=digest(counterpart))
            check_matched_context(cfg)
            for change in (dict(lr=.1),dict(seed=18),dict(data=['other-data'])) if mode=='same_data_no_anchors' else ():
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'fitting budget differ'):
                    check_matched_context(dict(cfg,**change))
            fitfile=root/'fit.json';fitfile.write_text(json.dumps(cfg))
            snapshot=lambda c:dict(config=c,model_files={},code_sha256={},call_context_baseline=baseline,
                **({'matched_context_fit':cfg['matched_context_fit']} if 'matched_context_fit' in cfg else {}))
            def inputs(paths,*args,**kwargs):
                if paths==[str(native)]:
                    self.assertTrue(kwargs['allow_unlabeled'])
                    self.assertEqual(kwargs['expected_call_baseline'],baseline)
                    return copy.deepcopy(native_parts),{},[dict(path=str(native))]
                return copy.deepcopy(parts),hashes,sources
            with (patch('jevmahjong.training.call_head.snapshot',side_effect=snapshot),
                  patch('jevmahjong.evaluation.call_head_audit.snapshot',side_effect=snapshot),
                  patch('jevmahjong.training.call_head.load_sources',side_effect=inputs),
                  patch('jevmahjong.policy.call_baseline.load',return_value=context),
                  patch('jevmahjong.training.call_head.make_head',side_effect=lambda *a:FrozenHead()),
                  patch('jevmahjong.evaluation.call_head_audit.make_head',side_effect=lambda *a:FrozenHead())):
                fit(cfg);result=json.loads((root/'fit/result.json').read_text())
                self.assertGreater(result['selected']['epoch'],0)
                self.assertEqual(result['selected']['held_policy']['mean_sampled_hand_win_gain'],.25)
                self.assertEqual(result['selected']['held_policy']['mean_sampled_net_point_gain'],-1500.)
                saved=torch.load(result['selected']['checkpoint'],weights_only=True)
                self.assertEqual(saved['call_context_baseline'],baseline)
                self.assertFalse(torch.count_nonzero(saved['heads']['ordinary']['offset']))
                self.assertEqual(digest(checkpoint),baseline['sha256'])
                checkcfg=dict(**roles,out=str(root/'audit'),fit_config=str(fitfile),fit_config_sha256=digest(fitfile),
                    expected=dict(train_states=1,held_states=1,train_pairs=1,held_pairs=1,anchor_states=int(mode=='calling_baseline_kl')),
                    minimum_held_closed_chi_recoveries=0,require_route_recovery_for_development=False,
                    call_context_baseline=baseline)
                checked=audit(checkcfg)
                self.assertTrue(checked['eligible_for_development'])
                self.assertEqual(checked['distillation_target'],target_mode)
                self.assertEqual(checked['target_losses_recomputed'],len(result['curve']))
                self.assertEqual(checked['win_reward_arms_recomputed'],16)
                self.assertEqual(checked['anchor_policy'],mode)
                self.assertEqual(checked['trainable_parameter_count'],2)
                if mode=='calling_baseline_kl':
                    loaded=load_inputs(cfg,snapshot(cfg));anchors=loaded[4]
                    self.assertEqual(len(anchors),1)
                    self.assertEqual(loaded[5]['excluded_reward_labelled_states'],1)
                    self.assertNotIn('training_credit',anchors[0])
                    self.assertIn('fixed_call_context_logits',anchors[0])
                    native_parts['train'][1]['episode_key']=(103,0,7)
                    with self.assertRaisesRegex(ValueError,'held validation wall'):load_inputs(cfg,snapshot(cfg))
                    native_parts['train'][1]['episode_key']=(100,0,7)
                    (native/'protocol.json').write_text(json.dumps({'collection_method':'paired_future_ev_v1'}))
                    with self.assertRaisesRegex(ValueError,'audited native calling source'):load_inputs(cfg,snapshot(cfg))
                    (native/'protocol.json').write_text(json.dumps({'collection_method':'native_call_states_v1'}))
                if target_mode=='win_compatible_v1':
                    bad=copy.deepcopy(result);bad['curve'][1]['held_loss']+=.01
                    (root/'fit/result.json').write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError,'Distillation target loss differs'):
                        audit(dict(checkcfg,out=str(root/'wrong-target-loss')))
                    (root/'fit/result.json').write_text(json.dumps(result))
                wrong=torch.load(result['curve'][0]['checkpoint'],weights_only=True)
                wrong['heads']['calling']['offset'][0]+=.01
                path=Path(result['curve'][0]['checkpoint']);torch.save(wrong,path)
                result['curve'][0]['sha256']=digest(path)
                (root/'fit/result.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'Frozen ordinary head or reference initialization changed'):
                    audit(dict(checkcfg,out=str(root/'wrong-start')))
                counterpart.write_text(counterpart.read_text()+'\n')
                if mode=='same_data_no_anchors':
                    with self.assertRaisesRegex(ValueError,'Matched context configuration changed'):check_matched_context(cfg)


if __name__=='__main__':unittest.main()
