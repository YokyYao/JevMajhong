import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from jevmahjong.training.call_head import complete_groups, anchor_kl, distillation_loss, profile, route_profile, policy_improved, fit
from jevmahjong.evaluation.call_head_audit import manual_profile, manual_route_profile, check_close, checked_receipt
from jevmahjong.core.io import digest


def row(action, utility, points):
    return dict(episode_key=(100,0,0,action),request_sha256='same',eligible=[0,1,2],teacher_action=0,
        public_request={'questions':{'action':{'criteria':{'a0':{'type':'none'},'a1':{'type':'chi'},'a2':{'type':'chi'}}}}},
        decide=torch.zeros(1),options=torch.tensor([2.,0.,-1.]),reference=torch.tensor([2.,0.,-1.]).softmax(-1),
        action=action,training_advantage=utility,training_credit={
            'control':dict(utility=0.,net_points=0.,fast_reward=0.,dealin=0.),
            'branch':dict(utility=utility,net_points=points,fast_reward=0.,dealin=0.)})


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.offset=torch.nn.Parameter(torch.zeros(3))
    def forward(self,decide,options): return options+self.offset


class CallHeadTests(unittest.TestCase):
    def test_audit_receipt_rechecks_real_files_and_binds_the_saved_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'fit';run.mkdir();aud=root/'audit';aud.mkdir()
            fitcfg={'out':str(run)};cfgfile=root/'fit.json';cfgfile.write_text(json.dumps(fitcfg))
            fit={'config':fitcfg,'selected':{'sha256':'0'*64}};fitfile=run/'result.json';fitfile.write_text(json.dumps(fit))
            cfg={'fit_config':str(cfgfile),'fit_config_sha256':digest(cfgfile),'out':str(aud)}
            auditcfg=root/'audit.json';auditcfg.write_text(json.dumps(cfg))
            (aud/'result.json').write_text(json.dumps({'status':'passed_complete','selected_sha256':'0'*64}))
            (aud/'protocol.json').write_text(json.dumps({'config':cfg,'model_files':{},'code_sha256':{},
                'source_files':{str(fitfile):digest(fitfile)}}))
            _,receipt=checked_receipt(auditcfg,run,fit,{})
            self.assertEqual(len(receipt),3)
            for file,sha in receipt.items():self.assertEqual(digest(Path(file)),sha)
            fitfile.write_text('{}')
            with self.assertRaises(ValueError):checked_receipt(auditcfg,run,fit,{})

    def test_complete_menu_is_one_unit_and_rejects_missing_or_conflicting_alternative(self):
        rows=[row(1,.1,800.),row(2,-.2,-1600.)]
        self.assertEqual(len(complete_groups(rows)),1)
        for bad in [rows[:1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):complete_groups(bad)
        bad=copy.deepcopy(rows);bad[1]['training_credit']['control']['net_points']=1
        with self.assertRaises(ValueError):complete_groups(bad)

    def test_profile_uses_full_menu_and_actual_points_not_pair_count(self):
        rows=[row(1,.1,800.),row(2,-.2,-1600.)];groups=complete_groups(rows);head=Head()
        first=profile(head,groups)
        self.assertEqual(first['states'],1);self.assertEqual(first['mean_sampled_net_point_gain'],0)
        with torch.no_grad():head.offset[1]=3
        after=profile(head,groups)
        self.assertEqual(after['changed_argmax_states'],1)
        self.assertEqual(after['mean_sampled_net_point_gain'],800.)
        self.assertAlmostEqual(after['label_regret'],0.)
        groups[0]['records']=rows
        check_close(after,manual_profile(head,groups))
        check_close(route_profile(head,groups),manual_route_profile(head,groups))
        wrong=dict(after,mean_sampled_net_point_gain=1600.)
        with self.assertRaises(ValueError):check_close(wrong,manual_profile(head,groups))

    def test_gradient_can_teach_call_without_changing_labels_and_anchor_penalizes_drift(self):
        rows=[row(1,.5,4000.),row(2,-.2,-1600.)];g=complete_groups(rows)[0];head=Head()
        loss=distillation_loss(head,g,.05);loss.backward()
        self.assertLess(head.offset.grad[1],0);self.assertGreater(head.offset.grad[0],0)
        before=float(anchor_kl(head,rows[0]).detach())
        with torch.no_grad():head.offset[1]=3
        self.assertGreater(float(anchor_kl(head,rows[0]).detach()),before+.1)

    def test_loss_or_speed_alone_cannot_reset_choice_early_stop(self):
        base={'held_policy':{'mean_label_utility_gain':0.,'mean_sampled_net_point_gain':0.},'anchor_kl':0.}
        best=copy.deepcopy(base);candidate=copy.deepcopy(base);candidate['held_loss']=-100
        self.assertFalse(policy_improved(candidate,best,base,.05))
        candidate['held_policy']['mean_label_utility_gain']=.2
        self.assertFalse(policy_improved(candidate,best,base,.05))
        candidate['held_policy']['mean_sampled_net_point_gain']=200
        self.assertTrue(policy_improved(candidate,best,base,.05))
        candidate['anchor_kl']=.1
        self.assertFalse(policy_improved(candidate,best,base,.05))

    def test_win_objective_can_prefer_more_actual_wins_with_lower_points(self):
        p=dict(mean_sampled_hand_win_gain=0.,mean_sampled_fast_gain=0.,mean_label_utility_gain=0.,
            mean_sampled_net_point_gain=0.,mean_sampled_dealin_change=0.)
        base={'held_policy':p,'anchor_kl':0.}
        candidate={'held_policy':dict(p,mean_sampled_hand_win_gain=.02,mean_sampled_fast_gain=.01,
            mean_label_utility_gain=.0225,mean_sampled_net_point_gain=-100.),'anchor_kl':.02}
        self.assertTrue(policy_improved(candidate,base,base,.05,'hand_win',.01))
        self.assertFalse(policy_improved(candidate,base,base,.05,'net_points'))
        candidate['held_policy']['mean_sampled_hand_win_gain']=-.01
        candidate['held_policy']['mean_sampled_net_point_gain']=10000.
        self.assertFalse(policy_improved(candidate,base,base,.05,'hand_win',.01))

    def test_win_credit_full_menu_and_independent_profile_agree(self):
        rows=[row(1,.1,-500.),row(2,-.2,2000.)]
        for r,win in zip(rows,[.375,.1875]):
            r['training_credit']['control']['own_win']=.25
            r['training_credit']['branch']['own_win']=win
        groups=complete_groups(rows);groups[0]['records']=rows;head=Head()
        with torch.no_grad():head.offset[1]=3
        actual=profile(head,groups)
        self.assertEqual(actual['mean_sampled_hand_win_gain'],.125)
        self.assertEqual(actual['mean_sampled_net_point_gain'],-500.)
        check_close(actual,manual_profile(head,groups))
        check_close(route_profile(head,groups),manual_route_profile(head,groups))

    def test_fit_saves_actual_choice_gain_or_keeps_reference_and_stops(self):
        # Exercise the optimizer, checkpoint selection and early stop without
        # downloading a language model. Source/audit I/O is tested separately.
        for useful in (True,False):
            with self.subTest(useful=useful), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for role in ('discard','call'):
                    (root/role).mkdir();torch.save({},root/role/'head.pt')
                rows=[row(1,.5 if useful else -.5,4000. if useful else -4000.),row(2,-.2,-1600.)]
                groups={'train':complete_groups(rows),'held':complete_groups(rows)}
                cfg=dict(out=str(root/'fit'),model_id='toy-call',seed=7,train_scopes=['calling'],decision_filter='all',
                    fit_model='complete_option_call_head',discard_checkpoint=str(root/'discard'),call_checkpoint=str(root/'call'),
                    anchor_beta=.001,anchor_kl_limit=100.,reference_prior_strength=.05,lr=.5,weight_decay=0.,epochs=12,
                    patience_epochs=4,batch=1,anchor_batch=1,reference_anchor_states=1,data=['mocked'])
                loaded=({'train':rows,'held':rows},groups,{},[],[rows[0]],{'source_sha256':{}},{})
                with patch('jevmahjong.training.call_head.snapshot',return_value={'config':cfg,'model_files':{}}), \
                     patch('jevmahjong.training.call_head.verify'), \
                     patch('jevmahjong.training.call_head.load_inputs',return_value=loaded), \
                     patch('jevmahjong.training.call_head.make_head',side_effect=lambda *a:Head()), \
                     patch('jevmahjong.training.call_head.route_groups',side_effect=lambda gs:{'closed_chi_tanyao_gain':gs}):
                    fit(cfg)
                result=json.loads((root/'fit/result.json').read_text())
                selected=torch.load(result['selected']['checkpoint'],weights_only=True)
                self.assertFalse(torch.count_nonzero(selected['heads']['ordinary']['offset']))
                self.assertEqual(result['held_policy_gain_improved'],useful)
                self.assertEqual(result['selected_is_starting_model'],not useful)
                self.assertLess(len(result['curve']),cfg['epochs']+1)
                self.assertEqual(result['stop_reason'],'held_actual_choices_stalled')
                if useful:self.assertGreater(result['selected']['held_policy']['mean_sampled_net_point_gain'],0)


if __name__=='__main__':unittest.main()
