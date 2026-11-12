"""Audit rejects missing actions, mismatched profiles and changed fit identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from jevmahjong.core.io import digest
from jevmahjong.evaluation.context_audit import aggregate,checked_receipt,check_profiles,held_groups,METRICS


class ContextAuditTest(unittest.TestCase):
    def test_state_aggregation_preserves_net_point_and_dealin_direction(self):
        first={k:0. for k in METRICS};first.update(mean_sampled_net_point_gain_per_state=100.,
            mean_sampled_dealin_change_per_state=.1,changed_argmax_states=1)
        other={k:0. for k in METRICS};other.update(mean_sampled_net_point_gain_per_state=-20.,
            mean_sampled_dealin_change_per_state=-.3,changed_argmax_states=0)
        result=aggregate([first,other])
        self.assertEqual(result['states'],2);self.assertEqual(result['mean_sampled_net_point_gain_per_state'],40.)
        self.assertAlmostEqual(result['mean_sampled_dealin_change_per_state'],-.1)
        self.assertEqual(result['changed_argmax_states'],1)
        with self.assertRaises(ValueError):aggregate([])

    def test_recomputed_selection_rejects_count_and_metric_changes(self):
        actual={k:0. for k in METRICS};actual.update(states=2,changed_argmax_states=1)
        self.assertEqual(check_profiles(actual,actual),0.)
        for key in ['states','changed_argmax_states','label_regret','policy_kl','mean_label_utility_gain_vs_play']:
            changed=dict(actual);changed[key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):check_profiles(actual,changed)

    def test_complete_option_audit_reconstructs_teacher_and_targets(self):
        base=dict(episode_key=(100,0,0),request_sha256='same',teacher_action=0,eligible=[0,1,2],
            public_request={'opaque':'public'},reference=torch.tensor([.6,.3,.1]))
        def pair(action,value):
            return dict(base,action=action,training_advantage=value,
                training_credit={'control':{'utility':0.,'net_points':0.,'dealin':0.,'fast_reward':0.},
                    'branch':{'utility':value,'net_points':value*8000,'dealin':0.,'fast_reward':0.}})
        rows=[pair(1,.2),pair(2,-.1)];sources=[{'path':'source','seed_start':100,'seeds':4}]
        with patch('jevmahjong.evaluation.context_audit.feature_vectors',return_value=[[0.],[1.],[2.]]):
            groups=held_groups(rows,sources,'test',.05)
            self.assertEqual(len(groups),1)
            self.assertAlmostEqual(groups[0][4][1].item(),.2)
            self.assertEqual(groups[0][6][1]['net_points'],1600.)
            for broken in [rows[:1],[rows[0],rows[0]], [dict(rows[0],teacher_action=1),rows[1]]]:
                with self.assertRaises(ValueError):held_groups(broken,sources,'test',.05)

    def test_development_receipt_rejects_changed_fit_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audit=root/'audit';audit.mkdir();fitroot=root/'fit';fitroot.mkdir()
            raw=root/'raw';raw.write_bytes(b'fixed inputs');fitfile=fitroot/'result.json';fitfile.write_text('{}')
            cfg={'out':str(audit)};configfile=root/'audit-config.json';configfile.write_text(json.dumps(cfg))
            fit={'model_id':'test','selected':{'sha256':'fixed'},'decision_filter':'defense_windows'}
            model_files={};protocol={'config':cfg,'model_files':model_files,'code_sha256':{},
                'source_files':{str(fitfile):digest(fitfile),str(raw):digest(raw)}}
            result={'status':'passed_complete','model_id':'test','checkpoint_sha256':'fixed',
                'fit_result_sha256':digest(fitfile),'activation':'defense_windows','frozen_original_heads':True,
                'whole_wall_disjoint':True,'eligible_for_development':True}
            (audit/'protocol.json').write_text(json.dumps(protocol));(audit/'result.json').write_text(json.dumps(result))
            checked,receipt=checked_receipt(configfile,fitroot,fit,model_files)
            self.assertTrue(checked['eligible_for_development']);self.assertEqual(len(receipt),3)
            wrong=copy.deepcopy(fit);wrong['selected']['sha256']='changed'
            with self.assertRaisesRegex(ValueError,'identity'):checked_receipt(configfile,fitroot,wrong,model_files)
            raw.write_bytes(b'changed inputs')
            with self.assertRaisesRegex(ValueError,'input changed'):checked_receipt(configfile,fitroot,fit,model_files)


if __name__=='__main__':unittest.main()
