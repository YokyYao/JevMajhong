import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from incremental_score import (IncrementalFrozenTeacherPolicy,ORIGINAL_HEAD,SELECTION_POLICY,
                               choose_positive,confirmation_config,incremental_gate,select_incremental)
from head_rl import summarize


class IncrementalScoreTests(unittest.TestCase):
    def test_small_positive_and_earlier_best_are_preserved(self):
        entry=lambda u,m:dict(update=u,metrics={'mean_net_points':m})
        self.assertEqual(choose_positive([entry(4,101.)],100.)['update'],4)
        self.assertEqual(choose_positive([entry(4,120.),entry(8,118.),entry(12,120.)],100.)['update'],4)
        self.assertIsNone(choose_positive([entry(4,100.),entry(8,99.)],100.))

    def test_continuation_ignores_desired_gain_but_keeps_regression_and_bounds(self):
        config=dict(desired_net_points_gain=500,net_points_drop_stop=500,max_validation_checks=4)
        base={'mean_net_points':100.};positive={'mean_net_points':101.}
        uncertain={'primary_delta_ci95':[-20.,22.]}
        self.assertEqual(incremental_gate(2,positive,base,config,uncertain)[0],'continue')
        self.assertEqual(incremental_gate(4,positive,base,config,uncertain)[0],'stop_bounded_data_phase')
        self.assertEqual(incremental_gate(1,{'mean_net_points':-500.},base,config,uncertain)[0],'stop_regression')
        self.assertEqual(incremental_gate(1,{'mean_net_points':99.},base,config,
                                         {'primary_delta_ci95':[-5.,-.1]})[0],'stop_regression')

    def test_positive_confirmation_rule_is_explicit_not_retroactive(self):
        config=dict(selection_policy=SELECTION_POLICY,confirmation_qualification='positive_paired_ci95',
                    confirmation_min_ci95_lower=0,confirmation_seeds=128,game_mode='4p-red-east',
                    reward_mode='net_points',seed_scheme='spaced-v1',final_benchmark_head_sha256=ORIGINAL_HEAD,
                    final_benchmark_reference='original2000',desired_net_points_gain=500)
        frozen=confirmation_config(config)
        self.assertEqual(frozen['target_net_points_gain'],0.)
        self.assertEqual(frozen['desired_net_points_gain'],500)
        self.assertEqual(frozen['reference'],'original2000')
        self.assertNotIn('target_net_points_gain',config)
        for key,value in [('selection_policy','historic_gate'),('confirmation_seeds',32),
                          ('final_benchmark_head_sha256','another_model')]:
            with self.assertRaises(ValueError):confirmation_config({**config,key:value})

    def test_multibatch_constructor_keeps_the_same_frozen_teacher(self):
        cfg=dict(training_method='point_delta_weighted_reference_relative_pairwise',
                 advantage_estimator='greedy_reference',reward_mode='net_points',max_stages=2,
                 teacher_control_stationary_across_stages=True,reference='example',initial_reference_head_sha256='pinned')
        def initialize(policy,config):
            policy.config=config;policy.head=torch.nn.Linear(2,1)
            policy.reference=copy.deepcopy(policy.head).requires_grad_(False)
        with patch('incremental_score.fingerprint',return_value='pinned'),patch('incremental_score.HeadPolicy.__init__',initialize):
            policy=IncrementalFrozenTeacherPolicy(cfg)
            self.assertEqual(policy.config['max_stages'],2)
            self.assertFalse(policy.controls)
            self.assertTrue(all(not p.requires_grad for p in policy.reference.parameters()))
        with patch('incremental_score.fingerprint',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'initialization changed'):IncrementalFrozenTeacherPolicy(cfg)

    def test_selection_recomputes_complete_native_records_and_keeps_small_gain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'run';source.mkdir()
            config=dict(development_seed_start=100,development_seeds=2,reward_scale=1000.,
                        selection_policy=SELECTION_POLICY)
            records=[]
            def matches(net):
                return [dict(seed=s,engine_seed=s<<32,seed_scheme='spaced-v1',seat=p,rank=2,score=25000+net,
                             game_mode='4p-red-east',reward_mode='net_points',reward_scale=1000.,reward=net/1000.,
                             sampling_seed=None,trainable_decisions=0,old_opponent=False)
                        for s in [100,101] for p in range(4)]
            for update,net in [(0,100),(4,101),(8,99)]:
                d=source/('development-baseline' if not update else f'development-update-{update:06d}');d.mkdir()
                rows=matches(net);(d/'games.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                if update:
                    cp=source/'checkpoints'/f'update-{update:06d}';cp.mkdir(parents=True)
                    (cp/'head.pt').write_bytes(str(update).encode())
                    (cp/'complete.json').write_text(json.dumps(dict(update=update,head_sha256=hashlib.sha256((cp/'head.pt').read_bytes()).hexdigest())))
                    records.append(dict(update=update,metrics=summarize(rows),checkpoint=str(cp)))
            protocol=dict(config=config,sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),code_sha256={})
            (source/'protocol.json').write_text(json.dumps(protocol))
            (source/'development-results.json').write_text(json.dumps(records))
            (source/'pilot-analysis.json').write_text(json.dumps(dict(status='pilot_complete',development=records)))
            result=select_incremental(source)
            self.assertEqual(result['selected']['update'],4)
            self.assertFalse(result['fixed_gain_gate_enabled'])
            records[0]['metrics']['mean_net_points']=10000
            (source/'development-results.json').write_text(json.dumps(records))
            (source/'pilot-analysis.json').write_text(json.dumps(dict(status='pilot_complete',development=records)))
            with self.assertRaisesRegex(ValueError,'summary differs'):select_incremental(source)


if __name__=='__main__':unittest.main()
