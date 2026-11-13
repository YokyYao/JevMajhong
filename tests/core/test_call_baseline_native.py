"""A pure-call source must audit the policy that actually generated its actions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from jevmahjong.core.io import digest,request_hash
from jevmahjong.data.native_states import CALL_METHOD
from jevmahjong.evaluation.native_state_audit import audit
from jevmahjong.policy.call_baseline import load,transform
from jevmahjong.policy.kan_context import feature_names
from jevmahjong.policy.kan_progress import VERSION
from jevmahjong.training.call_behavior import collection_mode
from tests.core.test_call_behavior import models,small_head
from tests.core.test_call_states import call_fixture
from tests.core.test_kan_context import request


class CallingBaselineNativeTests(unittest.TestCase):
    def test_audit_replays_stronger_call_distribution_and_rejects_raw_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);files,saved=models(root)
            for scope,role in [('ordinary','discard'),('calling','call')]:
                saved['heads'][scope]=torch.load(root/role/'head.pt',weights_only=True)['head']
            names=feature_names(VERSION);coef=torch.zeros(len(names));coef[names.index('pon')]=.5
            saved.update(decision_filter='call_windows',kan_context=dict(version=VERSION,
                names=list(names),coefficients=coef,reference_prior_strength=.05))
            checkpoint=root/'baseline.pt';torch.save(saved,checkpoint)
            spec=dict(checkpoint=str(checkpoint),sha256=digest(checkpoint))
            cfg=dict(train_scopes=['calling'],decision_filter='all',call_context_baseline=spec,
                training_seed_start=10,training_seeds_per_stage=1,game_mode='4p-red-east',seed_scheme='spaced-v1')
            source=root/'source';(source/'walls-private').mkdir(parents=True)
            protocol=dict(config=cfg,model_files=files,code_sha256={},call_context_baseline=spec,collection_method=CALL_METHOD)
            (source/'protocol.json').write_text(json.dumps(protocol))
            payload=call_fixture();row=payload['rows'][0];req=request('daiminkan')
            req['questions']['action']['criteria']['a1']=dict(type='pon',pai='E',consumed=['E','E'])
            row.update(public_request=req,request_sha256=request_hash(req),eligible=[0,1],
                action_keys=['pass','pon'],decide=torch.ones(4),options=torch.eye(4)[:2])
            head=small_head(torch.load(root/'call/head.pt',weights_only=True),'cpu')
            raw=head(row['decide'],row['options']).detach().softmax(-1)
            row['reference']=transform(req,raw,load(cfg,files))
            self.assertFalse(torch.allclose(raw,row['reference']))
            row['teacher_action']=int(row['reference'].argmax());row['action']=1-row['teacher_action']
            shard=source/'walls-private/10.pt'
            def save():
                torch.save(payload,shard)
                shard.with_suffix('.audit.json').write_text(json.dumps(dict(sha256=digest(shard),rows=1,controls=4)))
            save()
            (source/'collection-result.json').write_text(json.dumps(dict(status='collection_complete',
                method=CALL_METHOD,walls=1,rows=1,state_splits=dict(train=1,held=0))))
            auditcfg=dict(cfg,collection=str(source),out=str(root/'audit'),
                discard_checkpoint=str(root/'discard'),call_checkpoint=str(root/'call'))
            with (patch('jevmahjong.evaluation.native_state_audit.snapshot',side_effect=lambda c:dict(config=c,model_files=files,code_sha256={})),
                  patch('jevmahjong.evaluation.native_state_audit.make_head',side_effect=small_head),
                  patch('jevmahjong.evaluation.native_state_audit.RiichiEnv',return_value=SimpleNamespace(mjai_log=[])),
                  patch('jevmahjong.evaluation.native_state_audit.replay_prefix',return_value={0:'observation'}),
                  patch('jevmahjong.evaluation.native_state_audit.build_request',return_value=(req,row['action_keys'],[None]*2)),
                  patch('jevmahjong.evaluation.native_state_audit.action_key',side_effect=lambda x:x),
                  patch('jevmahjong.evaluation.native_state_audit.guarded_indices',return_value=[0,1]),
                  patch('jevmahjong.evaluation.native_state_audit.call_observation_digest',return_value='digest'),
                  patch('jevmahjong.evaluation.native_state_audit.hand_outcomes',return_value=[dict(draws=5)])):
                result=audit(auditcfg)
                self.assertEqual(result['status'],'passed_complete')
                self.assertEqual(result['calling_baseline_sha256'],spec['sha256'])
                row.update(reference=raw,teacher_action=int(raw.argmax()),action=1-int(raw.argmax()));save()
                with self.assertRaisesRegex(ValueError,'cached model probabilities'):
                    audit(dict(auditcfg,out=str(root/'wrong-audit')))

    def test_no_second_collection_policy_can_override_the_pinned_baseline(self):
        cfg=dict(call_collection_policy='calling_only_pointer',behavior_heads={'checkpoint':'head.pt'},
            call_context_baseline={'checkpoint':'context.pt'},train_scopes=['calling'],decision_filter='all')
        with self.assertRaisesRegex(ValueError,'cannot also use'):collection_mode(cfg)


if __name__=='__main__':unittest.main()
