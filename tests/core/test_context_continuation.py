"""Continuation updates one existing context without applying the old context twice."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from jevmahjong.core.io import digest
from jevmahjong.policy.call_baseline import replaces_calling_baseline,transform
from jevmahjong.policy.call_tile_context import VERSION
from jevmahjong.policy.kan_context import feature_names,probabilities
from jevmahjong.training.context_initialization import initial_context
from jevmahjong.training.context_network import make_network
from jevmahjong.training.context_policy import pack_options,policy_logits
from tests.core.test_call_behavior import models
from tests.core.test_win_context import pair


class ContextContinuationTests(unittest.TestCase):
    def test_explicit_identity_and_direct_single_network_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);files,saved=models(root)
            for scope,role in [('ordinary','discard'),('calling','call')]:
                saved['heads'][scope]=torch.load(root/role/'head.pt',weights_only=True)['head']
            torch.manual_seed(39);net=make_network(4,len(feature_names(VERSION)))
            with torch.no_grad():net[2].weight.fill_(.002)
            context=dict(version=VERSION,names=list(feature_names(VERSION)),model_kind='mlp',
                hidden=4,state_dict=copy.deepcopy(net.state_dict()),reference_prior_strength=.05)
            saved.update(decision_filter='call_windows',kan_context=context)
            path=root/'baseline.pt';torch.save(saved,path)
            cfg=dict(train_scopes=['calling'],decision_filter='all',feature_version=VERSION,hidden=4,
                reference_prior_strength=.05,context_initialization='calling_baseline',
                call_context_baseline=dict(checkpoint=str(path),sha256=digest(path)))
            loaded=initial_context(cfg,files);net.load_state_dict(loaded['state_dict'])
            row=pair(100);row['original_reference']=row['reference'].clone()
            row['reference']=transform(row['public_request'],row['reference'],loaded)
            group=pack_options([row],feature_version=VERSION)[0]
            start=policy_logits(net,group,.05).softmax(-1)
            self.assertTrue(torch.allclose(start,row['reference'].double(),atol=1e-7,rtol=0))
            with torch.no_grad():net[2].weight.add_(.03)
            updated=dict(loaded,state_dict=net.state_dict())
            direct=torch.tensor(probabilities(row['public_request'],row['original_reference'].tolist(),updated))
            fitted=policy_logits(net,group,.05).softmax(-1)
            self.assertTrue(torch.allclose(direct.double(),fitted,atol=1e-7,rtol=0))
            double_applied=torch.tensor(probabilities(row['public_request'],row['reference'].tolist(),updated))
            self.assertFalse(torch.allclose(direct,double_applied,atol=1e-7,rtol=0))
            metadata=dict(saved,context_initialization='calling_baseline',
                call_context_baseline=cfg['call_context_baseline'],context_replaces_calling_baseline=True)
            self.assertTrue(replaces_calling_baseline(metadata))
            for key,value in [('call_context_baseline',None),('context_initialization','zero_output'),
                    ('trained_scopes',['ordinary']),('context_replaces_calling_baseline','true')]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    replaces_calling_baseline(dict(metadata,**{key:value}))
            for change in (dict(hidden=8),dict(reference_prior_strength=.1),
                    dict(context_initialization='zero_output'),dict(call_context_baseline=None)):
                with self.subTest(change=change),self.assertRaises(ValueError):initial_context(dict(cfg,**change),files)


if __name__=='__main__':unittest.main()
