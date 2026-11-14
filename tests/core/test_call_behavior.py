"""Exploration inputs bind the acting call head and retain ordinary/kan policy."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from jevmahjong.core.io import digest
from jevmahjong.evaluation.native_state_audit import audit
from jevmahjong.training.call_behavior import collection_mode, load_collection_heads
from tests.core.test_call_states import call_fixture


class SmallPointer(torch.nn.Module):
    """Small controlled scorer; core tests do not download upstream models."""
    def __init__(self, d, dp):
        super().__init__(); self.q = torch.nn.Linear(d, dp); self.k = torch.nn.Linear(d, dp)

    def forward(self, decide, options):
        return self.k(options) @ self.q(decide) / self.q.out_features ** .5


def small_head(meta, device):
    dp, d = meta['head']['q.weight'].shape
    head = SmallPointer(d, dp).to(device); head.load_state_dict(meta['head']); return head


def models(root):
    torch.manual_seed(41)
    files = {}; states = {}
    for scope, role in [('ordinary', 'discard'), ('calling', 'call')]:
        path = root / role / 'head.pt'; path.parent.mkdir()
        head = SmallPointer(4, 3)
        states[scope] = copy.deepcopy(head.state_dict())
        torch.save(dict(head=states[scope], temperature=1., head_residual_dim=0), path)
        files[role] = {str(path): digest(path)}
    saved = dict(base_and_LoRA_frozen=True, model_files=files, trained_scopes=['calling'],
                 decision_filter='all', heads=copy.deepcopy(states))
    saved['heads']['calling']['q.weight'] *= -3
    return files, saved


def register(root, files, saved):
    path = root / 'behavior.pt'; torch.save(saved, path)
    receipt = dict(checkpoint=str(path), sha256=digest(path))
    cfg = dict(call_collection_policy='calling_only_pointer', behavior_heads=receipt,
               train_scopes=['calling'], decision_filter='all')
    protocol = dict(config=cfg, model_files=files, behavior_heads=receipt, code_sha256={})
    return cfg, protocol


class CallBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('jevmahjong.training.call_behavior.make_head', side_effect=small_head))

    def test_registered_call_changes_probabilities_but_ordinary_and_kan_head_stay_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); files, saved = models(root); cfg, protocol = register(root, files, saved)
            heads = load_collection_heads(cfg, protocol)
            original = saved['heads']['ordinary']
            self.assertTrue(all(torch.equal(v, original[k]) for k, v in heads['ordinary'].state_dict().items()))
            self.assertTrue(all(not p.requires_grad for h in heads.values() for p in h.parameters()))
            self.assertIsNone(load_collection_heads({}, dict(model_files=files)))
            with self.assertRaises(ValueError):
                collection_mode(dict(behavior_heads=cfg['behavior_heads']))
            for change in ['ordinary', 'scope', 'filter', 'encoder', 'nan', 'shape']:
                bad = copy.deepcopy(saved)
                if change == 'ordinary': bad['heads']['ordinary']['q.bias'] += 1
                if change == 'scope': bad['trained_scopes'] = ['ordinary', 'calling']
                if change == 'filter': bad['decision_filter'] = 'open_tenpai_riichi'
                if change == 'encoder': bad['model_files'] = {}
                if change == 'nan': bad['heads']['calling']['q.bias'][0] = float('nan')
                if change == 'shape': bad['heads']['calling']['q.bias'] = torch.zeros(9)
                c, p = register(root, files, bad)
                with self.subTest(change=change), self.assertRaises(ValueError): load_collection_heads(c, p)
            cfg, protocol = register(root, files, saved)
            torch.save({}, root / 'behavior.pt')
            with self.assertRaisesRegex(ValueError, 'receipt'): load_collection_heads(cfg, protocol)

    def test_native_audit_recomputes_exploration_probabilities_and_binds_its_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); files, saved = models(root); cfg, protocol = register(root, files, saved)
            cfg.update(training_seed_start=10, training_seeds_per_stage=1,
                       game_mode='4p-red-east', seed_scheme='spaced-v1')
            source = root / 'source'; (source / 'walls-private').mkdir(parents=True)
            from jevmahjong.data.native_states import CALL_METHOD
            protocol['collection_method'] = CALL_METHOD
            (source / 'protocol.json').write_text(json.dumps(protocol))
            payload = call_fixture(); row = payload['rows'][0]
            row['decide'] = torch.ones(4)
            row['options'] = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
            head = load_collection_heads(cfg, protocol)['calling']
            row['reference'] = head(row['decide'], row['options']).detach().softmax(-1)
            row['teacher_action'] = int(row['reference'].argmax())
            row['action'] = next(i for i in row['eligible'] if i != row['teacher_action'])
            shard = source / 'walls-private/10.pt'; torch.save(payload, shard)
            shard.with_suffix('.audit.json').write_text(json.dumps(dict(sha256=digest(shard), rows=1, controls=4)))
            (source / 'collection-result.json').write_text(json.dumps(dict(status='collection_complete',
                method=CALL_METHOD, walls=1, rows=1, state_splits=dict(train=1, held=0))))
            auditcfg = dict(collection=str(source), out=str(root / 'audit'),
                            discard_checkpoint=str(root / 'discard'), call_checkpoint=str(root / 'call'))
            with (patch('jevmahjong.evaluation.native_state_audit.snapshot', return_value=dict(
                    config=auditcfg, model_files=files, code_sha256={})),
                  patch('jevmahjong.evaluation.native_state_audit.RiichiEnv', return_value=SimpleNamespace(mjai_log=[])),
                  patch('jevmahjong.evaluation.native_state_audit.replay_prefix', return_value={0: 'observation'}),
                  patch('jevmahjong.evaluation.native_state_audit.build_request', return_value=(
                      row['public_request'], row['action_keys'], [None] * 3)),
                  patch('jevmahjong.evaluation.native_state_audit.action_key', side_effect=lambda x: x),
                  patch('jevmahjong.evaluation.native_state_audit.guarded_indices', return_value=[0, 1, 2]),
                  patch('jevmahjong.evaluation.native_state_audit.call_observation_digest', return_value='digest'),
                  patch('jevmahjong.evaluation.native_state_audit.hand_outcomes', return_value=[dict(draws=5)])):
                result = audit(auditcfg)
                self.assertEqual(result['status'], 'passed_complete')
                self.assertEqual(result['cached_head_probability_max_error'], 0.)
                self.assertEqual(result['behavior_checkpoint_sha256'], cfg['behavior_heads']['sha256'])
                receipt = json.loads((root / 'audit/protocol.json').read_text())
                self.assertEqual(receipt['source_files'][str(root / 'behavior.pt')], digest(root / 'behavior.pt'))
                # A baseline distribution cannot silently masquerade as the acting exploration head.
                row['reference'] = torch.tensor([.6, .3, .1]); row['teacher_action'] = 0; row['action'] = 1
                torch.save(payload, shard)
                shard.with_suffix('.audit.json').write_text(json.dumps(dict(sha256=digest(shard), rows=1, controls=4)))
                auditcfg['out'] = str(root / 'wrong-audit')
                with self.assertRaisesRegex(ValueError, 'cached model probabilities'): audit(auditcfg)


if __name__ == '__main__': unittest.main()
